// ===========================================================================
// returns.wgsl - Pass 1: on-chip two-state Markov switching skew-t returns.
// ===========================================================================
//
// One thread per (batch simulation); each thread walks its full month
// series sequentially because the active market state is a Markov chain
// (state_t depends on state_{t-1}). Across simulations the threads are
// independent and run in parallel. The Threefry counter uses the GLOBAL
// simulation index (params.generate.w + local batch index) so consecutive
// batches draw distinct, reproducible streams; within a simulation the
// counter advances with the month, exactly as before.
//
// Each month we draw:
//   - |N(0,1)| for the skew component                    (t = 0)
//   - a chi-square_nu component from nu squared normals (t = 1..nu)
//   - three independent normals driving the underlying funds (t = nu+1..nu+3)
//   - the Markov transition uniform from the otherwise-unused slot of pair 4
// then the regime state is advanced and the return is sampled from that
// state's own skew-t parameters (xi/omega/Cholesky for state 0 or 1).
//
// The state path is per simulation: month 0 starts from the stationary prior
// p(0) = (1-p11)/(2-p00-p11); every later month, if currently in state s the
// chain stays in s with probability p_ss (p00 / p11) and flips otherwise.

@compute @workgroup_size(64, 1, 1)
fn generate_returns(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sim_in_batch = gid.x;
    let batch_sims = params.generate.z;
    if (sim_in_batch >= batch_sims) {
        return;
    }
    let global_sim = params.generate.w + sim_in_batch;
    let total_months = params.dimensions.z;
    let nu = params.generate.y;
    let seed = params.generate.x;
    let p00 = markov_prob(0u);
    let p11 = markov_prob(1u);
    let prior0 = markov_prior0();

    var state: u32 = 0u;
    for (var month = 0u; month < total_months; month += 1u) {
        let counter = global_sim * total_months + month;
        // Linear normal stream t = 0..: t=0 skew, t=1..nu chi-square
        // components, t=nu+1..nu+3 the three independent normals driving the
        // underlying funds. Modes 3/4 (params.dispatch.y >= 3) instead draw
        // the chi2 scale directly from the second uniform of pair 0 via the
        // LUT. The pair-4 second slot (t = 2*4+1 = 9 at nu = 5) is otherwise
        // unused and carries the Markov transition uniform.
        let direct_w = params.dispatch.y >= 3u;
        let pairs_needed = (nu + 5u) / 2u;
        var scale = 0.0;
        var skew = 0.0;
        var fund_normal = vec3<f32>(0.0);
        var trans = 0.5;
        for (var pair = 0u; pair < pairs_needed; pair += 1u) {
            var u0 = 0.0;
            var u1 = 0.0;
            if (params.dispatch.y != 0u) {
                let coord = month * 10u + pair * 2u;
                u0 = rqmc_draw(global_sim, coord, seed, params.dispatch.y);
                u1 = rqmc_draw(global_sim, coord + 1u, seed, params.dispatch.y);
            } else {
                let uniforms = threefry_uniforms(counter, pair, seed);
                u0 = uniforms.x;
                u1 = uniforms.y;
            }
            let pair_normal = box_muller(u0, u1);
            if (direct_w) {
                if (pair == 0u) {
                    skew = abs(pair_normal.x);
                    scale = chi2_inv_cdf(u1, (10u * params.dimensions.z + career_years()) * rqmc_direction_bits());
                } else if (pair == 3u) {
                    fund_normal.x = pair_normal.x;
                    fund_normal.y = pair_normal.y;
                } else if (pair == 4u) {
                    fund_normal.z = pair_normal.x;
                    trans = u1;
                }
                continue;
            }
            let t0 = pair * 2u;
            let t1 = pair * 2u + 1u;
            if (t0 == 0u) {
                skew = abs(pair_normal.x);
            } else if (t0 <= nu) {
                scale += pair_normal.x * pair_normal.x;
            } else if (t0 == nu + 1u) {
                fund_normal.x = pair_normal.x;
            } else if (t0 == nu + 2u) {
                fund_normal.y = pair_normal.x;
            } else if (t0 == nu + 3u) {
                fund_normal.z = pair_normal.x;
            }
            if (t1 <= nu) {
                scale += pair_normal.y * pair_normal.y;
            } else if (t1 == nu + 1u) {
                fund_normal.x = pair_normal.y;
            } else if (t1 == nu + 2u) {
                fund_normal.y = pair_normal.y;
            } else if (t1 == nu + 3u) {
                fund_normal.z = pair_normal.y;
            }
            if (pair == 4u) {
                trans = u1;
            }
        }

        // Advance the Markov chain. Month 0 uses the stationary prior; later
        // months persist in the current state with probability p_ss.
        if (month == 0u) {
            state = select(1u, 0u, trans < prior0);
        } else if (state == 0u) {
            state = select(1u, 0u, trans < p00);
        } else {
            state = select(0u, 1u, trans < p11);
        }

        // scale = chi_square_nu / nu, matching the reference gamma(nu/2, 2/nu).
        let inv_scale = 1.0 / sqrt(max(scale / f32(nu), 1e-12));
        let xi0 = return_markov_at(state, 0u);
        let xi1 = return_markov_at(state, 1u);
        let xi2 = return_markov_at(state, 2u);
        let omega0 = return_markov_at(state, 3u);
        let omega1 = return_markov_at(state, 4u);
        let omega2 = return_markov_at(state, 5u);
        let delta0 = return_markov_at(state, 6u);
        let delta1 = return_markov_at(state, 7u);
        let delta2 = return_markov_at(state, 8u);
        // Row-major 3x3 Cholesky of the residual correlation.
        let l00 = return_markov_at(state, 9u);
        let l01 = return_markov_at(state, 10u);
        let l02 = return_markov_at(state, 11u);
        let l10 = return_markov_at(state, 12u);
        let l11 = return_markov_at(state, 13u);
        let l12 = return_markov_at(state, 14u);
        let l20 = return_markov_at(state, 15u);
        let l21 = return_markov_at(state, 16u);
        let l22 = return_markov_at(state, 17u);

        let veqt = max(-0.95, xi0 + omega0 * (delta0 * skew + fund_normal.x * l00 + fund_normal.y * l01 + fund_normal.z * l02) * inv_scale);
        let vgro = max(-0.95, xi1 + omega1 * (delta1 * skew + fund_normal.x * l10 + fund_normal.y * l11 + fund_normal.z * l12) * inv_scale);
        let vbal = max(-0.95, xi2 + omega2 * (delta2 * skew + fund_normal.x * l20 + fund_normal.y * l21 + fund_normal.z * l22) * inv_scale);

        let out_index = (sim_in_batch * total_months + month) * params.solver.y;
        scratch[out_index] = veqt;
        // Leverage-off runs never read funds 1/2 (no strategy references
        // them), so the transforms are skipped entirely when
        // params.dispatch.z is 0.
        if (params.dispatch.z != 0u) {
            let veqt15 = max(-0.95, 1.5 * veqt - 0.5 * params.generate1.x - params.generate1.y);
            let veqt2 = max(-0.95, 2.0 * veqt - params.generate1.x - params.generate1.z);
            scratch[out_index + 1u] = veqt15;
            scratch[out_index + 2u] = veqt2;
        }
        scratch[out_index + 3u] = vgro;
        scratch[out_index + 4u] = vbal;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: on-chip annual career layoff flags (0.5 salary on layoff year).
// One thread per (batch simulation, career year).
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn generate_layoffs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let years = career_years();
    let batch_sims = params.generate.z;
    if (index >= batch_sims * years) {
        return;
    }
    let global_sim = params.generate.w + index / years;
    let year = index % years;
    let counter = global_sim * years + year;
    var u = 0.0;
    if (params.dispatch.y != 0u) {
        u = rqmc_draw(global_sim, params.dimensions.z * 10u + year, params.generate.x, params.dispatch.y);
    } else {
        u = threefry_uniform(counter, params.generate.x ^ 0x1F2E3D4Cu, 0x90D3A51Fu);
    }
    scratch[returns_region_size() + index] = select(1.0, 0.5, u < params.generate1.w);
}