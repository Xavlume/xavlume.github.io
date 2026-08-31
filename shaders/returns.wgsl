// ===========================================================================
// returns.wgsl - Pass 1: on-chip multivariate skew-t return generation.
// ===========================================================================
//
// One thread per (batch simulation, month). The Threefry counter uses the
// GLOBAL simulation index (params.generate.w + local batch index) so that
// consecutive batches draw distinct, reproducible streams.
//
// Per sampled month we draw:
//   - |N(0,1)| for the skew component                    (t = 0)
//   - a chi-square_nu component from nu squared normals (t = 1..nu)
//   - three independent normals driving the underlying funds (t = nu+1..nu+3)
// then combine them with the calibrated xi/omega/delta/Cholesky and the
// leverage transforms for VEQT1.5 and VEQT2.

@compute @workgroup_size(64, 1, 1)
fn generate_returns(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let batch_sims = params.generate.z;
    let total_months = params.dimensions.z;
    if (index >= batch_sims * total_months) {
        return;
    }
    let global_sim = params.generate.w + index / total_months;
    let month = index % total_months;
    let counter = global_sim * total_months + month;

    let nu = params.generate.y;
    let seed = params.generate.x;
    // Linear normal stream t = 0..: t=0 skew, t=1..nu chi-square components,
    // t=nu+1..nu+3 the three independent normals driving the underlying funds.
    // Modes 3/4 (params.dispatch.y >= 3) instead draw the chi2 scale directly
    // from the second uniform of pair 0 via the LUT, keeping the fund normals
    // at pairs 3/4 - same marginals, volatility regime stratified per uniform.
    let direct_w = params.dispatch.y >= 3u;
    let pairs_needed = (nu + 5u) / 2u;
    var scale = 0.0;
    var skew = 0.0;
    var fund_normal = vec3<f32>(0.0);
    for (var pair = 0u; pair < pairs_needed; pair += 1u) {
        var u0 = 0.0;
        var u1 = 0.0;
        if (params.dispatch.y != 0u) {
            // RQMC: coordinate month*10 + pair*2 (+slot) of the (seed, global
            // sim) scrambled Sobol point; dispatch.y selects the scramble
            // (odd = digital shift, even > 0 = Owen nested).
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
    }

    // scale = chi_square_nu / nu, matching the reference gamma(nu/2, 2/nu).
    let inv_scale = 1.0 / sqrt(max(scale / f32(nu), 1e-12));
    let xi0 = return_model_at(0u);
    let xi1 = return_model_at(1u);
    let xi2 = return_model_at(2u);
    let omega0 = return_model_at(3u);
    let omega1 = return_model_at(4u);
    let omega2 = return_model_at(5u);
    let delta0 = return_model_at(6u);
    let delta1 = return_model_at(7u);
    let delta2 = return_model_at(8u);
    // Row-major 3x3 Cholesky of the residual correlation.
    let l00 = return_model_at(9u);
    let l01 = return_model_at(10u);
    let l02 = return_model_at(11u);
    let l10 = return_model_at(12u);
    let l11 = return_model_at(13u);
    let l12 = return_model_at(14u);
    let l20 = return_model_at(15u);
    let l21 = return_model_at(16u);
    let l22 = return_model_at(17u);

    let veqt = max(-0.95, xi0 + omega0 * (delta0 * skew + fund_normal.x * l00 + fund_normal.y * l01 + fund_normal.z * l02) * inv_scale);
    let vgro = max(-0.95, xi1 + omega1 * (delta1 * skew + fund_normal.x * l10 + fund_normal.y * l11 + fund_normal.z * l12) * inv_scale);
    let vbal = max(-0.95, xi2 + omega2 * (delta2 * skew + fund_normal.x * l20 + fund_normal.y * l21 + fund_normal.z * l22) * inv_scale);

    let out_index = index * params.solver.y;
    scratch[out_index] = veqt;
    // Leverage-off runs never read funds 1/2 (no strategy references them),
    // so the transforms are skipped entirely when params.dispatch.z is 0.
    if (params.dispatch.z != 0u) {
        let veqt15 = max(-0.95, 1.5 * veqt - 0.5 * params.generate1.x - params.generate1.y);
        let veqt2 = max(-0.95, 2.0 * veqt - params.generate1.x - params.generate1.z);
        scratch[out_index + 1u] = veqt15;
        scratch[out_index + 2u] = veqt2;
    }
    scratch[out_index + 3u] = vgro;
    scratch[out_index + 4u] = vbal;
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