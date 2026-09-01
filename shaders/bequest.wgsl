// ===========================================================================
// bequest.wgsl - Terminal-estate quantile ladders (bequest-adjusted CE).
// ===========================================================================
//
// SEPARATE compute module (own Params struct + bindings), run once per
// simulation AFTER the main pipeline and the quantile reduction. The solver
// is untouched: w* stays the maximum sustainable spending, and for every
// strategy and every estate-grid fraction f of w* each simulated life is
// re-walked through retirement at w = f * w* (the walk is a faithful copy of
// solver.wgsl's test_solvency minus the early insolvency bail) and the
// tax-adjusted estate at age 95 is recorded:
//
//     estate = TFSA balance                                    (tax-free)
//            + non-registered net of deemed disposition       (gains taxed at
//              capital-gains-inclusion x CG tax rate)
//            + RRSP/RRIF balance net of the top marginal bracket (fully taxed)
//            + home equity when bought                        (principal-
//              residence exemption; the mortgage amortizes to zero by
//              retirement; the value is fixed at the target property value)
//
// The engine rolls any FHSA balance into the RRSP by age 40 / retirement
// (accumulation.wgsl), so no separate FHSA account exists at death and its
// estate follows the RRSP rule above. The (theta, k) bequest preferences
// themselves never reach the GPU: the ladders are preference-independent and
// the utility maximization over f happens in JavaScript at re-rank time.
//
// THREE entry points, mirroring the solver's batching so no single dispatch
// outlives the Windows TDR watchdog (~2 s):
//   bequest_reset - zeroes the persistent per-(allocation, grid) histogram.
//   bequest_walk  - ONE dispatch PER SIMULATION BATCH: every workgroup
//                   (allocation x grid fraction) walks only this batch's
//                   lives at w = f * w* and atomically accumulates a fixed
//                   log2(estate + 1)-spaced histogram (plus the exact raw
//                   min/max via the u32 bit pattern of non-negative floats)
//                   into the persistent buffer. The fixed scale is what makes
//                   cross-batch accumulation exact: no discovery pass needed.
//   bequest_final - one tiny dispatch after the last batch: turns the
//                   accumulated histogram into the 201-point ladder
//                   (P0 / P100 exact min / max, interior points linearly
//                   interpolated in value space between log-bin edges).
//
// The grid fractions and the target property value live in the model buffer
// tail (see calibration.build_model_buffer / JS buildDynamicModel).
// ===========================================================================

struct Params {
    dimensions: vec4<u32>,   // total simulations, allocations, total months, accumulation months
    solver: vec4<u32>,       // retirement months, underlying-fund count, bisection steps, first age-75 month
    calendar: vec4<u32>,     // first post-pension month, current age, career start age, retirement age
    constants0: vec4<f32>,   // distribution yield, distribution tax, HISA return, CG inclusion
    constants1: vec4<f32>,   // CG tax rate, cash-wedge FRACTION of the retirement
                             // span (wedge = w * fraction * retirement months),
                             // meltdown target, OAS threshold
    constants2: vec4<f32>,   // OAS clawback rate, employer match rate, match percent, accumulation path count
    generate: vec4<u32>,     // PRNG seed, skew-t df, batch simulation count, batch offset
    generate1: vec4<f32>,    // real borrowing rate/12, extra MER 1.5/12, extra MER 2.0/12, layoff probability
    dispatch: vec4<u32>,     // unused here; mirrors the main Params layout (160 bytes)
    glide: vec4<f32>,        // glidepath shares; mirrors the main Params layout
};

@group(0) @binding(0) var<storage, read> params: Params;
// Max sustainable annual spending per (allocation, total simulation).
@group(0) @binding(1) var<storage, read> spending_results: array<f32>;
// Packed global per-simulation data (one buffer so the module stays within
// the max-8-storage-buffers-per-stage limit; mirrors the main module's
// scratch-packing philosophy):
//   [0, totalSims*totalMonths*funds)   monthly returns
//   [+, +houseCount*pathCount*totalSims*4) retirement states as vec4
//                                        (tfsa, rrsp, non_reg, non_reg_acb)
//   [+, +houseCount*totalSims*2)        house outcomes as vec2 (bought, month)
@group(0) @binding(2) var<storage, read> sim_data: array<f32>;
// Allocation metadata (12 u32 per strategy; see common.wgsl binding 2).
@group(0) @binding(3) var<storage, read> allocations: array<vec4<u32>>;
// Packed model buffer (career rows, month rows, tax tail, skew-t constants,
// 11 house constants incl. the target property value, estate-grid tail).
@group(0) @binding(4) var<storage, read> model_values: array<f32>;
// 201-point estate quantile ladder per (allocation, grid fraction),
// written only by bequest_final.
@group(0) @binding(5) var<storage, read_write> estate_output: array<f32>;
// Persistent accumulation buffer per (allocation, grid fraction):
// [0, 512) fixed log2(estate + 1) histogram bins (u32), then the exact raw
// min and max as the u32 bit pattern of the non-negative f32 estate.
// Written by bequest_walk, read by bequest_final. Two read-write bindings
// total in this module (5 and 6) - the AMD/D3D12 two-buffer limit holds.
@group(0) @binding(6) var<storage, read_write> estate_hist: array<atomic<u32>>;

const QUANTILE_WORKGROUP = 256u;
const HIST_BINS = 512u;
const HIST_WORDS = 514u;  // 512 bins + min + max
// Fixed log2(estate + 1) histogram ceiling: terminal estates of fat-tailed
// 10k-path runs stay well below 2^30 (~$1.07B), so the scale never needs a
// discovery pass and per-batch accumulation is exact. The log spacing keeps
// relative resolution fine exactly where the gamma > 1 bequest utility
// concentrates (the small estates).
const LOG_SCALE = 30.0;
const MIN_SENTINEL = 0xFFFFFFFFu;

var<workgroup> hist: array<atomic<u32>, HIST_BINS>;
var<workgroup> bin_scale_v: f32;
var<workgroup> q_bins: array<u32, 201>;
var<workgroup> q_cums: array<u32, 201>;
var<workgroup> q_counts: array<u32, 201>;

// --- model-buffer accessors (mirror common.wgsl, unrolled for these bindings)
fn career_years() -> u32 {
    return params.dimensions.w / 12u - (params.calendar.z - params.calendar.y);
}

fn return_at(simulation: u32, month: u32, fund: u32) -> f32 {
    return sim_data[(simulation * params.dimensions.z + month) * params.solver.y + fund];
}

fn states_region_offset() -> u32 {
    return params.dimensions.x * params.dimensions.z * params.solver.y;
}

fn states_at(house: u32, path: u32, simulation: u32) -> vec4<f32> {
    let index = states_region_offset()
        + ((house * u32(params.constants2.w) + path) * params.dimensions.x + simulation) * 4u;
    return vec4<f32>(sim_data[index], sim_data[index + 1u], sim_data[index + 2u], sim_data[index + 3u]);
}

fn house_outcomes_at(house: u32, simulation: u32) -> vec2<f32> {
    let index = states_region_offset()
        + house_count() * u32(params.constants2.w) * params.dimensions.x * 4u
        + (house * params.dimensions.x + simulation) * 2u;
    return vec2<f32>(sim_data[index], sim_data[index + 1u]);
}

fn house_const_at(index: u32) -> f32 {
    let offset = career_years() * 6u + params.solver.x * 8u + 54u + 38u;
    return model_values[offset + index];
}

fn house_count() -> u32 {
    return u32(house_const_at(9u));
}

fn month0_at(month: u32) -> vec4<f32> {
    let offset = career_years() * 6u + month * 4u;
    return vec4<f32>(model_values[offset], model_values[offset + 1u], model_values[offset + 2u], model_values[offset + 3u]);
}

fn month1_at(month: u32) -> vec4<f32> {
    let offset = career_years() * 6u + params.solver.x * 4u + month * 4u;
    return vec4<f32>(model_values[offset], model_values[offset + 1u], model_values[offset + 2u], model_values[offset + 3u]);
}

fn tax_at(index: u32) -> f32 {
    let offset = career_years() * 6u + params.solver.x * 8u;
    return model_values[offset + index];
}

fn monthly_tax(gross: f32) -> f32 {
    // tax tail [44..47] holds the four nonzero monthly bracket thresholds and
    // [49..53] holds their rates; slot 48 is an unused pad. tax_at(53) is the
    // top marginal bracket used for the deemed disposition of the RRSP/RRIF.
    let b1 = tax_at(44u);
    let b2 = tax_at(45u);
    let b3 = tax_at(46u);
    let b4 = tax_at(47u);
    var tax = 0.0;
    tax += max(0.0, min(gross, b1)) * tax_at(49u);
    tax += max(0.0, min(gross, b2) - b1) * tax_at(50u);
    tax += max(0.0, min(gross, b3) - b2) * tax_at(51u);
    tax += max(0.0, min(gross, b4) - b3) * tax_at(52u);
    tax += max(0.0, gross - b4) * tax_at(53u);
    return tax;
}

fn phase_fund(code: u32, month: u32, first_end: u32, second_end: u32) -> u32 {
    // first_end/second_end come from the allocation metadata's glide boundary
    // words (built from params.glide's configurable shares, mirrored in
    // calibration._glidepath_boundaries and the JS glidepathBoundaries).
    if (code < 5u) {
        return code;
    }
    if (code == 5u) {  // DECLINING: VEQT -> VGRO -> VBAL
        if (month < first_end) {
            return 0u;
        }
        if (month < second_end) {
            return 3u;
        }
        return 4u;
    }
    // RISING: VBAL -> VGRO -> VEQT
    if (month < first_end) {
        return 4u;
    }
    if (month < second_end) {
        return 3u;
    }
    return 0u;
}

fn net_monthly(gross: f32, oas_max: f32, threshold: f32) -> f32 {
    var clawback = 0.0;
    if (oas_max > 0.0) {
        clawback = min(oas_max, max(0.0, gross - threshold) * params.constants2.x);
    }
    return gross - monthly_tax(gross) - clawback;
}

fn interp_tax(target_net: f32, gross_offset: u32, net_offset: u32, count: u32) -> f32 {
    let first_net = tax_at(net_offset);
    let last_index = count - 1u;
    let last_net = tax_at(net_offset + last_index);
    if (target_net <= first_net) {
        return tax_at(gross_offset);
    }
    if (target_net >= last_net) {
        return tax_at(gross_offset + last_index);
    }
    var i = 1u;
    loop {
        if (i >= count) {
            break;
        }
        let n_hi = tax_at(net_offset + i);
        if (target_net <= n_hi) {
            let n_lo = tax_at(net_offset + i - 1u);
            let g_lo = tax_at(gross_offset + i - 1u);
            let g_hi = tax_at(gross_offset + i);
            let fraction = (target_net - n_lo) / (n_hi - n_lo);
            return g_lo + fraction * (g_hi - g_lo);
        }
        i += 1u;
    }
    return tax_at(gross_offset + last_index);
}

fn estate_grid_offset() -> u32 {
    return career_years() * 6u + params.solver.x * 8u + 54u + 38u + 11u;
}

fn estate_grid_count() -> u32 {
    return u32(model_values[estate_grid_offset()]);
}

fn estate_fraction(index: u32) -> f32 {
    return model_values[estate_grid_offset() + 1u + index];
}

// ---------------------------------------------------------------------------
// Retirement walk at annual net spending w (mirror of solver.wgsl
// test_solvency, without the early insolvency bail) -> tax-adjusted estate at
// age 95. At w = f * w* with f <= 1 the walk is solvent by construction; the
// withdrawal clamps keep every balance non-negative either way.
// ---------------------------------------------------------------------------
fn terminal_estate(w: f32, simulation: u32, allocation: u32) -> f32 {
    let allocation_info = allocations[allocation * 3u];
    let accumulation_schedule = allocations[allocation * 3u + 1u];
    let retirement_schedule = allocations[allocation * 3u + 2u];
    let accumulation_fund = allocation_info.x;
    let bridge_fund = allocation_info.y;
    let post_fund = allocation_info.z;
    let bridge_cash = (allocation_info.w & 1u) != 0u;
    let post_cash = (allocation_info.w & 2u) != 0u;
    let house = (allocation_info.w >> 2u) & 7u;
    let initial = states_at(house, accumulation_fund, simulation);

    let house_bought = house_outcomes_at(house, simulation).x;
    let housing_cost = select(house_const_at(4u), house_const_at(3u), house_bought > 0.5);

    var tfsa = initial.x;
    var rrsp = initial.y;
    var non_reg = initial.z;
    var non_reg_acb = initial.w;
    var cash_wedge = 0.0;

    // Bridge cash wedge: carve out cashWedgeFraction of the RETIREMENT SPAN of
    // annual spending up front, prorated across the accounts by their current
    // weights (wedge = w * fraction * retirement YEARS = months / 12).
    if (bridge_cash) {
        let total = tfsa + rrsp + non_reg;
        let actual_cash = min(total, w * params.constants1.y * f32(params.solver.x) / 12.0);
        let safe_total = max(total, 1e-10);
        let fraction = actual_cash / safe_total;
        tfsa -= tfsa * fraction;
        rrsp -= rrsp * fraction;
        if (non_reg > 1e-5) {
            non_reg_acb -= non_reg_acb * fraction;
        }
        non_reg -= non_reg * fraction;
        cash_wedge += actual_cash;
    }

    var post_wedge_established = false;

    for (var month = 0u; month < params.solver.x; month += 1u) {
        var phase_code = post_fund;
        var phase_month = month - params.calendar.x;
        var first_end = retirement_schedule.x;
        var second_end = retirement_schedule.y;
        if (month < params.calendar.x) {
            phase_code = bridge_fund;
            phase_month = month;
            first_end = accumulation_schedule.z;
            second_end = accumulation_schedule.w;
        }
        let fund = phase_fund(phase_code, phase_month, first_end, second_end);
        let r = return_at(simulation, params.dimensions.w + month, fund);
        tfsa *= 1.0 + r;
        rrsp *= 1.0 + r;

        let distribution = non_reg * params.constants0.x;
        let distribution_net = distribution * (1.0 - params.constants0.y);
        non_reg = non_reg * (1.0 + r - params.constants0.x) + distribution_net;
        non_reg_acb += distribution_net;
        cash_wedge *= 1.0 + params.constants0.z;

        // Post-pension cash wedge topped up at the end of the bridge phase.
        if (post_cash && !post_wedge_established && month == params.calendar.x) {
            let total = tfsa + rrsp + non_reg;
            let needed = max(0.0, w * params.constants1.y * f32(params.solver.x) / 12.0 - cash_wedge);
            let actual_cash = min(total, needed);
            let safe_total = max(total, 1e-10);
            let fraction = actual_cash / safe_total;
            tfsa -= tfsa * fraction;
            rrsp -= rrsp * fraction;
            if (non_reg > 1e-5) {
                non_reg_acb -= non_reg_acb * fraction;
            }
            non_reg -= non_reg * fraction;
            cash_wedge += actual_cash;
            post_wedge_established = true;
        }

        let month_values = month0_at(month);
        let later_values = month1_at(month);
        let required = w * month_values.x + month_values.y + housing_cost;
        let pension = month_values.z;
        let pension_net = month_values.w;
        var remaining = max(0.0, required - pension_net);

        // Cash wedge is spent first.
        let cash_withdrawal = min(cash_wedge, remaining);
        cash_wedge -= cash_withdrawal;
        remaining -= cash_withdrawal;

        if (month < params.calendar.x) {
            // Bridge phase: RRSP withdrawal grossed up to the marginal tax
            // rate, spillover into non-registered, then TFSA.
            let gross_needed = interp_tax(remaining, 0u, 6u, 6u);
            let gross_target = max(gross_needed, params.constants1.z);
            let rrsp_gross = min(rrsp, gross_target);
            rrsp -= rrsp_gross;

            let rrsp_net = net_monthly(rrsp_gross, later_values.y, params.constants1.w);
            let rrsp_spend = min(rrsp_net, remaining);
            remaining -= rrsp_spend;

            let excess_net = max(0.0, rrsp_net - rrsp_spend);
            non_reg += excess_net;
            non_reg_acb += excess_net;

            if (non_reg > 1e-5 && remaining > 1e-5) {
                let gain_fraction = max(0.0, (non_reg - non_reg_acb) / max(non_reg, 1e-10));
                let effective_cg_tax = gain_fraction * params.constants0.w * params.constants1.x;
                let non_reg_gross_needed = remaining / max(1e-5, 1.0 - effective_cg_tax);
                let non_reg_gross = min(non_reg, non_reg_gross_needed);
                non_reg_acb -= non_reg_acb * (non_reg_gross / non_reg);
                non_reg -= non_reg_gross;
                remaining -= non_reg_gross * (1.0 - effective_cg_tax);
            }

            let tfsa_withdrawal = min(tfsa, remaining);
            tfsa -= tfsa_withdrawal;
            remaining -= tfsa_withdrawal;
        } else {
            // Post-pension: RRIF minimum + top-up to the target net total.
            let minimum_rrsp_gross = (rrsp * later_values.x) / 12.0;
            let target_net_total = pension_net + remaining;
            var gross_needed_total: f32;
            if (month < params.solver.w) {
                gross_needed_total = interp_tax(target_net_total, 12u, 20u, 8u);
            } else {
                gross_needed_total = interp_tax(target_net_total, 28u, 36u, 8u);
            }
            let rrsp_gross_needed = max(0.0, gross_needed_total - pension);
            let rrsp_gross = min(rrsp, max(minimum_rrsp_gross, rrsp_gross_needed));
            rrsp -= rrsp_gross;

            let actual_gross = pension + rrsp_gross;
            let actual_net = net_monthly(actual_gross, later_values.y, params.constants1.w);
            let rrsp_net = max(0.0, actual_net - pension_net);
            let rrsp_spend = min(rrsp_net, remaining);
            remaining -= rrsp_spend;

            let excess_net = max(0.0, rrsp_net - rrsp_spend);
            non_reg += excess_net;
            non_reg_acb += excess_net;

            if (non_reg > 1e-5 && remaining > 1e-5) {
                let gain_fraction = max(0.0, (non_reg - non_reg_acb) / max(non_reg, 1e-10));
                let effective_cg_tax = gain_fraction * params.constants0.w * params.constants1.x;
                let non_reg_gross_needed = remaining / max(1e-5, 1.0 - effective_cg_tax);
                let non_reg_gross = min(non_reg, non_reg_gross_needed);
                non_reg_acb -= non_reg_acb * (non_reg_gross / non_reg);
                non_reg -= non_reg_gross;
                remaining -= non_reg_gross * (1.0 - effective_cg_tax);
            }

            let tfsa_withdrawal = min(tfsa, remaining);
            tfsa -= tfsa_withdrawal;
            remaining -= tfsa_withdrawal;
        }
        // No early insolvency bail here: the estate walk always runs the full
        // horizon so the terminal balances are well defined for every f.
    }

    // Deemed disposition at death (age 95):
    //   TFSA: liquidated 100% tax-free.
    //   Non-registered: gains taxed at the capital-gains inclusion x CG rate.
    //   RRSP/RRIF: fully taxed at the top marginal bracket (tax_at(53)).
    //   Home: principal-residence exemption; mortgage is zero by retirement;
    //   the value is fixed at the target property value (house_const_at(10)).
    let gain_fraction = max(0.0, (non_reg - non_reg_acb) / max(non_reg, 1e-10));
    let non_reg_estate = non_reg * (1.0 - gain_fraction * params.constants0.w * params.constants1.x);
    let rrsp_estate = rrsp * (1.0 - tax_at(53u));
    var estate = tfsa + non_reg_estate + rrsp_estate;
    if (house_bought > 0.5) {
        estate += house_const_at(10u);
    }
    return max(estate, 0.0);
}

// ---------------------------------------------------------------------------
// Pass 1: zero the persistent histograms (+ min/max sentinels) for the run.
// One workgroup per (allocation, grid fraction); trivial work.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn bequest_reset(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let grid = estate_grid_count();
    if (grid == 0u) {
        return;
    }
    let allocation = wgid.x / grid;
    if (allocation >= params.dimensions.y) {
        return;
    }
    let base = wgid.x * HIST_WORDS;
    for (var i = lid; i < HIST_BINS; i += QUANTILE_WORKGROUP) {
        atomicStore(&estate_hist[base + i], 0u);
    }
    if (lid == 0u) {
        atomicStore(&estate_hist[base + HIST_BINS], MIN_SENTINEL);
        atomicStore(&estate_hist[base + HIST_BINS + 1u], 0u);
    }
}

// ---------------------------------------------------------------------------
// Pass 2 (per simulation batch): walk this batch's lives at w = f * w* and
// fold the workgroup-local histogram into the persistent one. The exact raw
// min/max accumulate through atomicMin/atomicMax on the u32 bit pattern of
// the non-negative f32 estate (IEEE ordering is monotonic for floats >= 0).
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn bequest_walk(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let grid = estate_grid_count();
    if (grid == 0u) {
        return;
    }
    let allocation = wgid.x / grid;
    let fraction_index = wgid.x % grid;
    let n = params.dimensions.x;
    if (allocation >= params.dimensions.y) {
        return;
    }
    let count = params.generate.z;   // this batch's simulation count
    let offset = params.generate.w;  // this batch's global simulation offset
    let fraction = estate_fraction(fraction_index);
    let hist_base = wgid.x * HIST_WORDS;

    bin_scale_v = f32(HIST_BINS) / LOG_SCALE;
    for (var i = lid; i < HIST_BINS; i += QUANTILE_WORKGROUP) {
        atomicStore(&hist[i], 0u);
    }
    workgroupBarrier();

    var j = lid;
    while (j < count) {
        let simulation = offset + j;
        let estate = terminal_estate(fraction * spending_results[allocation * n + simulation], simulation, allocation);
        let bin = min(u32(log2(estate + 1.0) * bin_scale_v), HIST_BINS - 1u);
        atomicAdd(&hist[bin], 1u);
        atomicMin(&estate_hist[hist_base + HIST_BINS], bitcast<u32>(estate));
        atomicMax(&estate_hist[hist_base + HIST_BINS + 1u], bitcast<u32>(estate));
        j += QUANTILE_WORKGROUP;
    }
    workgroupBarrier();

    // Fold the local histogram into the persistent one (one global atomic per
    // non-empty bin, so cross-batch accumulation is an exact integer sum).
    for (var b = lid; b < HIST_BINS; b += QUANTILE_WORKGROUP) {
        let c = atomicLoad(&hist[b]);
        if (c > 0u) {
            atomicAdd(&estate_hist[hist_base + b], c);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 3 (once after the last batch): reduce the accumulated histogram to the
// 201-point ladder. P0 / P100 are the exact raw min / max; interior points
// interpolate linearly in value space between the log2(estate + 1) bin edges
// (inverting the +1 shift), exactly like quantiles.wgsl's interpolation.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn bequest_final(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let grid = estate_grid_count();
    if (grid == 0u) {
        return;
    }
    let allocation = wgid.x / grid;
    let fraction_index = wgid.x % grid;
    let n = params.dimensions.x;
    if (allocation >= params.dimensions.y) {
        return;
    }
    let hist_base = wgid.x * HIST_WORDS;
    let out_base = (allocation * grid + fraction_index) * 201u;
    let step_v = LOG_SCALE / f32(HIST_BINS);

    let vmin = bitcast<f32>(atomicLoad(&estate_hist[hist_base + HIST_BINS]));
    let vmax = bitcast<f32>(atomicLoad(&estate_hist[hist_base + HIST_BINS + 1u]));

    if (lid == 0u) {
        estate_output[out_base] = vmin;
        estate_output[out_base + 200u] = vmax;
        if (vmax <= 0.0) {
            // Degenerate all-zero estate: every quantile is exactly 0.
            for (var z = 1u; z < 200u; z += 1u) {
                estate_output[out_base + z] = 0.0;
            }
        } else {
            var bin = 0u;
            var cum_before = 0u;
            for (var i = 1u; i <= 199u; i += 1u) {
                let position = f32(n - 1u) * f32(i) / 200.0;
                while (bin < HIST_BINS - 1u && f32(cum_before + atomicLoad(&estate_hist[hist_base + bin])) < position) {
                    cum_before += atomicLoad(&estate_hist[hist_base + bin]);
                    bin += 1u;
                }
                q_bins[i] = bin;
                q_cums[i] = cum_before;
                q_counts[i] = atomicLoad(&estate_hist[hist_base + bin]);
            }
        }
    }
    workgroupBarrier();

    if (lid == 0u) {
        for (var i = 1u; i <= 199u; i += 1u) {
            let bin = q_bins[i];
            let rank = f32(n - 1u) * f32(i) / 200.0;
            var p_fine = clamp(rank - f32(q_cums[i]), 0.0, f32(q_counts[i]));
            let sub_frac = p_fine / max(f32(q_counts[i]), 1.0);
            let edge = f32(bin) * step_v;
            let lo = exp2(edge) - 1.0;
            let hi = exp2(edge + step_v) - 1.0;
            estate_output[out_base + i] = lo + sub_frac * (hi - lo);
        }
    }
}
