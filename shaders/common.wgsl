// ===========================================================================
// common.wgsl - Shared buffer schemas, memory layout and helpers
// ===========================================================================
//
// THIS FILE IS A FRAGMENT, not a standalone module. build_html.py and
// engine.py concatenate the pass files in this fixed order into one WGSL
// compute module:
//
//     common.wgsl  ->  returns.wgsl  ->  accumulation.wgsl  ->
//     solver.wgsl  ->  drawdown_ui.wgsl
//
// quantiles.wgsl is a separate, self-contained module (its own Params and
// bindings) because it runs as an independent reduced pipeline.
//
// The Params struct is a 144-byte uniform packed exactly like the JS
// makeParams() and Python _make_params(). The storage-buffer memory layout is
// documented here so no magic integer offset lives inside a compute entry.

struct Params {
    dimensions: vec4<u32>,   // total simulations, allocations, total months, accumulation months
    solver: vec4<u32>,       // retirement months, underlying-fund count, bisection steps, first age-75 month
    calendar: vec4<u32>,     // first post-pension month, current age, career start age, retirement age
    constants0: vec4<f32>,   // distribution yield, distribution tax, HISA return, CG inclusion
    constants1: vec4<f32>,   // CG tax rate, cash-wedge years, meltdown target, OAS threshold
    constants2: vec4<f32>,   // OAS clawback rate, employer match rate, match percent, accumulation path count
    generate: vec4<u32>,     // PRNG seed, skew-t df, batch simulation count, batch offset
    generate1: vec4<f32>,    // real borrowing rate/12, extra MER 1.5/12, extra MER 2.0/12, layoff probability
    dispatch: vec4<u32>,     // allocation columns per workgroup (compat slicing); 1 = one column per thread
};

@group(0) @binding(0) var<storage, read> params: Params;

//
// Binding 1 - packed on-chip scratch, written and read by the compute passes.
// Layout (float32 words, sized by the ACTUAL batch count params.generate.z):
//   [0, RET)                      monthly returns
//                                 index = (sim * totalMonths + month)*fundCount + fund,
//                                 fund 0..4 = VEQT, VEQT1.5, VEQT2, VGRO, VBAL
//   [RET, RET + LAY)              annual career layoff multipliers (0.5 or 1.0)
//                                 index = sim * careerYears + year
//   [RET+LAY, +5*7*4*batch)       35 retirement states per simulation
//                                 (5 house strategies x 7 accumulation paths) as vec4
//                                 (tfsa, rrsp, non_reg, non_reg_acb)
//   [+ 5*2*batch)                 per (house, simulation) house outcomes:
//                                 (bought flag, buy month)
//   [+ DRAW)                      one Composite Ulcer Index score per
//                                 (simulation, allocation)
//   [+ UI_ACC)                    accumulation-phase Ulcer Index per
//                                 (accumulation path, simulation), memoized
//                                 once by the accumulate pass (the index
//                                 depends only on the path's glidepath) so
//                                 track_drawdowns never re-walks the
//                                 accumulation months per allocation
//
@group(0) @binding(1) var<storage, read_write> scratch: array<f32>;

// Binding 2 - per-strategy allocation metadata as 3 vec4<u32> rows per
// strategy (12 u32 total): [accumCode, bridgeCode, postCode, flags,
// accumGlide.xy, bridgeGlide.xy, postGlide.xy, 0, 0]. Flags: bit0 cash bridge,
// bit1 cash post, bits2-4 house code.
@group(0) @binding(2) var<storage, read> allocations: array<vec4<u32>>;

// Binding 3 - the packed model buffer (career rows, month0 rows, month1 rows,
// tax values, the 18 calibrated skew-t constants, then the 10 house
// constants). See calibration.build_model_buffer.
@group(0) @binding(3) var<storage, read> model_values: array<f32>;

// Bindings 4-5 are unused read-only placeholders kept to satisfy the fixed
// 7-entry layout the Python and JS runtimes share.
@group(0) @binding(4) var<storage, read> unused_a: array<f32>;
@group(0) @binding(5) var<storage, read> unused_b: array<f32>;

// Binding 6 - max sustainable annual spending per (allocation, simulation),
// index = allocation * totalSimulations + simulation.
@group(0) @binding(6) var<storage, read_write> spending_results: array<f32>;

// ---------------------------------------------------------------------------
// Memory-layout helpers
// ---------------------------------------------------------------------------
fn career_years() -> u32 {
    return params.dimensions.w / 12u - (params.calendar.z - params.calendar.y);
}

fn returns_region_size() -> u32 {
    return params.generate.z * params.dimensions.z * params.solver.y;
}

fn states_region_offset() -> u32 {
    return returns_region_size() + params.generate.z * career_years();
}

fn states_region_size() -> u32 {
    return house_count() * u32(params.constants2.w) * params.generate.z * 4u;
}

fn house_flags_region_offset() -> u32 {
    return states_region_offset() + states_region_size();
}

fn drawdown_region_offset() -> u32 {
    return house_flags_region_offset() + house_count() * params.generate.z * 2u;
}

fn accum_ui_region_offset() -> u32 {
    return drawdown_region_offset() + params.generate.z * params.dimensions.y;
}

fn accum_ui_at(path: u32, simulation: u32) -> f32 {
    return scratch[accum_ui_region_offset() + path * params.generate.z + simulation];
}

fn return_at(simulation: u32, month: u32, fund: u32) -> f32 {
    let index = (simulation * params.dimensions.z + month) * params.solver.y + fund;
    return scratch[index];
}

fn layoffs_at(simulation: u32, year: u32) -> f32 {
    return scratch[returns_region_size() + simulation * career_years() + year];
}

// Career table stride is six floats per year:
//   (net retirement stream, net house stream, tax rate, cumulative TFSA room,
//    cumulative RRSP room, salary)
fn career0_at(year: u32) -> vec4<f32> {
    let offset = year * 6u;
    return vec4<f32>(model_values[offset], model_values[offset + 1u], model_values[offset + 2u], model_values[offset + 3u]);
}

fn career_scalar(year: u32, index: u32) -> f32 {
    return model_values[year * 6u + index];
}

// Ten house constants appended after the 54-value tax tail and the 18
// calibrated skew-t constants:
//   0 target house capital (down payment + closing costs)
//   1 mortgage principal
//   2 monthly real mortgage rate
//   3 monthly property taxes & condo maintenance
//   4 monthly market rent (renters)
//   5 FHSA annual limit
//   6 FHSA lifetime maximum
//   7 HBP maximum withdrawal
//   8 HBP repayment years
//   9 house strategy count
fn house_const_at(index: u32) -> f32 {
    let offset = career_years() * 6u + params.solver.x * 8u + 54u + 18u;
    return model_values[offset + index];
}

fn house_count() -> u32 {
    return u32(house_const_at(9u));
}

fn states_at(house: u32, path: u32, simulation: u32) -> vec4<f32> {
    let index = states_region_offset()
        + ((house * u32(params.constants2.w) + path) * params.generate.z + simulation) * 4u;
    return vec4<f32>(scratch[index], scratch[index + 1u], scratch[index + 2u], scratch[index + 3u]);
}

fn states_set(house: u32, path: u32, simulation: u32, value: vec4<f32>) {
    let index = states_region_offset()
        + ((house * u32(params.constants2.w) + path) * params.generate.z + simulation) * 4u;
    scratch[index] = value.x;
    scratch[index + 1u] = value.y;
    scratch[index + 2u] = value.z;
    scratch[index + 3u] = value.w;
}

fn house_outcomes_at(house: u32, simulation: u32) -> vec2<f32> {
    let index = house_flags_region_offset() + (house * params.generate.z + simulation) * 2u;
    return vec2<f32>(scratch[index], scratch[index + 1u]);
}

fn house_outcomes_set(house: u32, simulation: u32, value: vec2<f32>) {
    let index = house_flags_region_offset() + (house * params.generate.z + simulation) * 2u;
    scratch[index] = value.x;
    scratch[index + 1u] = value.y;
}

fn month0_at(month: u32) -> vec4<f32> {
    // (smile/12, healthcare/12, gross pension, net pension)
    let offset = career_years() * 6u + month * 4u;
    return vec4<f32>(model_values[offset], model_values[offset + 1u], model_values[offset + 2u], model_values[offset + 3u]);
}

fn month1_at(month: u32) -> vec4<f32> {
    // (RRIF factor, OAS maximum, 0, 0)
    let offset = career_years() * 6u + params.solver.x * 4u + month * 4u;
    return vec4<f32>(model_values[offset], model_values[offset + 1u], model_values[offset + 2u], model_values[offset + 3u]);
}

fn tax_at(index: u32) -> f32 {
    let offset = career_years() * 6u + params.solver.x * 8u;
    return model_values[offset + index];
}

// 18 calibrated skew-t constants appended after the 54-value tax tail:
// xi[3], omega[3], delta[3], row-major 3x3 Cholesky [9].
fn return_model_at(index: u32) -> f32 {
    let offset = career_years() * 6u + params.solver.x * 8u + 54u;
    return model_values[offset + index];
}

fn monthly_tax(gross: f32) -> f32 {
    // tax tail [44..47] holds the four nonzero monthly bracket thresholds and
    // [49..53] holds their rates; slot 48 is an unused pad.
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

// ---------------------------------------------------------------------------
// Counter-based PRNG: Threefry-2x32-20 (Random123), no 64-bit arithmetic.
// ---------------------------------------------------------------------------
const ROTATION = array<u32, 4>(13u, 15u, 26u, 6u);

fn rotl32(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn threefry_u32(ctr0: u32, ctr1: u32, key0: u32, key1: u32) -> vec2<u32> {
    let ks0 = key0;
    let ks1 = key1;
    let ks2 = key0 ^ key1 ^ 0x1BD11BDAu;
    var x0 = ctr0 + ks0;
    var x1 = ctr1 + ks1;
    for (var round = 0u; round < 20u; round += 1u) {
        x0 = x0 + x1;
        x1 = rotl32(x1, ROTATION[round % 4u]);
        x1 = x1 ^ x0;
        if (round == 3u || round == 7u || round == 11u || round == 15u || round == 19u) {
            let k = (round + 1u) / 4u;
            let index_a = k % 3u;
            let index_b = (k + 1u) % 3u;
            let ka = select(select(ks2, ks1, index_a == 1u), ks0, index_a == 0u);
            let kb = select(select(ks2, ks1, index_b == 1u), ks0, index_b == 0u);
            x0 = x0 + ka;
            x1 = x1 + kb;
        }
    }
    return vec2<u32>(x0, x1);
}

// Two strictly-positive uniforms in (0, 1) from one Threefry call.
fn threefry_uniforms(index: u32, pair: u32, key0: u32) -> vec2<f32> {
    let out = threefry_u32(index, pair, key0, 0xC2B2AE3Du ^ (pair * 0x9E3779B9u));
    let u0 = (f32(out.x >> 8u) + 0.5) / 16777216.0;
    let u1 = (f32(out.y >> 8u) + 0.5) / 16777216.0;
    return vec2<f32>(u0, u1);
}

fn threefry_uniform(index: u32, key0: u32, key1: u32) -> f32 {
    let out = threefry_u32(index, 0u, key0, key1);
    return (f32(out.x >> 8u) + 0.5) / 16777216.0;
}

fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let radius = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;
    return vec2<f32>(radius * cos(theta), radius * sin(theta));
}