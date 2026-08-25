// ===========================================================================
// quantiles.wgsl - GPU-native 2-pass histogram quantile reduction.
// ===========================================================================
//
// This is a SEPARATE compute module with its own Params struct and bindings,
// run after the main pipeline to reduce the per-(allocation, simulation)
// spending buffer to 201 quantiles per allocation. The GPU-to-CPU readback
// drops from simulations*4 bytes to 201*4 bytes per allocation.
//
// One 256-thread workgroup per allocation builds a coarse 512-bin log2
// histogram of the monthly spending values in workgroup shared memory
// (workgroupBarrier), locates each of the 201 target quantiles, then refines
// each quantile with a 128-sub-bin fine histogram over its coarse bin.

struct Params {
    dimensions: vec4<u32>,   // total simulations, allocations, total months, accumulation months
    solver: vec4<u32>,
    calendar: vec4<u32>,
    constants0: vec4<f32>,
    constants1: vec4<f32>,
    constants2: vec4<f32>,
    generate: vec4<u32>,     // PRNG seed, skew-t df, batch simulation count, batch offset
    generate1: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> params: Params;
// Max sustainable annual spending per (allocation, total simulation).
@group(0) @binding(1) var<storage, read> spending_results: array<f32>;
// 201 monthly-spending quantiles per allocation (P0, P0.5, P1, ..., P100).
@group(0) @binding(2) var<storage, read_write> quantile_output: array<f32>;

const QUANTILE_WORKGROUP = 256u;
const HIST_BINS = 512u;
const REFINE_BATCH = 16u;
const REFINE_SUBBINS = 128u;
var<workgroup> hist: array<atomic<u32>, HIST_BINS>;
var<workgroup> partial_min: array<f32, QUANTILE_WORKGROUP>;
var<workgroup> partial_max: array<f32, QUANTILE_WORKGROUP>;
var<workgroup> log_min_v: f32;
var<workgroup> log_scale_v: f32;
var<workgroup> bin_scale_v: f32;
var<workgroup> step_v: f32;
var<workgroup> fine_scale_v: f32;
var<workgroup> q_bins: array<u32, 201>;
var<workgroup> q_cums: array<u32, 201>;
var<workgroup> q_counts: array<u32, 201>;
var<workgroup> fine_hist: array<atomic<u32>, REFINE_BATCH * REFINE_SUBBINS>;

@compute @workgroup_size(256, 1, 1)
fn quantiles(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // One 256-thread workgroup per allocation: the allocation index is the
    // workgroup id, and every thread in the workgroup cooperates on the same
    // shared-memory histograms.
    let allocation = wgid.x;
    let n = params.dimensions.x;
    if (allocation >= params.dimensions.y) {
        return;
    }

    for (var i = lid; i < HIST_BINS; i += QUANTILE_WORKGROUP) {
        atomicStore(&hist[i], 0u);
    }
    workgroupBarrier();

    // Pass 1: per-thread min/max, then a workgroup tree reduction.
    var local_min = 1.0e30;
    var local_max = -1.0e30;
    for (var j = lid; j < n; j += QUANTILE_WORKGROUP) {
        let v = spending_results[allocation * n + j] / 12.0;
        local_min = min(local_min, v);
        local_max = max(local_max, v);
    }
    partial_min[lid] = local_min;
    partial_max[lid] = local_max;
    workgroupBarrier();

    var stride = QUANTILE_WORKGROUP / 2u;
    while (stride > 0u) {
        if (lid < stride) {
            partial_min[lid] = min(partial_min[lid], partial_min[lid + stride]);
            partial_max[lid] = max(partial_max[lid], partial_max[lid + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (lid == 0u) {
        let vmin = partial_min[0u];
        let vmax = partial_max[0u];
        // Clamp the log scale to a tiny positive value so the degenerate
        // all-equal case flows through the same code path: with scale ~1e-12
        // every value lands in bin 0 and every quantile evaluates to vmin.
        log_min_v = log2(vmin);
        log_scale_v = max(log2(vmax) - log2(vmin), 1e-12);
        bin_scale_v = f32(HIST_BINS) / log_scale_v;
        step_v = log_scale_v / f32(HIST_BINS);
        fine_scale_v = f32(REFINE_SUBBINS) / step_v;
    }
    workgroupBarrier();

    // Pass 2: coarse log2-spaced histogram.
    for (var j = lid; j < n; j += QUANTILE_WORKGROUP) {
        let v = spending_results[allocation * n + j] / 12.0;
        let bin = min(u32((log2(v) - log_min_v) * bin_scale_v), HIST_BINS - 1u);
        atomicAdd(&hist[bin], 1u);
    }
    workgroupBarrier();

    // Pass 3: single-thread walk of the cumulative coarse histogram records
    // the locating bin, cumulative count before it and its count per quantile.
    // P0 and P100 are exact (sample min / max).
    if (lid == 0u) {
        quantile_output[allocation * 201u] = partial_min[0u];
        quantile_output[allocation * 201u + 200u] = partial_max[0u];
        var bin = 0u;
        var cum_before = 0u;
        for (var i = 1u; i <= 199u; i += 1u) {
            let position = f32(n - 1u) * f32(i) / 200.0;
            while (bin < HIST_BINS - 1u && f32(cum_before + atomicLoad(&hist[bin])) < position) {
                cum_before += atomicLoad(&hist[bin]);
                bin += 1u;
            }
            q_bins[i] = bin;
            q_cums[i] = cum_before;
            q_counts[i] = atomicLoad(&hist[bin]);
        }
    }
    workgroupBarrier();

    // Pass 4: refine each quantile with a 128-sub-bin fine histogram over its
    // coarse bin, then interpolate linearly in value space within the sub-bin.
    for (var batch = 0u; batch * REFINE_BATCH < 199u; batch += 1u) {
        for (var i = lid; i < REFINE_BATCH * REFINE_SUBBINS; i += QUANTILE_WORKGROUP) {
            atomicStore(&fine_hist[i], 0u);
        }
        workgroupBarrier();
        let base = batch * REFINE_BATCH;
        for (var j = lid; j < n; j += QUANTILE_WORKGROUP) {
            let lv = log2(spending_results[allocation * n + j] / 12.0);
            for (var g = 0u; g < REFINE_BATCH; g += 1u) {
                let qi = base + g;
                if (qi > 0u && qi < 200u) {
                    let edge = log_min_v + f32(q_bins[qi]) * step_v;
                    let rel = (lv - edge) * fine_scale_v;
                    if (rel >= 0.0 && rel < f32(REFINE_SUBBINS)) {
                        atomicAdd(&fine_hist[g * REFINE_SUBBINS + u32(rel)], 1u);
                    }
                }
            }
        }
        workgroupBarrier();
        if (lid == 0u) {
            for (var g = 0u; g < REFINE_BATCH; g += 1u) {
                let qi = base + g;
                if (qi > 0u && qi < 200u) {
                    let rank = f32(n - 1u) * f32(qi) / 200.0;
                    var p_fine = clamp(rank - f32(q_cums[qi]), 0.0, f32(q_counts[qi]));
                    var sub = 0u;
                    var cum = 0u;
                    while (sub < REFINE_SUBBINS - 1u && f32(cum + atomicLoad(&fine_hist[g * REFINE_SUBBINS + sub])) < p_fine) {
                        cum += atomicLoad(&fine_hist[g * REFINE_SUBBINS + sub]);
                        sub += 1u;
                    }
                    let sub_count = atomicLoad(&fine_hist[g * REFINE_SUBBINS + sub]);
                    var sub_frac = 0.0;
                    if (sub_count > 0u) {
                        sub_frac = (p_fine - f32(cum)) / f32(sub_count);
                    }
                    sub_frac = clamp(sub_frac, 0.0, 1.0);
                    let edge = log_min_v + f32(q_bins[qi]) * step_v;
                    let lo = exp2(edge + f32(sub) * step_v / f32(REFINE_SUBBINS));
                    let hi = exp2(edge + f32(sub + 1u) * step_v / f32(REFINE_SUBBINS));
                    quantile_output[allocation * 201u + qi] = lo + sub_frac * (hi - lo);
                }
            }
        }
        workgroupBarrier();
    }
}