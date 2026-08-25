// ===========================================================================
// drawdown_ui.wgsl - Pass 5: Composite Ulcer Index (UI) drawdown tracking.
// ===========================================================================
//
// Drawdowns are measured on CUMULATIVE RETURN INDICES rather than net liquid
// wealth, so planned decumulation (spending the portfolio down to $0 by the
// death age) does not masquerade as a drawdown:
//     accumulation: R^accum_t = prod(1 + r_accum)   (allocation glidepath)
//     retirement:   R^retire_t = prod(1 + r_phase)  (fund return on the
//                   invested portion, HISA return on any cash wedge, blended
//                   by the actual wealth split each month)
// with D_t = (R_t - max_{s<=t} R_s) / max_{s<=t} R_s * 100 (<= 0), phase UI =
// sqrt(mean over phase months of D_t^2), and the per-path composite
//     UI_comp = wB*UI_bridge + wP*UI_post + wA*UI_accum
// with default weights (0.60, 0.25, 0.15) re-normalized to (0, 0.65, 0.35)
// when the bridge phase has zero months. The per-path composite is written to
// the scratch drawdown region, indexed simulation * allocation count +
// allocation, and reduced to the strategy mean on the CPU.
//
// The retirement walk below is the exact test_solvency lifecycle (needed for
// the cash-wedge split) and starts from the stored post-accumulation state of
// the (house, accumulation path) pair.

fn drawdown_ui(w: f32, simulation: u32, allocation: u32) -> vec4<f32> {
    let allocation_info = allocations[allocation * 3u];
    let accumulation_schedule = allocations[allocation * 3u + 1u];
    let retirement_schedule = allocations[allocation * 3u + 2u];
    let accumulation_fund = allocation_info.x;
    let bridge_fund = allocation_info.y;
    let post_fund = allocation_info.z;
    let bridge_cash = (allocation_info.w & 1u) != 0u;
    let post_cash = (allocation_info.w & 2u) != 0u;
    let house = (allocation_info.w >> 2u) & 7u;

    let accum_months = params.dimensions.w;
    let bridge_months = params.calendar.x;

    // --- Accumulation-phase drawdown on the cumulative return index of the
    // --- allocation's accumulation glidepath.
    var r_accum = 1.0;
    var peak_accum = 1.0;
    var sum_sq_accum = 0.0;

    for (var month = 0u; month < accum_months; month += 1u) {
        let fund = accum_fund(accumulation_fund, month);
        let r = return_at(simulation, month, fund);

        r_accum *= 1.0 + r;
        peak_accum = max(peak_accum, r_accum);
        let dd_accum = (r_accum - peak_accum) / peak_accum * 100.0;
        sum_sq_accum += dd_accum * dd_accum;
    }

    // --- Retirement months: byte-for-byte test_solvency at the solved w,
    // --- starting from the stored (house, accumulation path) state.
    let initial = states_at(house, accumulation_fund, simulation);
    var tfsa = initial.x;
    var rrsp = initial.y;
    var non_reg = initial.z;
    var non_reg_acb = initial.w;
    let house_bought = house_outcomes_at(house, simulation).x;
    let housing_cost = select(house_const_at(4u), house_const_at(3u), house_bought > 0.5);

    var cash_wedge = 0.0;
    if (bridge_cash) {
        let total = tfsa + rrsp + non_reg;
        let actual_cash = min(total, w * params.constants1.y);
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
    var r_retire = 1.0;
    var peak_retire = 1.0;
    var sum_sq_bridge = 0.0;
    var sum_sq_post = 0.0;

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

        // Cumulative return index: the fund return applies to the invested
        // (non-cash) portion and the HISA return to any cash wedge, blended
        // by the wealth split at the start of the month. With no assets left
        // the index is flat (no investment drawdown from an empty portfolio).
        let total_now = tfsa + rrsp + non_reg + cash_wedge;
        var blended = 1.0;
        if (total_now > 1e-10) {
            blended = ((tfsa + rrsp + non_reg) * (1.0 + r) + cash_wedge * (1.0 + params.constants0.z)) / total_now;
        }
        r_retire *= blended;
        peak_retire = max(peak_retire, r_retire);
        let dd_retire = (r_retire - peak_retire) / peak_retire * 100.0;
        if (month < params.calendar.x) {
            sum_sq_bridge += dd_retire * dd_retire;
        } else {
            sum_sq_post += dd_retire * dd_retire;
        }

        tfsa *= 1.0 + r;
        rrsp *= 1.0 + r;

        let distribution = non_reg * params.constants0.x;
        let distribution_net = distribution * (1.0 - params.constants0.y);
        non_reg = non_reg * (1.0 + r - params.constants0.x) + distribution_net;
        non_reg_acb += distribution_net;
        cash_wedge *= 1.0 + params.constants0.z;

        if (post_cash && !post_wedge_established && month == params.calendar.x) {
            let total = tfsa + rrsp + non_reg;
            let needed = max(0.0, w * params.constants1.y - cash_wedge);
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

        let cash_withdrawal = min(cash_wedge, remaining);
        cash_wedge -= cash_withdrawal;
        remaining -= cash_withdrawal;

        if (month < params.calendar.x) {
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
    }

    let ui_accum = sqrt(sum_sq_accum / max(f32(accum_months), 1.0));
    let ui_bridge = sqrt(sum_sq_bridge / max(f32(bridge_months), 1.0));
    let ui_post = sqrt(sum_sq_post / max(f32(params.solver.x - bridge_months), 1.0));
    var w_b = 0.60;
    var w_p = 0.25;
    var w_a = 0.15;
    if (bridge_months == 0u) {
        w_b = 0.0;
        w_p = 0.65;
        w_a = 0.35;
    }
    let ui_comp = w_b * ui_bridge + w_p * ui_post + w_a * ui_accum;
    return vec4<f32>(ui_accum, ui_bridge, ui_post, ui_comp);
}

@compute @workgroup_size(64, 1, 1)
fn track_drawdowns(@builtin(global_invocation_id) gid: vec3<u32>) {
    let simulation = gid.x;
    if (simulation >= params.generate.z) {
        return;
    }
    // Compatibility mode: mirror of solve(); see solver.wgsl. dispatch.x = 1
    // reproduces the original one-column-per-thread behavior.
    let columns = max(1u, params.dispatch.x);
    let dispatch_y = max(1u, (params.dimensions.y + columns - 1u) / columns);
    for (var k = 0u; k < columns; k += 1u) {
        let allocation = gid.y + k * dispatch_y;
        if (allocation >= params.dimensions.y) {
            break;
        }
        let global_sim = params.generate.w + simulation;
        let w = spending_results[allocation * params.dimensions.x + global_sim];
        let ui = drawdown_ui(w, simulation, allocation);
        let out_index = drawdown_region_offset() + simulation * params.dimensions.y + allocation;
        scratch[out_index] = ui.w;
    }
}