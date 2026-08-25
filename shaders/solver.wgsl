// ===========================================================================
// solver.wgsl - Pass 4: sustainable-spending bisection.
// ===========================================================================
//
// One thread per (batch simulation, allocation) runs a fixed 24-step parallel
// bisection over the annual net lifestyle spending w in [300, 10,000,000].
// test_solvency() walks the whole retirement lifecycle for a candidate w:
// account growth, cash-wedge establishment, RRSP meltdown, OAS clawback,
// Quebec tax interpolation, non-registered capital-gains taxation and the
// housing cost (market rent for renters, property taxes + condo for owners).
// The solved w is NET disposable lifestyle spending, so owning and renting
// rank on the same net scale.

fn test_solvency(w: f32, simulation: u32, allocation: u32) -> bool {
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

    // Bridge cash wedge: carve out cashWedgeYears of annual spending up
    // front, prorated across the accounts by their current weights.
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
    var solvent = true;

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

        if (remaining > 1e-2) {
            solvent = false;
        }
    }
    return solvent;
}

@compute @workgroup_size(64, 1, 1)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let simulation = gid.x;
    if (simulation >= params.generate.z) {
        return;
    }
    // Compatibility mode: one thread walks `dispatch.x` allocation columns
    // (stride = dispatch_y) so the runtime can shrink the dispatch grid
    // without splitting it into multiple dispatches (which some AMD D3D12
    // drivers silently corrupt). dispatch.x = 1 reproduces the original
    // one-column-per-thread behavior.
    let columns = max(1u, params.dispatch.x);
    let dispatch_y = max(1u, (params.dimensions.y + columns - 1u) / columns);
    for (var k = 0u; k < columns; k += 1u) {
        let allocation = gid.y + k * dispatch_y;
        if (allocation >= params.dimensions.y) {
            break;
        }
        var low = 300.0;
        var high = 10000000.0;
        for (var step = 0u; step < params.solver.z; step += 1u) {
            let middle = (low + high) * 0.5;
            if (test_solvency(middle, simulation, allocation)) {
                low = middle;
            } else {
                high = middle;
            }
        }
        spending_results[allocation * params.dimensions.x + params.generate.w + simulation] = (low + high) * 0.5;
    }
}