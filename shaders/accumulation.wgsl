// ===========================================================================
// accumulation.wgsl - Pass 3: retirement-account accumulation states.
// ===========================================================================
//
// One thread per (batch simulation, house strategy, accumulation path):
// gid.y = house * pathCount + path. Paths 0..4 stay 100% in one underlying
// fund; path 5 (DECLINING) switches VEQT -> VGRO -> VBAL and path 6 (RISING)
// switches VBAL -> VGRO -> VEQT on a full monthly schedule.
//
// House mechanics per path:
//   - HOUSE_NONE (renter): both savings streams merge; $8k/yr FHSA (max
//     $40k) earns the accumulation fund return and transfers tax-free into
//     the RRSP at FHSA year 15 (age 40) or at retirement, without consuming
//     RRSP room.
//   - Buyers: the house stream fills FHSA -> voluntary RRSP (HBP pool, max
//     $60k) -> house non-registered, all invested in the house allocation
//     (CASH = HISA, VBAL, VGRO or VEQT). The house is bought the month the
//     house fund first reaches the target capital (down payment + closing
//     costs). The mortgage amortizes to exactly zero at retirement
//     (N = retirement month - buy month), and the post-purchase cash-flow
//     spillover uses the market rent as the housing baseline (the buyer no
//     longer pays rent, so the rent is credited toward the mortgage).
// A surplus boosts retirement savings; a shortfall cancels the year's
// contributions and then draws down the already-accumulated retirement
// accounts (TFSA -> non-registered -> RRSP at the marginal rate) until they
// are exhausted.

fn accum_fund(path: u32, month: u32) -> u32 {
    if (path < 5u) {
        return path;
    }
    let months = params.dimensions.w;
    let half = (months + 1u) / 2u;
    let quarter = (months + 2u) / 4u;
    if (path == 5u) {  // DECLINING
        if (month < half) {
            return 0u;
        }
        if (month < half + quarter) {
            return 3u;
        }
        return 4u;
    }
    // RISING
    if (month < quarter) {
        return 4u;
    }
    if (month < quarter * 2u) {
        return 3u;
    }
    return 0u;
}

fn employer_match_contrib(net_avail: f32, salary: f32, tax_rate: f32, rrsp_room: f32) -> vec2<f32> {
    // Returns (total match gross into the RRSP, net cost to the saver).
    let match_rate = params.constants2.y;
    let match_pct = params.constants2.z;
    let max_ee_match_gross = salary * match_rate;
    let max_ee_match_net = max_ee_match_gross * (1.0 - tax_rate);
    let ee_match_net = min(net_avail, max_ee_match_net);
    let ee_match_gross = ee_match_net / max(1e-5, 1.0 - tax_rate);

    let potential_total_match = ee_match_gross * (1.0 + match_pct);
    let total_match_gross = min(potential_total_match, rrsp_room);
    let actual_ee_match_gross = total_match_gross / max(1e-5, 1.0 + match_pct);
    let actual_ee_match_net = actual_ee_match_gross * (1.0 - tax_rate);
    return vec2<f32>(total_match_gross, actual_ee_match_net);
}

fn after_match_contrib(net_rem: f32, tax_rate: f32, rrsp_room: f32, tfsa_room: f32) -> vec4<f32> {
    // TFSA -> voluntary RRSP -> non-registered overflow.
    // Returns (tfsa_add, rrsp_add, non_reg_add, non_reg_acb_add).
    var remaining = max(0.0, net_rem);

    let tfsa_contrib = min(remaining, tfsa_room);
    remaining -= tfsa_contrib;

    let wanted_vol_rrsp_gross = remaining / max(1e-5, 1.0 - tax_rate);
    let vol_rrsp_gross = min(wanted_vol_rrsp_gross, rrsp_room);
    remaining -= vol_rrsp_gross * (1.0 - tax_rate);

    return vec4<f32>(tfsa_contrib, vol_rrsp_gross, remaining, remaining);
}

fn house_fund_return(house: u32, simulation: u32, month: u32) -> f32 {
    if (house == 1u) {
        return params.constants0.z;  // HOUSE_CASH: HISA monthly return
    }
    if (house == 2u) {
        return return_at(simulation, month, 4u);  // HOUSE_VBAL
    }
    if (house == 3u) {
        return return_at(simulation, month, 3u);  // HOUSE_VGRO
    }
    return return_at(simulation, month, 0u);  // HOUSE_VEQT
}

@compute @workgroup_size(64, 1, 1)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let simulation = gid.x;
    let combined = gid.y;
    let path_count = u32(params.constants2.w);
    let house = combined / path_count;
    let path = combined % path_count;
    if (simulation >= params.generate.z || house >= house_count() || path >= path_count) {
        return;
    }

    var tfsa = 0.0;
    var rrsp = 0.0;
    var non_reg = 0.0;
    var non_reg_acb = 0.0;
    var tfsa_used = 0.0;
    var rrsp_used = 0.0;

    // House-fund buckets (buyers only; renters use just the FHSA).
    var fhsa = 0.0;
    var hbp_rrsp = 0.0;
    var h_non_reg = 0.0;
    var h_non_reg_acb = 0.0;
    var fhsa_total = 0.0;
    var hbp_total = 0.0;
    var house_bought = 0.0;
    var buy_month = 0.0;
    var mortgage_payment = 0.0;
    var hbp_repay_monthly = 0.0;

    let target_capital = house_const_at(0u);
    let mortgage_principal = house_const_at(1u);
    let mortgage_monthly_rate = house_const_at(2u);
    let property_taxes_condo = house_const_at(3u);
    let market_rent = house_const_at(4u);
    let fhsa_annual = house_const_at(5u);
    let fhsa_max = house_const_at(6u);
    let hbp_max = house_const_at(7u);
    let hbp_years = house_const_at(8u);
    let accum_months = params.dimensions.w;
    let is_renter = house == 0u;

    for (var month = 0u; month < accum_months; month += 1u) {
        let fund = accum_fund(path, month);
        let r = return_at(simulation, month, fund);
        tfsa *= 1.0 + r;
        rrsp *= 1.0 + r;

        let distribution = non_reg * params.constants0.x;
        let distribution_net = distribution * (1.0 - params.constants0.y);
        non_reg = non_reg * (1.0 + r - params.constants0.x) + distribution_net;
        non_reg_acb += distribution_net;

        if (is_renter) {
            fhsa *= 1.0 + r;
        } else {
            // House-fund buckets ride the house allocation until purchase;
            // after purchase any leftover balance switches to the
            // accumulation allocation.
            let r_house = select(house_fund_return(house, simulation, month), r, house_bought > 0.5);
            fhsa *= 1.0 + r_house;
            hbp_rrsp *= 1.0 + r_house;
            let h_distribution = h_non_reg * params.constants0.x;
            let h_distribution_net = h_distribution * (1.0 - params.constants0.y);
            h_non_reg = h_non_reg * (1.0 + r_house - params.constants0.x) + h_distribution_net;
            h_non_reg_acb += h_distribution_net;
        }

        // HBP repayments: 1/15th of the HBP withdrawal each year, paid into
        // the retirement RRSP every month.
        if (house_bought > 0.5) {
            rrsp += hbp_repay_monthly;
        }

        if (month % 12u == 0u) {
            let age = params.calendar.y + month / 12u;
            if (age >= params.calendar.z && age < params.calendar.w) {
                let year = age - params.calendar.z;
                let career_values = career0_at(year);
                let layoff_mult = layoffs_at(simulation, year);
                let net_ret = career_values.x * layoff_mult;
                let net_house = career_values.y * layoff_mult;
                let tax_rate = career_values.z;
                let salary = career_scalar(year, 5u) * layoff_mult;
                let rrsp_room = max(0.0, career_scalar(year, 4u) - rrsp_used);
                let tfsa_room = max(0.0, career_values.w - tfsa_used);

                if (is_renter) {
                    // Merged stream = legacy retirement stream + house stream.
                    // Employer match keeps priority 1, then the FHSA ($8k/yr,
                    // $40k total) with its upfront Quebec deduction, then the
                    // TFSA -> voluntary RRSP -> non-registered order. The FHSA
                    // transfers into the RRSP at FHSA year 15 (age 40).
                    var merged = net_ret + net_house;
                    let match_contrib = employer_match_contrib(merged, salary, tax_rate, rrsp_room);
                    rrsp += match_contrib.x;
                    rrsp_used += match_contrib.x;
                    var rem = max(0.0, merged - match_contrib.y);
                    if (fhsa_total < fhsa_max) {
                        let gross = min(fhsa_annual, min(fhsa_max - fhsa_total, rem / max(1e-5, 1.0 - tax_rate)));
                        fhsa += gross;
                        fhsa_total += gross;
                        rem -= gross * (1.0 - tax_rate);
                    }
                    if (year == 15u && fhsa > 0.0) {
                        rrsp += fhsa;
                        fhsa = 0.0;
                    }
                    let after = after_match_contrib(rem, tax_rate, rrsp_room - match_contrib.x, tfsa_room);
                    tfsa += after.x;
                    tfsa_used += after.x;
                    rrsp += after.y;
                    rrsp_used += after.y;
                    non_reg += after.z;
                    non_reg_acb += after.w;
                } else if (house_bought < 0.5) {
                    // Pre-purchase with the strict RRSP headroom invariant:
                    // employer match (priority 1) consumes room first, then
                    // the HBP-pool RRSP (priority 2) from the house stream,
                    // then the retirement voluntary RRSP (priority 3) from
                    // whatever room remains - no dollar is double-counted.
                    let combined = net_ret + net_house;
                    let match_contrib = employer_match_contrib(combined, salary, tax_rate, rrsp_room);
                    rrsp += match_contrib.x;
                    rrsp_used += match_contrib.x;
                    var room_left = max(0.0, rrsp_room - match_contrib.x);
                    let match_cost = match_contrib.y;
                    let ret_rem = max(0.0, net_ret - match_cost);
                    var house_net = max(0.0, net_house - max(0.0, match_cost - net_ret));

                    if (fhsa_total < fhsa_max) {
                        let gross = min(fhsa_annual, min(fhsa_max - fhsa_total, house_net / max(1e-5, 1.0 - tax_rate)));
                        fhsa += gross;
                        fhsa_total += gross;
                        house_net -= gross * (1.0 - tax_rate);
                    }
                    if (house_net > 0.0 && hbp_total < hbp_max && room_left > 0.0) {
                        let gross = min(house_net / max(1e-5, 1.0 - tax_rate), min(hbp_max - hbp_total, room_left));
                        hbp_rrsp += gross;
                        hbp_total += gross;
                        rrsp_used += gross;
                        house_net -= gross * (1.0 - tax_rate);
                        room_left -= gross;
                    }
                    h_non_reg += house_net;
                    h_non_reg_acb += house_net;

                    let after = after_match_contrib(ret_rem, tax_rate, room_left, tfsa_room);
                    tfsa += after.x;
                    tfsa_used += after.x;
                    rrsp += after.y;
                    rrsp_used += after.y;
                    non_reg += after.z;
                    non_reg_acb += after.w;
                } else {
                    // Post-purchase spillover with the rent baseline: the
                    // buyer no longer pays the market rent, so the rent is
                    // credited toward the housing obligation. A surplus boosts
                    // retirement savings; a shortfall cancels contributions
                    // and draws down accumulated accounts (TFSA -> non-reg ->
                    // RRSP), never creating debt.
                    let housing_cost = mortgage_payment * 12.0
                        + property_taxes_condo * 12.0
                        + hbp_repay_monthly * 12.0;
                    let delta = net_house + market_rent * 12.0 - housing_cost;
                    var actual_ret = net_ret + delta;
                    let match_contrib = employer_match_contrib(max(0.0, actual_ret), salary, tax_rate, rrsp_room);
                    rrsp += match_contrib.x;
                    rrsp_used += match_contrib.x;
                    let after = after_match_contrib(max(0.0, actual_ret - match_contrib.y), tax_rate, rrsp_room - match_contrib.x, tfsa_room);
                    tfsa += after.x;
                    tfsa_used += after.x;
                    rrsp += after.y;
                    rrsp_used += after.y;
                    non_reg += after.z;
                    non_reg_acb += after.w;
                    // Negative spillover: the housing shortfall draws down the
                    // already-accumulated retirement accounts.
                    var shortfall = max(0.0, -actual_ret);
                    if (shortfall > 0.0) {
                        let tfsa_take = min(tfsa, shortfall);
                        tfsa -= tfsa_take;
                        shortfall -= tfsa_take;
                        if (shortfall > 0.0 && non_reg > 0.0) {
                            let nr_take = min(non_reg, shortfall);
                            non_reg_acb -= non_reg_acb * (nr_take / non_reg);
                            non_reg -= nr_take;
                            shortfall -= nr_take;
                        }
                        if (shortfall > 0.0 && rrsp > 0.0) {
                            let rrsp_take = min(rrsp, shortfall / max(1e-5, 1.0 - tax_rate));
                            rrsp -= rrsp_take;
                            shortfall -= rrsp_take * (1.0 - tax_rate);
                        }
                    }
                }
            }
        }

        // Stochastic purchase: the house is bought the first month the house
        // fund reaches the target capital (before retirement by construction
        // of this loop). FHSA is spent first (tax-free), then the HBP-pool
        // RRSP withdrawal (up to $60k), then house non-registered.
        if (!is_renter && house_bought < 0.5) {
            let house_fund = fhsa + hbp_rrsp + h_non_reg;
            if (house_fund >= target_capital) {
                house_bought = 1.0;
                buy_month = f32(month);
                var needed = target_capital;
                let fhsa_use = min(fhsa, needed);
                fhsa -= fhsa_use;
                needed -= fhsa_use;
                let hbp_use = min(min(hbp_rrsp, hbp_max), needed);
                hbp_rrsp -= hbp_use;
                needed -= hbp_use;
                if (needed > 0.0 && h_non_reg > 0.0) {
                    let fraction = needed / h_non_reg;
                    h_non_reg_acb -= h_non_reg_acb * fraction;
                    h_non_reg -= needed;
                }
                let n_months = max(f32(accum_months) - f32(month), 1.0);
                let growth = pow(1.0 + mortgage_monthly_rate, n_months);
                mortgage_payment = mortgage_principal * mortgage_monthly_rate * growth / max(growth - 1.0, 1e-12);
                hbp_repay_monthly = hbp_use / hbp_years / 12.0;
            }
        }
    }

    // At retirement: house-fund balances never spent on a purchase roll into
    // retirement assets. FHSA and HBP-pool RRSP balances transfer into the
    // RRSP tax-free; house non-registered joins the non-registered account.
    if (fhsa > 0.0) {
        rrsp += fhsa;
        fhsa = 0.0;
    }
    if (hbp_rrsp > 0.0) {
        rrsp += hbp_rrsp;
        hbp_rrsp = 0.0;
    }
    if (h_non_reg > 0.0) {
        non_reg += h_non_reg;
        non_reg_acb += h_non_reg_acb;
        h_non_reg = 0.0;
    }

    states_set(house, path, simulation, vec4<f32>(tfsa, rrsp, non_reg, non_reg_acb));
    if (path == 0u) {
        house_outcomes_set(house, simulation, vec2<f32>(house_bought, buy_month));
    }
}