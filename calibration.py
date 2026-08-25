"""Statistical calibration, buffer serialization and CPU reference samplers.

This module is the single home for everything that turns the raw price history
in ``downloaded_prices.csv`` into the skewed, heavy-tailed market model that
the shaders sample. It provides:

* :func:`load_returns`           - monthly simple returns for VEQT/VGRO/VBAL.
* :func:`calibrate_skew_t`       - multivariate skew-t (xi, omega, delta,
                                   correlation, Cholesky, nu) from history.
* :func:`build_model_buffer`     - the exact packed f32 model buffer the WGSL
                                   shaders consume (career + month + tax +
                                   skew-t + house constants).
* :func:`build_allocation_map`   - the 5,040-strategy phase schedule metadata.
* :func:`build_payload`          - the full JSON payload embedded in the HTML.
* :func:`returns_cpu`            - a deterministic NumPy reference of the WGSL
                                   Threefry skew-t sampler, used by the parity
                                   test to prove CPU/GPU agreement.

Every derived table (salary trajectory, net savings, tax tables, pension,
smile, intertemporal weights) is computed here from the configuration objects
in :mod:`config`, so there is one code path shared by the Python engine and the
HTML model payload.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config as cfg


# ---------------------------------------------------------------------------
# Historical returns
# ---------------------------------------------------------------------------
def load_returns(price_path) -> "object":
    """Read the price CSV and return a DataFrame of monthly simple returns.

    Columns are ``Date``, ``VEQT``, ``VGRO``, ``VBAL``. Prices are forwarded
    over missing values and the first (return-free) row is dropped.
    """
    import pandas as pd

    prices = pd.read_csv(price_path)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    prices[list(cfg.BASE_UNDERLYING)] = (
        prices[list(cfg.BASE_UNDERLYING)]
        .ffill()
        .dropna(subset=list(cfg.BASE_UNDERLYING))
        .reset_index(drop=True)
    )
    values = prices[list(cfg.BASE_UNDERLYING)].values
    returns_df = pd.DataFrame(values[1:] / values[:-1] - 1.0, columns=cfg.BASE_UNDERLYING)
    returns_df.insert(0, "Date", prices["Date"].iloc[1:].values)
    returns_df = returns_df.dropna(subset=list(cfg.BASE_UNDERLYING)).reset_index(drop=True)
    returns_df["Date"] = pd.to_datetime(returns_df["Date"])
    return returns_df


# ---------------------------------------------------------------------------
# Exact b_nu for integer degrees of freedom (no Lanczos approximation).
# ---------------------------------------------------------------------------
def _ln_gamma_integer(n: float) -> float:
    # ln Gamma(n) = ln((n-1)!)
    return sum(math.log(i) for i in range(1, int(n)))


def _ln_gamma_half_integer(k: float) -> float:
    # ln Gamma(k + 0.5) = ln((2k)! / (4^k k!) * sqrt(pi))
    total = 0.5 * math.log(math.pi)
    total += sum(math.log(i) for i in range(1, 2 * int(k) + 1))
    total -= int(k) * math.log(4)
    total -= sum(math.log(i) for i in range(1, int(k) + 1))
    return total


def b_nu(nu: int) -> float:
    """b_nu = sqrt(nu/pi) * Gamma((nu-1)/2) / Gamma(nu/2), exact for integer nu."""
    if nu <= 1:
        return 0.0
    half = (nu - 1) / 2.0
    other = nu / 2.0
    log_g1 = _ln_gamma_half_integer(half) if half != int(half) else _ln_gamma_integer(int(half))
    log_g2 = _ln_gamma_half_integer(other) if other != int(other) else _ln_gamma_integer(int(other))
    return math.sqrt(nu / math.pi) * math.exp(log_g1 - log_g2)


def _estimate_delta(returns: np.ndarray, cap: float) -> np.ndarray:
    centered = returns - returns.mean(axis=0)
    m2 = (centered ** 2).mean(axis=0)
    m3 = (centered ** 3).mean(axis=0)
    return cap * np.tanh((m3 / (m2 ** 1.5)) / 3.0)


def _clamp_delta(delta: np.ndarray, corr: np.ndarray, tol: float) -> np.ndarray:
    quadratic = float(delta @ np.linalg.inv(corr) @ delta)
    if quadratic >= tol:
        delta = delta * math.sqrt(tol / quadratic)
    return delta


def calibrate_skew_t(
    returns: np.ndarray,
    calibration: cfg.ModelCalibrationConfig,
    cma: cfg.CMAConfig,
) -> Dict[str, object]:
    """Calibrate the multivariate skew-t return model from an (N, 3) returns array.

    Matches the JavaScript ``calibrateReturnModel`` exactly (sample covariance,
    forward-looking CMA means, moment-based delta with clamp, closed-form omega
    and correlation, Cholesky of the residual with diagonal jitter).
    """
    count = returns.shape[0]
    means = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    diag = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(diag, diag)

    if cma.use_forward_looking_cmas:
        mean_returns = np.array(
            [
                math.log1p(cma.cmas[fund]) / 12.0 + 0.5 * covariance[i, i]
                for i, fund in enumerate(cfg.BASE_UNDERLYING)
            ]
        )
    else:
        mean_returns = means

    nu = calibration.skew_degrees_freedom
    b = b_nu(nu)
    delta = _clamp_delta(
        _estimate_delta(returns, calibration.delta_cap),
        correlation,
        calibration.delta_tolerance,
    )
    omega = np.sqrt(np.diag(covariance) / (nu / (nu - 2) - delta ** 2 * b ** 2))
    inverse_omega = np.diag(1.0 / omega)
    calibrated_correlation = ((nu - 2) / nu) * (
        inverse_omega @ covariance @ inverse_omega + b ** 2 * np.outer(delta, delta)
    )
    np.fill_diagonal(calibrated_correlation, 1.0)
    residual = calibrated_correlation - np.outer(delta, delta)

    cholesky = _jittered_cholesky(residual)
    xi = mean_returns - omega * delta * b
    return {
        "xi": xi,
        "omega": omega,
        "correlation": calibrated_correlation,
        "delta": delta,
        "cholesky": cholesky,
        "nu": nu,
        "mean_returns": mean_returns,
        "observations": count,
    }


def _jittered_cholesky(matrix: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() <= 1e-12:
        matrix = matrix + (-eigenvalues.min() + 1e-12) * np.eye(matrix.shape[0])
    for jitter in (0.0, 1e-10, 1e-8, 1e-6):
        candidate = matrix.copy()
        candidate[np.arange(matrix.shape[0]), np.arange(matrix.shape[0])] += jitter
        try:
            chol = np.linalg.cholesky(candidate)
            if np.isfinite(chol).all():
                return chol
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError("Return calibration could not produce a positive-definite skew-t covariance.")


# ---------------------------------------------------------------------------
# Deterministic skew-t sampler (NumPy reference, mirrors the WGSL sampler).
# ---------------------------------------------------------------------------
def _rotl32(x: np.ndarray, n: int) -> np.ndarray:
    n = n % 32
    return (((x << n) | (x >> ((32 - n) % 32))) & 0xFFFFFFFF).astype(np.uint32)


def _threefry_u32(ctr0, ctr1, key0, key1):
    """Vectorized Threefry-2x32-20, matching the WGSL ``threefry_u32``."""
    ks0 = np.uint32(key0 & 0xFFFFFFFF)
    ks1 = np.uint32(key1 & 0xFFFFFFFF)
    ks2 = np.uint32((np.uint32(key0) ^ np.uint32(key1) ^ np.uint32(0x1BD11BDA)) & 0xFFFFFFFF)
    ctr0 = np.asarray(ctr0, dtype=np.uint32)
    ctr1 = np.asarray(ctr1, dtype=np.uint32)
    x0 = np.array(ctr0 + ks0, dtype=np.uint32)
    x1 = np.array(ctr1 + ks1, dtype=np.uint32)
    rot = (13, 15, 26, 6)
    for round_ in range(20):
        x0 = np.array(x0 + x1, dtype=np.uint32)
        x1 = _rotl32(x1, rot[round_ % 4])
        x1 = np.array(x1 ^ x0, dtype=np.uint32)
        if round_ in (3, 7, 11, 15, 19):
            k = (round_ + 1) // 4
            index_a = k % 3
            index_b = (k + 1) % 3
            keys = (ks0, ks1, ks2)
            ka = keys[index_a]
            kb = keys[index_b]
            x0 = np.array(x0 + ka, dtype=np.uint32)
            x1 = np.array(x1 + kb, dtype=np.uint32)
    return x0, x1


def threefry_uniforms(index, pair, key0):
    """Two 24-bit uniforms in (0, 1), mirroring the WGSL helper exactly."""
    pair_mul = (np.uint64(pair) * np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)
    key1 = np.uint32((np.uint32(0xC2B2AE3D) ^ np.uint32(pair_mul)) & np.uint32(0xFFFFFFFF))
    out0, out1 = _threefry_u32(index, pair, key0, key1)
    u0 = ((out0 >> np.uint32(8)).astype(np.float64) + 0.5) / 16777216.0
    u1 = ((out1 >> np.uint32(8)).astype(np.float64) + 0.5) / 16777216.0
    return u0, u1


def _box_muller(u1, u2):
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = 6.283185307179586 * u2
    return radius * np.cos(theta), radius * np.sin(theta)


def returns_cpu(simulations: int, total_months: int, model: Dict[str, object],
                seed: int, config: Dict[str, object]) -> np.ndarray:
    """Deterministic CPU mirror of the WGSL ``generate_returns`` pass.

    Returns a float32 array of shape (simulations, total_months, 5) with the
    five underlying fund monthly returns. Used by the parity test.
    """
    nu = model["nu"]
    xi = model["xi"].astype(np.float64)
    omega = model["omega"].astype(np.float64)
    delta = model["delta"].astype(np.float64)
    cholesky = model["cholesky"]

    pairs_needed = (nu + 5) // 2
    counts = simulations * total_months
    months = np.tile(np.arange(total_months, dtype=np.uint32), simulations)
    sims = np.repeat(np.arange(simulations, dtype=np.uint32), total_months)
    counter = sims * np.uint32(total_months) + months

    scale = np.zeros(counts)
    skew = np.zeros(counts)
    fund_normal = np.zeros((counts, 3))
    base = np.zeros((counts, 3))
    for pair in range(pairs_needed):
        u0, u1 = threefry_uniforms(counter, np.uint32(pair), np.uint32(seed))
        n0, n1 = _box_muller(u0, u1)
        t0 = pair * 2
        t1 = pair * 2 + 1
        if t0 == 0:
            skew += np.abs(n0)
        elif t0 <= nu:
            scale += n0 * n0
        elif t0 == nu + 1:
            fund_normal[:, 0] += n0
        elif t0 == nu + 2:
            fund_normal[:, 1] += n0
        elif t0 == nu + 3:
            fund_normal[:, 2] += n0
        if t1 <= nu:
            scale += n1 * n1
        elif t1 == nu + 1:
            fund_normal[:, 0] += n1
        elif t1 == nu + 2:
            fund_normal[:, 1] += n1
        elif t1 == nu + 3:
            fund_normal[:, 2] += n1

    inv_scale = 1.0 / np.sqrt(np.maximum(scale / float(nu), 1e-12))
    for i in range(3):
        base[:, i] = xi[i] + omega[i] * (
            delta[i] * skew
            + fund_normal[:, 0] * cholesky[i, 0]
            + fund_normal[:, 1] * cholesky[i, 1]
            + fund_normal[:, 2] * cholesky[i, 2]
        ) * inv_scale
    base = np.maximum(base, -0.95)

    veqt = base[:, 0]
    cma_cfg = config["cma"]
    borrow = cma_cfg.real_borrow_rate_annual / 12.0
    fee_15 = cma_cfg.extra_mer_15 / 12.0
    fee_20 = cma_cfg.extra_mer_20 / 12.0
    out = np.stack(
        (
            veqt,
            np.maximum(1.5 * veqt - 0.5 * borrow - fee_15, -0.95),
            np.maximum(2.0 * veqt - borrow - fee_20, -0.95),
            base[:, 1],
            base[:, 2],
        ),
        axis=-1,
    )
    return out.reshape(simulations, total_months, 5).astype(np.float32)


# ---------------------------------------------------------------------------
# Derived lifecycle / fiscal tables (shared by engine and browser payload).
# ---------------------------------------------------------------------------
def _salaries_and_targets(config):
    lc = config["lifecycle"]
    co = config["career"]
    career_years = lc.retirement_age - lc.career_start_age
    salaries = np.zeros(career_years, dtype=np.float32)
    salary = co.starting_salary
    for index in range(career_years):
        age = lc.career_start_age + index
        growth = 0.0
        if index > 0:
            for start_age, end_age, rate in co.promotion_phases:
                if start_age <= age < end_age:
                    growth = rate
                    break
            salary *= 1.0 + growth
        salaries[index] = salary
    return salaries


def _monthly_tax_vector(gross, tax):
    brackets = [0.0] + [t / 12.0 for t in tax.tax_thresholds_annual]
    tax_value = np.zeros_like(gross, dtype=np.float64)
    for index in range(5):
        upper = brackets[index + 1] if index < 4 else gross
        tax_value += np.maximum(0.0, np.minimum(gross, upper) - brackets[index]) * tax.tax_rates[index]
    return tax_value


def _annual_tax_vector(gross, tax):
    income_tax = _monthly_tax_vector(np.asarray(gross, dtype=np.float64) / 12.0, tax) * 12.0
    qpp_tier1 = np.maximum(0.0, np.minimum(gross, tax.qpp_maximum_annual) - tax.qpp_basic_annual) * tax.qpp_rate
    qpp_tier2 = np.maximum(0.0, np.minimum(gross, tax.qpp_maximum_msga) - tax.qpp_maximum_annual) * tax.qpp_rate_tier2
    ei = np.minimum(gross, tax.ei_maximum_annual) * tax.ei_rate
    rqap = np.minimum(gross, tax.rqap_maximum_annual) * tax.rqap_rate
    return income_tax + qpp_tier1 + qpp_tier2 + ei + rqap


def _pension_amounts(config):
    lc = config["lifecycle"]
    co = config["career"]
    tax = config["tax"]
    career_years = lc.retirement_age - lc.career_start_age
    salary = co.starting_salary
    salary_sum = 0.0
    for index in range(career_years):
        age = lc.career_start_age + index
        if index > 0:
            for start_age, end_age, rate in co.promotion_phases:
                if start_age <= age < end_age:
                    salary *= 1.0 + rate
                    break
        salary_sum += min(1.0, salary / tax.qpp_maximum_annual)
    ratio = salary_sum / career_years if career_years > 0 else 0.0
    base_qpp = tax.max_qpp_age_65 * min(1.0, career_years / 40.0) * ratio
    eff_qpp_age = min(72.0, max(60.0, lc.pension_start_age))
    qpp_mult = 1.0 + (eff_qpp_age - 65.0) * tax.qpp_deferral_annual if eff_qpp_age >= 65 else 1.0 - (65.0 - eff_qpp_age) * 0.072
    qpp = base_qpp * qpp_mult
    eff_oas_age = min(70.0, max(65.0, lc.pension_start_age))
    oas_mult = min(tax.oas_deferral_cap, 1.0 + (eff_oas_age - 65.0) * tax.oas_deferral_annual)
    oas7074 = tax.max_oas_age_65 * oas_mult
    oas75 = oas7074 * tax.oas75_increase
    return {"qpp": qpp, "oas7074": oas7074, "oas75": oas75}


def _net_taxable_income(gross, age, config, oas7074, oas75):
    lc = config["lifecycle"]
    tax = config["tax"]
    oas_eligible = age >= lc.pension_start_age and age >= 65
    oas_max = (oas75 if age >= 75 else oas7074) / 12.0 if oas_eligible else 0.0
    clawback = min(oas_max, max(0.0, gross - tax.oas_clawback_threshold / 12.0) * tax.oas_clawback_rate) if oas_max > 0 else 0.0
    return gross - _monthly_tax_vector(gross, tax) - clawback


def _unique_sorted(values):
    return np.unique(np.round(np.asarray(values, dtype=np.float64), 6)).astype(np.float32)


def build_model_tables(config):
    """Compute the career, month0/month1, tax, smile and intertemporal tables.

    Returns a dict with every packed f32 array the shaders consume, computed
    from the supplied configuration (mirrors the JS ``buildDynamicModel``).
    """
    lc = config["lifecycle"]
    co = config["career"]
    hg = config["housing"]
    tax = config["tax"]
    cma = config["cma"]
    sm = config["smile"]
    cal = config["calibration"]

    career_years = lc.retirement_age - lc.career_start_age
    accum_months = (lc.retirement_age - lc.current_age) * 12
    bridge_months = (lc.pension_start_age - lc.retirement_age) * 12
    retire_months = (lc.death_age - lc.retirement_age) * 12

    salaries = _salaries_and_targets(config)
    net_salaries = salaries - _annual_tax_vector(salaries.astype(np.float64), tax)
    retirement_start_rate = co.retirement_savings_start_annual / float(net_salaries[0])
    house_start_rate = co.house_savings_start_annual / float(net_salaries[0])

    career = np.zeros(career_years * 6, dtype=np.float32)
    cumulative_rrsp = 0.0
    for index in range(career_years):
        salary = float(salaries[index])
        bracket_index = next(
            (i for i, threshold in enumerate(tax.tax_thresholds_annual) if salary < threshold),
            4,
        )
        rate = tax.tax_rates[bracket_index]
        retirement_rate = min(retirement_start_rate + index * co.retirement_savings_escalation_rate, co.savings_max_fraction)
        house_rate = min(house_start_rate + index * co.house_savings_escalation_rate, co.house_savings_max_fraction)
        tfsa_room = (lc.career_start_age - 18 + 1) * co.tfsa_annual_limit - co.other_tfsa + index * co.tfsa_annual_limit
        row = [
            retirement_rate * float(net_salaries[index]),
            house_rate * float(net_salaries[index]),
            rate,
            tfsa_room,
            cumulative_rrsp,
            salary,
        ]
        career[index * 6:index * 6 + 6] = row
        cumulative_rrsp += min(co.rrsp_contribution_rate * salary, co.rrsp_max_contribution)

    pension = _pension_amounts(config)

    month0 = np.zeros(retire_months * 4, dtype=np.float32)
    month1 = np.zeros(retire_months * 4, dtype=np.float32)
    smile = np.zeros(retire_months, dtype=np.float32)
    smile_current = 1.0
    for month in range(retire_months):
        age = lc.retirement_age + month / 12.0
        smile_phase = next(
            ((start, end, change) for start, end, change in sm.smile_schedule
             if start <= age < end),
            None,
        )
        smile[month] = smile_current
        if smile_phase and smile_phase[2] != 0.0:
            smile_current *= 1.0 + smile_phase[2] / 12.0
        qpp_monthly = pension["qpp"] / 12.0 if age >= lc.pension_start_age else 0.0
        oas_eligible = age >= lc.pension_start_age and age >= 65
        oas_monthly = ((pension["oas75"] if age >= 75 else pension["oas7074"]) / 12.0) if oas_eligible else 0.0
        gross_pension = qpp_monthly + oas_monthly
        health = lc.post50_healthcare_annual / 12.0 if age < lc.healthcare_end_age else 0.0
        month0[month * 4:month * 4 + 4] = [
            smile[month] / 12.0, health, gross_pension, 0.0,
        ]
        month0[month * 4 + 3] = _net_taxable_income(gross_pension, age, config, pension["oas7074"], pension["oas75"]) if gross_pension > 0 else 0.0
        whole_age = int(math.floor(age))
        rrif = 0.0 if whole_age < 71 else (0.2 if whole_age >= 95 else tax_rrif_factor(tax, whole_age))
        oas_max = ((pension["oas75"] if age >= 75 else pension["oas7074"]) / 12.0) if oas_eligible else 0.0
        month1[month * 4:month * 4 + 4] = [rrif, oas_max, 0.0, 0.0]

    # Intertemporal (mortality-adjusted, discounted) weights.
    adjusted_mortality = np.asarray(MORTALITY_ANNUAL_PROBABILITY, dtype=np.float64) ** lc.mortality_reduction_factor
    cpm_weights = np.zeros(retire_months, dtype=np.float32)
    weight_sum = 0.0
    for month in range(retire_months):
        position = month * (len(adjusted_mortality) - 1) / max(1, retire_months - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        survival = adjusted_mortality[low] + (adjusted_mortality[high] - adjusted_mortality[low]) * (position - low)
        weight = (1.0 + lc.discount_rate_annual / 12.0) ** (-month) * survival
        cpm_weights[month] = weight
        weight_sum += weight
    cpm_weights /= weight_sum

    # Tax interpolation tables.
    gross_pre = np.concatenate(([0.0], np.asarray(tax.tax_thresholds_annual, dtype=np.float32) / 12.0, [1_000_000.0 / 12.0]))
    net_pre = np.array(
        [_net_taxable_income(v, lc.pension_start_age - 1, config, pension["oas7074"], pension["oas75"]) for v in gross_pre],
        dtype=np.float32,
    )

    def post_grid(oas):
        oas_monthly = oas / 12.0
        threshold = tax.oas_clawback_threshold / 12.0
        raw = [
            0.0,
            tax.tax_thresholds_annual[0] / 12.0,
            tax.tax_thresholds_annual[1] / 12.0,
            threshold,
            tax.tax_thresholds_annual[2] / 12.0,
            threshold + oas_monthly / tax.oas_clawback_rate,
            tax.tax_thresholds_annual[3] / 12.0,
            1_000_000.0 / 12.0,
        ]
        return _unique_sorted(raw)

    gross70 = post_grid(pension["oas7074"])
    gross75 = post_grid(pension["oas75"])
    net70 = np.array(
        [_net_taxable_income(v, max(70, lc.pension_start_age), config, pension["oas7074"], pension["oas75"]) for v in gross70],
        dtype=np.float32,
    )
    net75 = np.array(
        [_net_taxable_income(v, 75, config, pension["oas7074"], pension["oas75"]) for v in gross75],
        dtype=np.float32,
    )
    tax_values = np.zeros(54, dtype=np.float32)
    tax_values[0:6] = gross_pre
    tax_values[6:12] = net_pre
    tax_values[12:20] = gross70
    tax_values[20:28] = net70
    tax_values[28:36] = gross75
    tax_values[36:44] = net75
    tax_values[44:48] = np.asarray(tax.tax_thresholds_annual, dtype=np.float32) / 12.0
    # Note: index 48 is an unused pad; the five tax rates occupy slots 49-53,
    # exactly as the WGSL monthly_tax() and the JS buildDynamicModel expect.
    tax_values[49:54] = np.asarray(tax.tax_rates, dtype=np.float32)

    return {
        "career_years": career_years,
        "accum_months": accum_months,
        "bridge_months": bridge_months,
        "retire_months": retire_months,
        "career": career,
        "month0": month0,
        "month1": month1,
        "tax_values": tax_values,
        "smile": smile,
        "cpm_weights": cpm_weights,
        "pension": pension,
        "salaries": salaries,
        "net_salaries": net_salaries,
    }


def tax_rrif_factor(tax: cfg.TaxFiscalConfig, age: int) -> float:
    """RRIF minimum-withdrawal factor by integer age (71-94)."""
    factors = {
        71: 0.0528, 72: 0.0540, 73: 0.0553, 74: 0.0567, 75: 0.0582,
        76: 0.0598, 77: 0.0617, 78: 0.0636, 79: 0.0658, 80: 0.0682,
        81: 0.0708, 82: 0.0738, 83: 0.0771, 84: 0.0808, 85: 0.0851,
        86: 0.0899, 87: 0.0955, 88: 0.1021, 89: 0.1099, 90: 0.1192,
        91: 0.1306, 92: 0.1449, 93: 0.1634, 94: 0.1879,
    }
    return factors.get(age, 0.2)


def house_constants(config) -> np.ndarray:
    """The ten f32 house constants appended to the model buffer."""
    lc = config["lifecycle"]
    hg = config["housing"]
    target_capital = hg.property_value * hg.down_payment_fraction + hg.closing_costs
    mortgage_principal = hg.property_value * (1.0 - hg.down_payment_fraction)
    mortgage_monthly_rate = (1.0 + hg.real_mortgage_rate_annual) ** (1.0 / 12.0) - 1.0
    return np.array(
        [
            target_capital,
            mortgage_principal,
            mortgage_monthly_rate,
            hg.monthly_property_taxes_condo,
            hg.monthly_market_rent,
            hg.fhsa_annual_limit,
            hg.fhsa_max_balance,
            hg.hbp_max_withdrawal,
            float(hg.hbp_repayment_years),
            float(cfg.HOUSE_COUNT),
        ],
        dtype=np.float32,
    )


def lifecycle_constants(config) -> Dict[str, object]:
    """The base constants dict embedded in the payload (MODEL.constants)."""
    lc = config["lifecycle"]
    co = config["career"]
    hg = config["housing"]
    tax = config["tax"]
    cma = config["cma"]
    tables = build_model_tables(config)

    return {
        "currentAge": lc.current_age,
        "careerStartAge": lc.career_start_age,
        "retirementAge": lc.retirement_age,
        "pensionStartAge": lc.pension_start_age,
        "deathAge": lc.death_age,
        "accumMonths": tables["accum_months"],
        "bridgeMonths": tables["bridge_months"],
        "retireMonths": tables["retire_months"],
        "totalMonths": tables["accum_months"] + tables["retire_months"],
        "careerYears": tables["career_years"],
        "funds": list(cfg.FUNDS) + list(cfg.GLIDEPATH_OPTIONS),
        "annualDistributionYield": float(tax.annual_distribution_yield),
        "taxOnDistributions": float(tax.tax_on_distributions),
        "capitalGainsInclusion": float(tax.capital_gains_inclusion_rate),
        "capitalGainsTaxRate": float(tax.capital_gains_tax_rate),
        "hisaMonthly": float((1.0 + cma.hisa_annual_real_return) ** (1.0 / 12.0) - 1.0),
        "cashWedgeYears": float(cma.cash_wedge_years),
        "meltdownMonthly": float(tax.meltdown_bracket_annual / 12.0),
        "oasThresholdMonthly": float(tax.oas_clawback_threshold / 12.0),
        "realBorrowRateAnnual": float(cma.real_borrow_rate_annual),
        "extraMer15": float(cma.extra_mer_15),
        "extraMer20": float(cma.extra_mer_20),
        "layoffAnnualProbability": float(co.layoff_annual_probability),
        "qppAnnual": float(tables["pension"]["qpp"]),
        "oasAnnual7074": float(tables["pension"]["oas7074"]),
        "oasAnnual75Plus": float(tables["pension"]["oas75"]),
        "mortalityReductionFactor": float(lc.mortality_reduction_factor),
        "discountRateAnnual": float(lc.discount_rate_annual),
        "m75Start": int((75 - lc.retirement_age) * 12),
        "postWedgeMonth": int((lc.pension_start_age - lc.retirement_age) * 12),
        "bisectionSteps": config["simulation"].bisection_steps,
        "employerMatchRate": float(co.employer_match_rate),
        "employerMatchPercent": float(co.employer_match_percent),
        "houses": list(cfg.HOUSE_OPTIONS),
        "houseCount": cfg.HOUSE_COUNT,
        "targetHouseCapital": float(hg.property_value * hg.down_payment_fraction + hg.closing_costs),
        "mortgagePrincipal": float(hg.property_value * (1.0 - hg.down_payment_fraction)),
        "mortgageMonthlyRate": float((1.0 + hg.real_mortgage_rate_annual) ** (1.0 / 12.0) - 1.0),
        "monthlyPropertyTaxesCondo": float(hg.monthly_property_taxes_condo),
        "monthlyMarketRent": float(hg.monthly_market_rent),
        "fhsaAnnualLimit": float(hg.fhsa_annual_limit),
        "fhsaMaxBalance": float(hg.fhsa_max_balance),
        "hbpMaxWithdrawal": float(hg.hbp_max_withdrawal),
        "hbpRepaymentYears": hg.hbp_repayment_years,
    }


def model_defaults(config) -> Dict[str, object]:
    """The complete editable input set embedded in the payload (MODEL.inputs)."""
    lc = config["lifecycle"]
    co = config["career"]
    hg = config["housing"]
    tax = config["tax"]
    cma = config["cma"]
    cal = config["calibration"]
    sm = config["smile"]

    return {
        "currentAge": lc.current_age,
        "careerStartAge": lc.career_start_age,
        "retirementAge": lc.retirement_age,
        "pensionStartAge": lc.pension_start_age,
        "deathAge": lc.death_age,
        "startingSalary": co.starting_salary,
        "promotionPhases": [
            {"start": start, "end": end, "growth": rate}
            for start, end, rate in co.promotion_phases
        ],
        "savingsStartAnnual": co.retirement_savings_start_annual,
        "retirementSavingsStartAnnual": co.retirement_savings_start_annual,
        "retirementSavingsEscalationRate": co.retirement_savings_escalation_rate,
        "houseSavingsStartAnnual": co.house_savings_start_annual,
        "houseSavingsEscalationRate": co.house_savings_escalation_rate,
        "savingsMaxFraction": co.savings_max_fraction,
        "houseSavingsMaxFraction": co.house_savings_max_fraction,
        "propertyValue": hg.property_value,
        "downPaymentFraction": hg.down_payment_fraction,
        "closingCosts": hg.closing_costs,
        "realMortgageRateAnnual": hg.real_mortgage_rate_annual,
        "monthlyPropertyTaxesCondo": hg.monthly_property_taxes_condo,
        "monthlyMarketRent": hg.monthly_market_rent,
        "fhsaAnnualLimit": hg.fhsa_annual_limit,
        "fhsaMaxBalance": hg.fhsa_max_balance,
        "hbpMaxWithdrawal": hg.hbp_max_withdrawal,
        "hbpRepaymentYears": hg.hbp_repayment_years,
        "tfsaAnnualLimit": co.tfsa_annual_limit,
        "rrspContributionRate": co.rrsp_contribution_rate,
        "rrspMaxContribution": co.rrsp_max_contribution,
        "otherTfsa": co.other_tfsa,
        "layoffAnnualProbability": co.layoff_annual_probability,
        "post50HealthcareAnnual": lc.post50_healthcare_annual,
        "healthcareEndAge": lc.healthcare_end_age,
        "annualDistributionYield": tax.annual_distribution_yield,
        "taxOnDistributions": tax.tax_on_distributions,
        "capitalGainsInclusionRate": tax.capital_gains_inclusion_rate,
        "capitalGainsTaxRate": tax.capital_gains_tax_rate,
        "realBorrowRateAnnual": cma.real_borrow_rate_annual,
        "extraMer15": cma.extra_mer_15,
        "extraMer20": cma.extra_mer_20,
        "hisaAnnualRealReturn": cma.hisa_annual_real_return,
        "cashWedgeYears": cma.cash_wedge_years,
        "meltdownBracketAnnual": tax.meltdown_bracket_annual,
        "oasClawbackThreshold": tax.oas_clawback_threshold,
        "oasClawbackRate": tax.oas_clawback_rate,
        "maxQppAge65": tax.max_qpp_age_65,
        "maxOasAge65": tax.max_oas_age_65,
        "qppDeferralAnnual": tax.qpp_deferral_annual,
        "qppDeferralCap": tax.qpp_deferral_cap,
        "oasDeferralAnnual": tax.oas_deferral_annual,
        "oasDeferralCap": tax.oas_deferral_cap,
        "oas75Increase": tax.oas75_increase,
        "taxThresholdsAnnual": list(tax.tax_thresholds_annual),
        "taxRates": list(tax.tax_rates),
        "qppBasicAnnual": tax.qpp_basic_annual,
        "qppMaximumAnnual": tax.qpp_maximum_annual,
        "qppMaximumMSGA": tax.qpp_maximum_msga,
        "qppRate": tax.qpp_rate,
        "qppRateTier2": tax.qpp_rate_tier2,
        "eiMaximumAnnual": tax.ei_maximum_annual,
        "eiRate": tax.ei_rate,
        "rqapMaximumAnnual": tax.rqap_maximum_annual,
        "rqapRate": tax.rqap_rate,
        "cmas": dict(cma.cmas),
        "useForwardLookingCmas": cma.use_forward_looking_cmas,
        "skewDegreesFreedom": cal.skew_degrees_freedom,
        "deltaCap": cal.delta_cap,
        "deltaTolerance": cal.delta_tolerance,
        "mortalityReductionFactor": lc.mortality_reduction_factor,
        "discountRateAnnual": lc.discount_rate_annual,
        "smileSchedule": [
            {"start": start, "end": end, "change": change}
            for start, end, change in sm.smile_schedule
        ],
        "rrifFactors": {str(age): tax_rrif_factor(tax, age) for age in range(71, 95)},
        "employerMatchRate": co.employer_match_rate,
        "employerMatchPercent": co.employer_match_percent,
    }


# ---------------------------------------------------------------------------
# Allocation phase schedules & metadata
# ---------------------------------------------------------------------------
def _rounded_fraction_months(months, numerator, denominator):
    return (months * numerator + denominator // 2) // denominator


def _phase_funds(option, months, fund_indices):
    base_option = option.replace("+CASH", "")
    if base_option in cfg.GLIDEPATH_OPTIONS and option != base_option:
        raise ValueError(f"{base_option} cannot be combined with cash")
    if base_option not in cfg.GLIDEPATH_OPTIONS:
        return np.full(months, fund_indices[base_option], dtype=np.int8)
    quarter = _rounded_fraction_months(months, 1, 4)
    half = _rounded_fraction_months(months, 1, 2)
    if base_option == "DECLINING":
        counts = (half, quarter, months - half - quarter)
        funds = (fund_indices[cfg.FUNDS[0]], fund_indices[cfg.FUNDS[3]], fund_indices[cfg.FUNDS[4]])
    else:
        counts = (quarter, quarter, months - 2 * quarter)
        funds = (fund_indices[cfg.FUNDS[4]], fund_indices[cfg.FUNDS[3]], fund_indices[cfg.FUNDS[0]])
    return np.repeat(funds, counts).astype(np.int8)


def allocation_phase_code(option):
    base_option = option.replace("+CASH", "")
    if base_option in cfg.GLIDEPATH_OPTIONS and option != base_option:
        raise ValueError(f"{base_option} cannot be combined with cash")
    if base_option == "DECLINING":
        return len(cfg.FUNDS)
    if base_option == "RISING":
        return len(cfg.FUNDS) + 1
    return cfg.FUNDS.index(base_option)


def build_allocation_defs():
    """Return {name: (accum_list, bridge_list, post_list)} for all 5,040 strategies."""
    definitions = {}
    accum_options = list(cfg.FUNDS) + list(cfg.GLIDEPATH_OPTIONS)
    for house in cfg.HOUSE_OPTIONS:
        for accum in accum_options:
            for bridge in cfg.PHASE_OPTIONS:
                for post in cfg.PHASE_OPTIONS:
                    name = f"{house}_{accum}_{bridge}_{post}"
                    definitions[name] = ([accum], [bridge], [post])
    return definitions


def build_allocation_map(config):
    """Return ``(mapping, names)`` for every strategy.

    ``mapping`` has shape (5040, total_months) of int8 fund codes per month;
    ``names`` is the sorted strategy name list. Deterministic and shared by the
    accelerator metadata builder and the HTML payload.
    """
    lc = config["lifecycle"]
    definitions = build_allocation_defs()
    names = sorted(definitions.keys())
    accum_months = (lc.retirement_age - lc.current_age) * 12
    bridge_months = (lc.pension_start_age - lc.retirement_age) * 12
    retire_months = (lc.death_age - lc.retirement_age) * 12
    total_months = accum_months + retire_months
    pension_month = (lc.pension_start_age - lc.current_age) * 12
    fund_indices = {fund: index for index, fund in enumerate(cfg.FUNDS)}

    mapping = np.zeros((len(names), total_months), dtype=np.int8)
    for index, name in enumerate(names):
        accum_list, bridge_list, post_list = definitions[name]
        mapping[index, :accum_months] = _phase_funds(accum_list[0], accum_months, fund_indices)
        mapping[index, accum_months:pension_month] = _phase_funds(bridge_list[0], bridge_months, fund_indices)
        mapping[index, pension_month:] = _phase_funds(post_list[0], retire_months - bridge_months, fund_indices)
    return mapping, names


def _glidepath_boundaries(code, months):
    if code == len(cfg.FUNDS):  # DECLINING
        half = (months + 1) // 2
        return [half, half + (months + 2) // 4]
    if code == len(cfg.FUNDS) + 1:  # RISING
        quarter = (months + 2) // 4
        return [quarter, quarter * 2]
    return [0, 0]


def allocation_metadata(config, allocation_indices=None):
    """Build the 12-u32-per-strategy metadata buffer and the names list.

    Matches the browser's ``selectedAllocationBuffer`` layout exactly:
    ``[accumCode, bridgeCode, postCode, flags, accumGlide.xy, bridgeGlide.xy,
    postGlide.xy, 0, 0]``. This full form is what engine.py uploads to the
    GPU.
    """
    lc = config["lifecycle"]
    _, names = build_allocation_map(config)
    if allocation_indices is None:
        allocation_indices = list(range(len(names)))
    accum_months = (lc.retirement_age - lc.current_age) * 12
    bridge_months = (lc.pension_start_age - lc.retirement_age) * 12
    retire_months = (lc.death_age - lc.retirement_age) * 12

    metadata = np.zeros((len(allocation_indices), 12), dtype=np.uint32)
    out_names = []
    for out_index, allocation_index in enumerate(allocation_indices):
        name = names[allocation_index]
        parts = name.split("_")
        house = cfg.HOUSE_OPTIONS.index(parts[0] + "_" + parts[1])
        flags = (1 if "+CASH" in parts[3] else 0) | (2 if "+CASH" in parts[4] else 0) | (house << 2)
        codes = [
            allocation_phase_code(parts[2]),
            allocation_phase_code(parts[3]),
            allocation_phase_code(parts[4]),
            flags,
        ]
        metadata[out_index, 0:4] = codes
        metadata[out_index, 4:6] = _glidepath_boundaries(codes[0], accum_months)
        metadata[out_index, 6:8] = _glidepath_boundaries(codes[1], bridge_months)
        metadata[out_index, 8:10] = _glidepath_boundaries(codes[2], retire_months - bridge_months)
        out_names.append(name)
    return metadata, out_names


def allocation_codes(config, allocation_indices=None):
    """Compact 4-u32-per-strategy codes: [accumCode, bridgeCode, postCode, flags].

    This is the form embedded in the HTML payload; the browser rebuilds the
    full 12-u32 GPU rows (adding the glidepath boundaries) at run time.
    """
    _, names = build_allocation_map(config)
    if allocation_indices is None:
        allocation_indices = list(range(len(names)))
    codes = np.zeros((len(allocation_indices), 4), dtype=np.uint32)
    for out_index, allocation_index in enumerate(allocation_indices):
        name = names[allocation_index]
        parts = name.split("_")
        house = cfg.HOUSE_OPTIONS.index(parts[0] + "_" + parts[1])
        flags = (1 if "+CASH" in parts[3] else 0) | (2 if "+CASH" in parts[4] else 0) | (house << 2)
        codes[out_index] = [
            allocation_phase_code(parts[2]),
            allocation_phase_code(parts[3]),
            allocation_phase_code(parts[4]),
            flags,
        ]
    return codes


# ---------------------------------------------------------------------------
# Packed model buffer + full HTML payload
# ---------------------------------------------------------------------------
def build_model_buffer(config, price_path) -> np.ndarray:
    """The complete packed f32 model buffer uploaded onto the GPU.

    Layout (float32 words):
        [0, career_years*6)      career rows
        [*, *+retire*4)          month0 rows
        [*, *+retire*4)          month1 rows
        [*, *+54)                tax values
        [*, *+18)                skew-t constants (xi, omega, delta, Cholesky)
        [*, *+10)                house constants
    """
    tables = build_model_tables(config)
    returns_df = load_returns(price_path)
    model = calibrate_skew_t(
        returns_df[list(cfg.BASE_UNDERLYING)].values.astype(np.float64),
        config["calibration"],
        config["cma"],
    )
    return_model = np.concatenate(
        [
            model["xi"].astype(np.float64),
            model["omega"].astype(np.float64),
            model["delta"].astype(np.float64),
            model["cholesky"].astype(np.float64).reshape(-1),
        ]
    ).astype(np.float32)
    return np.concatenate(
        [
            tables["career"].reshape(-1),
            tables["month0"].reshape(-1),
            tables["month1"].reshape(-1),
            tables["tax_values"].reshape(-1),
            return_model,
            house_constants(config),
        ]
    ).astype(np.float32)


# Mortality table used for the intertemporal spending weights (CPM2014-like).
MORTALITY_ANNUAL_PROBABILITY = [
    1.000, 0.995, 0.990, 0.984, 0.978, 0.971, 0.963, 0.954, 0.944, 0.933,
    0.920, 0.906, 0.889, 0.870, 0.848, 0.823, 0.795, 0.763, 0.727, 0.687,
    0.643, 0.594, 0.541, 0.485, 0.426, 0.366, 0.307, 0.250, 0.198, 0.152,
    0.113, 0.081, 0.056, 0.038, 0.025, 0.016, 0.010, 0.006, 0.003, 0.001,
]


def fund_volatilities(price_path):
    returns_df = load_returns(price_path)
    vols = {
        fund: float(np.std(returns_df[fund].values, ddof=1) * np.sqrt(12) * 100)
        for fund in cfg.BASE_UNDERLYING
    }
    return {
        "VEQT": round(vols["VEQT"], 1),
        "VEQT1.5": round(vols["VEQT"] * 1.5, 1),
        "VEQT2": round(vols["VEQT"] * 2.0, 1),
        "VGRO": round(vols["VGRO"], 1),
        "VBAL": round(vols["VBAL"], 1),
    }


def build_payload(price_path, config=None) -> Dict[str, object]:
    """Serialize the complete data needed by the standalone HTML.

    The result is embedded as ``__MODEL_JSON__`` in every built page.
    """
    if config is None:
        config = cfg.instance_config()
    returns_df = load_returns(price_path)
    returns = returns_df[list(cfg.BASE_UNDERLYING)].values.astype(np.float64)
    model = calibrate_skew_t(returns, config["calibration"], config["cma"])
    tables = build_model_tables(config)
    # The payload embeds the compact 4-u32 codes; the browser expands them
    # into the full 12-u32 GPU rows at run time (selectedAllocationBuffer).
    codes = allocation_codes(config)
    names = sorted(build_allocation_defs().keys())
    constants = lifecycle_constants(config)

    defaults = {
        "simulations": config["simulation"].simulations,
        "batchSize": config["simulation"].batch_size,
        "columnsPerWorkgroup": config["simulation"].columns_per_workgroup,
        "gamma": config["simulation"].gamma,
        "floorPercentile": config["simulation"].floor_percentile,
        "targetSpending": config["simulation"].target_spending_monthly,
        "allocationCount": len(names),
    }

    dates = returns_df["Date"]
    return {
        "defaultSeed": 42,
        "defaults": defaults,
        "inputs": model_defaults(config),
        "historicalReturns": returns.reshape(-1).tolist(),
        "historicalReturnCount": int(len(returns_df)),
        "mortalityAnnualProbability": list(MORTALITY_ANNUAL_PROBABILITY),
        "returnModel": {
            "kind": "calibrated-skew-t",
            "xi": model["xi"].round(10).tolist(),
            "omega": model["omega"].round(10).tolist(),
            "correlation": np.asarray(model["correlation"], dtype=np.float64).round(10).reshape(-1).tolist(),
            "delta": model["delta"].round(10).tolist(),
            "cholesky": np.asarray(model["cholesky"], dtype=np.float64).round(10).reshape(-1).tolist(),
            "nu": int(model["nu"]),
            "observations": int(len(returns_df)),
            "dateStart": dates.iloc[0].strftime("%Y-%m-%d"),
            "dateEnd": dates.iloc[-1].strftime("%Y-%m-%d"),
        },
        "career": tables["career"].reshape(-1).tolist(),
        "month0": tables["month0"].reshape(-1).tolist(),
        "month1": tables["month1"].reshape(-1).tolist(),
        "taxValues": tables["tax_values"].reshape(-1).tolist(),
        "houseConstants": house_constants(config).reshape(-1).tolist(),
        "smile": tables["smile"].reshape(-1).tolist(),
        "cpmWeights": tables["cpm_weights"].reshape(-1).tolist(),
        "allocations": {
            "names": names,
            "metadata": codes.reshape(-1).tolist(),
            "count": len(names),
            "houses": list(cfg.HOUSE_OPTIONS),
        },
        "constants": constants,
        "fundVolatility": fund_volatilities(price_path),
    }