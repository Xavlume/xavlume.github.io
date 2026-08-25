"""Single source of truth for every parameter in the engine.

This module holds the strongly-typed configuration for a lifetime asset
allocation and retirement simulation of a Quebec (Montreal) resident. Every
other module in the project imports its defaults from here, so there is exactly
one copy of each fiscal rule, career milestone, CMA and real-estate number
instead of the scattered hardcoded literals of the old engine.

The module also provides the human-friendly formatting/parsing helpers that the
web UI uses to present raw engine decimals as ``20.0%``, ``$88,000``,
``4.30% / yr`` and so on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Strongly-typed configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LifecycleConfig:
    """Timeline and mortality assumptions."""

    current_age: int = 21
    career_start_age: int = 25
    retirement_age: int = 55
    pension_start_age: int = 65        # QPP/OAS start; QPP may defer to 72
    death_age: int = 95
    healthcare_end_age: int = 65       # post-50 healthcare is paid until here
    post50_healthcare_annual: float = 3_600.0
    mortality_reduction_factor: float = 0.50
    discount_rate_annual: float = 0.01


@dataclass(frozen=True)
class CareerConfig:
    """Salary trajectory, savings streams and workplace match."""

    starting_salary: float = 88_000.0
    # (start_age, end_age, annual_growth) promotion tiers.
    promotion_phases: List[Tuple[int, int, float]] = field(
        default_factory=lambda: [
            (25, 32, 0.050),
            (32, 42, 0.020),
            (42, 52, 0.010),
            (52, 60, 0.005),
        ]
    )
    retirement_savings_start_annual: float = 2_400.0
    retirement_savings_escalation_rate: float = 0.01
    savings_max_fraction: float = 0.20
    house_savings_start_annual: float = 1_200.0
    house_savings_escalation_rate: float = 0.02
    house_savings_max_fraction: float = 0.10
    tfsa_annual_limit: float = 7_000.0
    rrsp_contribution_rate: float = 0.18
    rrsp_max_contribution: float = 31_560.0
    other_tfsa: float = 10_000.0
    employer_match_rate: float = 0.04      # 4% of salary matched
    employer_match_percent: float = 1.00   # 100% employer match
    layoff_annual_probability: float = 0.05


@dataclass(frozen=True)
class HousingConfig:
    """Buy-versus-rent property, mortgage and registered-home-program numbers."""

    property_value: float = 500_000.0
    down_payment_fraction: float = 0.20
    closing_costs: float = 10_000.0
    real_mortgage_rate_annual: float = 0.02
    monthly_property_taxes_condo: float = 650.0
    monthly_market_rent: float = 2_500.0
    fhsa_annual_limit: float = 8_000.0
    fhsa_max_balance: float = 40_000.0
    hbp_max_withdrawal: float = 60_000.0
    hbp_repayment_years: int = 15


@dataclass(frozen=True)
class TaxFiscalConfig:
    """Quebec + federal income tax, payroll, QPP/OAS and capital-gains rules."""

    tax_thresholds_annual: List[float] = field(
        default_factory=lambda: [16_000.0, 52_000.0, 104_000.0, 175_000.0]
    )
    tax_rates: List[float] = field(
        default_factory=lambda: [0.00, 0.27, 0.37, 0.47, 0.53]
    )
    qpp_basic_annual: float = 3_500.0
    qpp_maximum_annual: float = 71_300.0     # Tier 1 MGA / YMPE
    qpp_maximum_msga: float = 81_200.0       # Tier 2 MSGA / YAMPE
    qpp_rate: float = 0.0640                 # Tier 1
    qpp_rate_tier2: float = 0.0400           # Tier 2
    ei_maximum_annual: float = 65_700.0
    ei_rate: float = 0.0131
    rqap_maximum_annual: float = 98_000.0
    rqap_rate: float = 0.00494
    oas_clawback_threshold: float = 95_323.0
    oas_clawback_rate: float = 0.15
    meltdown_bracket_annual: float = 55_000.0      # RRSP meltdown ceiling
    capital_gains_inclusion_rate: float = 0.50
    capital_gains_tax_rate: float = 0.30
    max_qpp_age_65: float = 25_500.0               # enhanced QPP at 65
    max_oas_age_65: float = 8_907.72
    qpp_deferral_annual: float = 0.084             # +8.4%/yr to 72
    qpp_deferral_cap: float = 1.588                # age-72 cap
    oas_deferral_annual: float = 0.072             # +7.2%/yr to 70
    oas_deferral_cap: float = 1.36
    oas75_increase: float = 1.10
    annual_distribution_yield: float = 0.018
    tax_on_distributions: float = 0.25


@dataclass(frozen=True)
class CMAConfig:
    """Expected returns, borrowing rates, fees and the cash buffer."""

    cmas: Dict[str, float] = field(
        default_factory=lambda: {"VEQT": 0.0430, "VGRO": 0.0371, "VBAL": 0.0312}
    )
    use_forward_looking_cmas: bool = True
    hisa_annual_real_return: float = -0.005
    cash_wedge_years: float = 3.0
    real_borrow_rate_annual: float = 0.0225
    extra_mer_15: float = 0.0020
    extra_mer_20: float = 0.0040


@dataclass(frozen=True)
class ModelCalibrationConfig:
    """Heavy-tailed multivariate skew-t return calibration controls."""

    skew_degrees_freedom: int = 5
    delta_cap: float = 0.85
    delta_tolerance: float = 0.99


@dataclass(frozen=True)
class SmileConfig:
    """Retirement spending 'smile' schedule: (start, end, annual_change)."""

    smile_schedule: List[Tuple[int, int, float]] = field(
        default_factory=lambda: [
            (60, 70, 0.000),
            (70, 77, -0.010),
            (77, 78, -0.015),
            (78, 95, +0.012),
        ]
    )


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime solver/tuning defaults exposed to the web UI."""

    simulations: int = 1_000
    batch_size: int = 1_024
    columns_per_workgroup: int = 128
    gamma: float = 3.0
    floor_percentile: int = 10
    target_spending_monthly: float = 3_500.0
    bisection_steps: int = 24


def default_config() -> Dict[str, object]:
    """Return the complete configuration as a flat dict keyed by category."""
    return {
        "lifecycle": LifecycleConfig(),
        "career": CareerConfig(),
        "housing": HousingConfig(),
        "tax": TaxFiscalConfig(),
        "cma": CMAConfig(),
        "calibration": ModelCalibrationConfig(),
        "smile": SmileConfig(),
        "simulation": SimulationConfig(),
    }


def instance_config(
    lifecycle: LifecycleConfig | None = None,
    career: CareerConfig | None = None,
    housing: HousingConfig | None = None,
    tax: TaxFiscalConfig | None = None,
    cma: CMAConfig | None = None,
    calibration: ModelCalibrationConfig | None = None,
    smile: SmileConfig | None = None,
    simulation: SimulationConfig | None = None,
) -> Dict[str, object]:
    """Compose an explicit configuration, filling unspecified categories with their defaults."""
    defaults = default_config()
    return {
        "lifecycle": lifecycle or defaults["lifecycle"],
        "career": career or defaults["career"],
        "housing": housing or defaults["housing"],
        "tax": tax or defaults["tax"],
        "cma": cma or defaults["cma"],
        "calibration": calibration or defaults["calibration"],
        "smile": smile or defaults["smile"],
        "simulation": simulation or defaults["simulation"],
    }


# ---------------------------------------------------------------------------
# Human-friendly formatting & parsing utilities
# ---------------------------------------------------------------------------
def frac_to_pct(fraction: float) -> str:
    """``0.20`` -> ``20.0%``."""
    return f"{fraction * 100:.1f}%"


def pct_to_frac(text: str) -> float:
    """``'20.0%'`` -> ``0.2`` (tolerates a bare number meaning percent)."""
    clean = str(text).strip().rstrip("%").replace(",", "").strip()
    return float(clean) / 100.0


def money(value: float) -> str:
    """``88000`` -> ``$88,000``."""
    return f"${value:,.0f}"


def money_parse(text: str) -> float:
    """``'$88,000'``/``'88,000'`` -> ``88000.0``."""
    clean = re.sub(r"[^0-9.\-]", "", str(text))
    return float(clean) if clean not in ("", "-") else 0.0


def years(value: float) -> str:
    return f"{value:g}"


def years_parse(text: str) -> float:
    return float(str(text).strip().rstrip("yrs").strip())


def rate_per_yr(rate: float) -> str:
    """A per-annum growth/return rate as ``+X.X% / yr``."""
    sign = "+" if rate >= 0 else ""
    return f"{sign}{rate * 100:.1f}% / yr"


def annual_rate_parse(text: str) -> float:
    """Parse a ``% / yr``/``%`` field back to the raw annual fraction."""
    clean = re.sub(r"[^0-9.\-]", "", str(text))
    return float(clean) / 100.0 if clean not in ("", "-") else 0.0


# ---------------------------------------------------------------------------
# References shared across the engine
# ---------------------------------------------------------------------------
# The five underlying return series. ``VEQT1.5``/``VEQT2`` are synthetic
# leveraged transforms of ``VEQT``; the three historical series are VEQT, VGRO,
# VBAL.
FUNDS = ("VEQT", "VEQT1.5", "VEQT2", "VGRO", "VBAL")
BASE_UNDERLYING = ("VEQT", "VGRO", "VBAL")
GLIDEPATH_OPTIONS = ("DECLINING", "RISING")
# Accumulation paths = the five funds plus the two glidepaths.
PATH_COUNT = len(FUNDS) + len(GLIDEPATH_OPTIONS)

HOUSE_OPTIONS = (
    "HOUSE_NONE", "HOUSE_CASH", "HOUSE_VBAL", "HOUSE_VGRO", "HOUSE_VEQT",
)
HOUSE_COUNT = len(HOUSE_OPTIONS)

# Bridge and post-pension phase options: every pure fund, its ``+CASH`` wedge
# variant, plus the two glidepaths.
PHASE_OPTIONS = (
    "VEQT", "VEQT+CASH", "VEQT1.5", "VEQT1.5+CASH", "VEQT2", "VEQT2+CASH",
    "VGRO", "VGRO+CASH", "VBAL", "VBAL+CASH", "DECLINING", "RISING",
)

# Composite Ulcer Index blend weights: bridge/post/accumulation.
UI_WEIGHTS = (0.60, 0.25, 0.15)
UI_WEIGHTS_NO_BRIDGE = (0.00, 0.65, 0.35)