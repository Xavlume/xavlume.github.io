"""Unit tests for the configuration and statistical calibration modules.

Covers:

* the human-friendly formatting/parsing utilities in :mod:`config`;
* the exact integer-nu ``b_nu`` constant vs ``scipy.special.gamma``;
* the multivariate skew-t calibration (finite moments, positive-definite
  correlation, consistent Cholesky, capped delta);
* the derived lifecycle/fiscal tables (career rows, month tables, tax tail,
  smile, intertemporal weights);
* the 5,040-strategy allocation metadata buffer;
* the serialized HTML payload structure.

Run with:  python tests/test_calibration.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import calibration as cal
import config as cfg

PRICE_PATH = ROOT / "downloaded_prices.csv"


class TestFormatting(unittest.TestCase):
    def test_percent_roundtrip(self):
        for fraction in (0.2, 0.04, 0.005, -0.005, 0.018, 1.0, 0.0):
            self.assertAlmostEqual(cfg.pct_to_frac(cfg.frac_to_pct(fraction)), fraction, places=6)

    def test_money_roundtrip(self):
        for value in (88_000.0, 2_400.0, 500_000.0, 650.0):
            self.assertEqual(cfg.money_parse(cfg.money(value)), value)

    def test_annual_rate_helpers(self):
        self.assertEqual(cfg.annual_rate_parse("4.30% / yr"), 0.043)
        self.assertEqual(cfg.annual_rate_parse("0.20%"), 0.002)
        self.assertIn("+1.0% / yr", cfg.rate_per_yr(0.01))


class TestBNu(unittest.TestCase):
    def test_b_nu_matches_scipy(self):
        from scipy.special import gamma

        for nu in (3, 4, 5, 6, 7, 8, 10, 12):
            expected = math.sqrt(nu / math.pi) * gamma((nu - 1) / 2) / gamma(nu / 2)
            self.assertAlmostEqual(cal.b_nu(nu), expected, places=12, msg=f"nu={nu}")

    def test_b_nu_small_df(self):
        self.assertEqual(cal.b_nu(1), 0.0)


class TestSkewTCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cfg.instance_config()
        cls.returns_df = cal.load_returns(PRICE_PATH)
        cls.returns = cls.returns_df[list(cfg.BASE_UNDERLYING)].values.astype(np.float64)
        cls.model = cal.calibrate_skew_t(cls.returns, cls.config["calibration"], cls.config["cma"])

    def test_model_shape_and_finiteness(self):
        for key in ("xi", "omega", "delta"):
            self.assertEqual(self.model[key].shape, (3,))
            self.assertTrue(np.isfinite(self.model[key]).all())
        self.assertEqual(self.model["cholesky"].shape, (3, 3))
        self.assertEqual(self.model["nu"], 5)

    def test_correlation_positive_definite(self):
        corr = self.model["correlation"]
        self.assertTrue(np.isfinite(corr).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(corr) > -1e-8))
        self.assertTrue(np.allclose(np.diag(corr), 1.0))

    def test_cholesky_reproduces_residual(self):
        residual = self.model["correlation"] - np.outer(self.model["delta"], self.model["delta"])
        self.assertTrue(
            np.allclose(self.model["cholesky"] @ self.model["cholesky"].T, residual, atol=1e-8)
        )

    def test_delta_within_cap(self):
        self.assertTrue(np.all(np.abs(self.model["delta"]) <= 0.85))

    def test_omega_positive(self):
        self.assertTrue(np.all(self.model["omega"] > 0))

    def test_xi_close_to_historical_mean(self):
        # With forward-looking CMAs the implied mean should sit near history.
        implied = self.model["xi"] + self.model["omega"] * self.model["delta"] * cal.b_nu(5)
        historical = self.returns.mean(axis=0)
        self.assertTrue(np.allclose(implied, historical, atol=5e-3))


class TestDerivedTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cfg.instance_config()
        cls.tables = cal.build_model_tables(cls.config)

    def test_career_table(self):
        lc = self.config["lifecycle"]
        self.assertEqual(self.tables["career"].size, 6 * self.tables["career_years"])
        self.assertTrue(np.isfinite(self.tables["career"]).all())
        # Cumulative RRSP room must be monotonically non-decreasing.
        rrsp = self.tables["career"][4::6]
        self.assertTrue(np.all(np.diff(rrsp) >= 0))

    def test_month_tables(self):
        self.assertEqual(self.tables["month0"].size, 4 * self.tables["retire_months"])
        self.assertEqual(self.tables["month1"].size, 4 * self.tables["retire_months"])
        # Pension appears after the pension start age.
        pension_gross = self.tables["month0"][2::4]
        pre = pension_gross[: self.tables["bridge_months"]]
        post = pension_gross[self.tables["bridge_months"]:]
        self.assertTrue(np.all(pre == 0.0))
        self.assertTrue(np.all(post > 0.0))

    def test_tax_tail_layout(self):
        tax = self.tables["tax_values"]
        self.assertEqual(tax.size, 54)
        thresholds = self.config["tax"].tax_thresholds_annual
        rates = self.config["tax"].tax_rates
        self.assertTrue(np.allclose(tax[44:48], np.array(thresholds) / 12.0, atol=1e-6))
        self.assertTrue(np.allclose(tax[49:54], rates, atol=1e-6))

    def test_smile_and_cpm_weights(self):
        smile = self.tables["smile"]
        self.assertTrue(np.all(smile > 0.0))
        cpm = self.tables["cpm_weights"]
        self.assertAlmostEqual(float(cpm.sum()), 1.0, places=5)
        self.assertTrue(np.all(cpm > 0.0))

    def test_model_buffer_layout(self):
        buf = cal.build_model_buffer(self.config, PRICE_PATH)
        lc = self.config["lifecycle"]
        career = self.tables["career_years"] * 6
        retire = self.tables["retire_months"]
        expected = career + 8 * retire + 54 + 18 + 11 + 1 + len(
            self.config["bequest"].estate_grid_fractions
        )
        self.assertEqual(buf.size, expected)
        self.assertTrue(np.isfinite(buf).all())

    def test_model_buffer_bequest_tail(self):
        buf = cal.build_model_buffer(self.config, PRICE_PATH)
        beq = self.config["bequest"]
        retire = self.tables["retire_months"]
        house_base = self.tables["career_years"] * 6 + 8 * retire + 54 + 18
        # House constant index 10 = the target property value (bequest home equity).
        self.assertEqual(float(buf[house_base + 10]), self.config["housing"].property_value)
        # Estate-grid tail: [grid count, fractions...].
        tail = buf[house_base + 11:]
        self.assertEqual(int(tail[0]), len(beq.estate_grid_fractions))
        self.assertTrue(np.allclose(tail[1:], beq.estate_grid_fractions, atol=1e-6))

    def test_bequest_payload(self):
        payload = cal.build_payload(PRICE_PATH, cfg.instance_config())
        # DFJ-aligned defaults: gamma 4, theta 0.5 (parity), k $200k.
        self.assertEqual(payload["defaults"]["gamma"], 4.0)
        self.assertEqual(payload["defaults"]["bequestIntensity"], 0.5)
        self.assertEqual(payload["defaults"]["bequestCurvature"], 200_000.0)
        self.assertEqual(
            payload["constants"]["estateGridFractions"],
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        self.assertEqual(payload["constants"]["minBequestCurvature"], 10_000.0)
        self.assertEqual(payload["constants"]["bequestParityReferenceEstate"], 500_000.0)
        self.assertEqual(payload["constants"]["bequestParityReferenceSpending"], 5_000.0)
        self.assertEqual(payload["constants"]["propertyValue"], 500_000.0)


class TestAllocationMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cfg.instance_config()
        cls.mapping, cls.names = cal.build_allocation_map(cls.config)
        cls.metadata, cls.out_names = cal.allocation_metadata(cls.config)

    def test_strategy_count(self):
        self.assertEqual(len(self.names), 5 * 7 * 12 * 12)
        self.assertEqual(len(self.names), 5040)

    def test_name_convention(self):
        for name in self.names[:100]:
            parts = name.split("_")
            self.assertEqual(len(parts), 5)
            self.assertTrue(parts[0] + "_" + parts[1] in cfg.HOUSE_OPTIONS)
            self.assertIn(parts[2], list(cfg.FUNDS) + list(cfg.GLIDEPATH_OPTIONS))
            self.assertIn(parts[3], cfg.PHASE_OPTIONS)
            self.assertIn(parts[4], cfg.PHASE_OPTIONS)

    def test_metadata_shape(self):
        self.assertEqual(self.metadata.shape, (5040, 12))
        self.assertEqual(self.out_names, self.names)

    def test_metadata_flags(self):
        # HOUSE_NONE renter has house bits zero; cash flags match +CASH suffix.
        index = self.names.index("HOUSE_NONE_VEQT_VEQT+CASH_VEQT")
        flags = self.metadata[index, 3]
        self.assertEqual(flags & 1, 1)          # cash bridge
        self.assertEqual(flags & 2, 0)          # no cash post
        self.assertEqual((flags >> 2) & 7, 0)   # house code 0 = renter
        # A DECLINING accumulation glidepath gets within-month boundaries.
        lc = self.config["lifecycle"]
        accum_months = (lc.retirement_age - lc.current_age) * 12
        index = self.names.index("HOUSE_NONE_DECLINING_VEQT+CASH_VEQT")
        glide = self.metadata[index, 4:6]
        self.assertTrue(glide[0] > 0 and glide[1] > glide[0] and glide[1] <= accum_months)

    def test_allocation_phase_codes(self):
        self.assertEqual(cal.allocation_phase_code("DECLINING"), 5)
        self.assertEqual(cal.allocation_phase_code("RISING"), 6)
        self.assertEqual(cal.allocation_phase_code("VEQT1.5"), 1)


class TestPayload(unittest.TestCase):
    def test_payload_structure(self):
        config = cfg.instance_config()
        payload = cal.build_payload(PRICE_PATH, config)
        self.assertEqual(payload["allocations"]["count"], 5040)
        self.assertEqual(len(payload["allocations"]["metadata"]), 5040 * 4)
        self.assertEqual(payload["returnModel"]["nu"], 5)
        self.assertEqual(payload["historicalReturnCount"], payload["returnModel"]["observations"])
        self.assertEqual(len(payload["historicalReturns"]), 3 * payload["historicalReturnCount"])
        self.assertEqual(len(payload["mortalityAnnualProbability"]), 40)
        self.assertEqual(payload["defaults"]["allocationCount"], 5040)
        self.assertEqual(len(payload["inputs"]["promotionPhases"]), 4)
        self.assertEqual(len(payload["inputs"]["smileSchedule"]), 4)
        self.assertEqual(len(payload["constants"]["funds"]), 7)
        # Dates parse as ISO strings.
        self.assertRegex(payload["returnModel"]["dateStart"], r"^\d{4}-\d{2}-\d{2}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)