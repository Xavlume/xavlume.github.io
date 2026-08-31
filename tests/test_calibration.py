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


class TestSobolRqmc(unittest.TestCase):
    """The committed Joe-Kuo Sobol direction-number asset (RQMC sampler)."""

    def test_direction_number_table_shape_and_properties(self):
        config = cfg.instance_config()
        dims = cal.sobol_dimensions(config)
        lc = config["lifecycle"]
        total_months = (lc.retirement_age - lc.current_age) * 12 + (
            lc.death_age - lc.retirement_age
        ) * 12
        career_years = lc.retirement_age - lc.career_start_age
        self.assertEqual(dims, 10 * total_months + career_years)
        v = cal.sobol_direction_numbers(config)
        self.assertEqual(v.shape, (dims, cal.SOBOL_BITS))
        self.assertEqual(v.dtype, np.uint32)
        # Every bit level must have at least one non-zero direction number
        # across the dimensions (a degenerate level would zero the stream).
        self.assertTrue(np.all(v.any(axis=0)), "every direction-number column must be non-trivial")

    def test_first_direction_number_is_msb(self):
        # Point index 1 = v[:, 0]: every coordinate's first direction number
        # must be exactly 2^31 (the standard Sobol marginals).
        config = cfg.instance_config()
        v = cal.sobol_direction_numbers(config)
        self.assertTrue(np.all(v[:, 0] == np.uint32(0x80000000)))


class TestChi2Lut(unittest.TestCase):
    """The direct-chi2 quantile LUT appended to the RQMC buffer."""

    def test_lut_monotone_and_bounded(self):
        config = cfg.instance_config()
        lut = cal.chi2_lut(config)
        self.assertEqual(lut.shape, (cal.CHI2_LUT_PTS,))
        self.assertEqual(lut.dtype, np.float32)
        self.assertTrue(np.all(np.diff(lut.astype(np.float64)) > 0.0), "LUT must be strictly increasing")
        self.assertGreater(lut[0], 0.0)   # lower tail resolved (not collapsed)
        self.assertLess(lut[0], 0.02)     # floor ~ chi2(5).ppf(1e-7)
        self.assertGreater(lut[-1], 20.0)

    def test_lut_matches_scipy_at_grid(self):
        from scipy.stats import chi2 as _chi2
        config = cfg.instance_config()
        lut = cal.chi2_lut(config)
        lo = np.log(cal.CHI2_LUT_U_MIN)
        span = np.log(cal.CHI2_LUT_U_MAX) - lo
        pts = cal.CHI2_LUT_PTS
        grid = np.exp(lo + span * (np.arange(pts, dtype=np.float64) + 0.5) / pts)
        ref = _chi2.ppf(grid, float(config["calibration"].skew_degrees_freedom)).astype(np.float32)
        self.assertLess(float(np.abs(lut - ref).max()), 1e-4)

    def test_index_clamps_out_of_range(self):
        # u below U_MIN and above U_MAX must saturate at the edge bins
        # (frac clamped to [0, 1]), never extrapolate W negative.
        i, frac = cal.chi2_lut_index(np.array([1e-12, 0.5, 1.0 - 1e-12]))
        self.assertEqual(int(i[0]), 0)
        self.assertEqual(frac[0], 0.0)
        self.assertEqual(int(i[2]), cal.CHI2_LUT_PTS - 1)
        self.assertEqual(frac[2], 1.0)
        self.assertGreaterEqual(frac[1], 0.0)
        self.assertLessEqual(frac[1], 1.0)

    def test_rqmc_buffer_appends_lut(self):
        config = cfg.instance_config()
        buf = cal.sobol_rqmc_buffer(config)
        dims = cal.sobol_dimensions(config)
        off = dims * cal.SOBOL_BITS
        self.assertEqual(buf.shape[0], off + cal.CHI2_LUT_PTS)
        # LUT bits round-trip to the same f32 values
        self.assertTrue(
            np.array_equal(buf[off:].view(np.float32), cal.chi2_lut(config))
        )

    def test_rqmc_buffer_stores_top_bits(self):
        # The full storage buffer holds each direction number shifted right
        # by (32 - SOBOL_BITS); the WGSL load-shift reconstructs the exact
        # word, so for k <= SOBOL_BITS the truncated form is lossless.
        config = cfg.instance_config()
        v = cal.sobol_direction_numbers(config)
        stored = (v >> np.uint32(32 - cal.SOBOL_BITS)).astype(np.uint32)
        self.assertTrue(np.array_equal(
            (stored.astype(np.uint64) << np.uint64(32 - cal.SOBOL_BITS)).astype(np.uint32),
            v,
        ), "top-bits storage must reconstruct the full direction numbers")


class TestSobolCompactTable(unittest.TestCase):
    """The 14-bit browser table (?sampler=rqmc)."""

    def test_truncation_is_lossless_for_browser_bits(self):
        config = cfg.instance_config()
        v = cal.sobol_direction_numbers(config)
        bits = cal.SOBOL_BROWSER_BITS
        top = (v[:, :bits] >> np.uint32(32 - bits)).astype(np.uint32)
        back = (top.astype(np.uint64) << np.uint64(32 - bits)).astype(np.uint32)
        self.assertTrue(np.array_equal(back, v[:, :bits]),
                        "top-14 truncation must be exact for k <= 14")

    def test_pack_unpack_roundtrip(self):
        config = cfg.instance_config()
        bits = cal.SOBOL_BROWSER_BITS
        c = cal.sobol_compact_topbits(config, bits)
        packed = cal.sobol_pack_bits(c, bits)
        unpacked = cal.sobol_unpack_bits(packed, bits)
        self.assertTrue(np.array_equal(unpacked, c.reshape(-1)))

    def test_compact_net_matches_full_table_for_small_indices(self):
        # For simulation indices < 2^14 the compact table's reconstructed
        # direction numbers are identical to the full table's, so the RQMC
        # stream must be byte-identical.
        config = cfg.instance_config()
        v = cal.sobol_direction_numbers(config)
        bits = cal.SOBOL_BROWSER_BITS
        compact = cal.sobol_compact_topbits(config, bits)
        full_shape = np.zeros((v.shape[0], 20), dtype=np.uint32)
        full_shape[:, :bits] = (compact.astype(np.uint64) << np.uint64(32 - bits)).astype(np.uint32)
        rng = np.random.default_rng(7)
        idx = rng.integers(0, 1 << bits, size=200).astype(np.uint32)
        coords = rng.integers(0, v.shape[0], size=200).astype(np.uint32)
        u_full = cal.rqmc_uniforms(idx, coords, 42, v)
        u_compact = cal.rqmc_uniforms(idx, coords, 42, full_shape)
        self.assertTrue(np.array_equal(u_full, u_compact))

    def test_table_b64_roundtrip(self):
        config = cfg.instance_config()
        import base64
        b64 = cal.sobol_table_b64(config)
        self.assertGreater(len(b64), 1000)
        c = cal.sobol_compact_topbits(config)
        packed = cal.sobol_pack_bits(c)
        self.assertEqual(base64.b64decode(b64), packed)


if __name__ == "__main__":
    unittest.main(verbosity=2)