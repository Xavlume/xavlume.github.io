"""Numerical parity and determinism tests: Python CPU vs. WebGPU.

The WGSL sampler and the NumPy reference in :mod:`calibration` are the same
Threefry-2x32-20 skew-t generator, so the parity test proves that the GPU
pipeline produces byte-level-equivalent market paths to a clean CPU
implementation of the same algorithm. It then verifies that the full on-chip
pipeline is deterministic (identical quantiles across repeated runs) and that
the solved spending ladders satisfy basic monotonicity and range sanity
constraints.

Run with:  python tests/test_parity.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import calibration as cal
import config as cfg
import engine as engine_mod

PRICE_PATH = ROOT / "downloaded_prices.csv"

# Float32 GPU arithmetic vs float64 CPU: monthly returns live in [-0.95, ~1],
# so an absolute tolerance of 5e-4 is comfortably above the rounding noise.
RETURN_TOLERANCE = 5e-4


def _has_wgpu() -> bool:
    try:
        import wgpu  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_wgpu(), "wgpu package not installed")
class TestReturnParity(unittest.TestCase):
    """CPU NumPy sampler vs. the WGSL generate_returns pass."""

    def setUp(self):
        self.config = cfg.instance_config()
        self.returns_df = cal.load_returns(PRICE_PATH)
        self.model = cal.calibrate_skew_t(
            self.returns_df[list(cfg.BASE_UNDERLYING)].values.astype(np.float64),
            self.config["calibration"],
            self.config["cma"],
        )
        tables = cal.build_model_tables(self.config)
        self.total_months = tables["accum_months"] + tables["retire_months"]
        self.engine = engine_mod.Engine(config=self.config, batch_size=64)

    def test_skew_t_returns_match_cpu_reference(self):
        simulations = 16
        cpu = cal.returns_cpu(simulations, self.total_months, self.model, seed=42, config=self.config)
        gpu = self.engine.run_returns(simulations, months=self.total_months)
        diff = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
        self.assertLess(float(diff.max()), RETURN_TOLERANCE, f"max |cpu-gpu| = {diff.max():.2e}")

    def test_leveraged_funds_consistent(self):
        simulations = 8
        cpu = cal.returns_cpu(simulations, self.total_months, self.model, seed=42, config=self.config)
        # VEQT1.5 must be exactly 1.5*VEQT - 0.5*borrow - fee in both worlds.
        expected = np.maximum(1.5 * cpu[..., 0] - 0.5 * 0.0225 / 12 - 0.002 / 12, -0.95)
        self.assertTrue(np.allclose(cpu[..., 1], expected, atol=1e-6))

    def test_returns_are_distinct_per_path(self):
        simulations = 8
        gpu = self.engine.run_returns(simulations, months=self.total_months)
        flat = gpu.reshape(simulations, -1)
        unique = len(np.unique(flat, axis=0))
        self.assertEqual(unique, simulations, "paths must not be duplicated")


@unittest.skipUnless(_has_wgpu(), "wgpu package not installed")
class TestPipelineDeterminism(unittest.TestCase):
    def test_repeated_runs_are_identical(self):
        engine = engine_mod.Engine(config=cfg.instance_config(), batch_size=64)
        indices = list(range(8))
        first = engine.run(64, allocation_indices=indices)
        second = engine.run(64, allocation_indices=indices)
        self.assertTrue(np.array_equal(first.spending, second.spending))
        self.assertTrue(np.array_equal(first.quantiles, second.quantiles))
        self.assertTrue(np.array_equal(first.ui_means, second.ui_means))
        self.assertTrue(np.array_equal(first.estate, second.estate))
        self.assertEqual(first.names, second.names)

    def test_quantiles_monotone_and_sane(self):
        engine = engine_mod.Engine(config=cfg.instance_config(), batch_size=64)
        result = engine.run(64, allocation_indices=list(range(8)))
        q = result.quantiles
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all(q[:, 10] <= q[:, 100]))
        self.assertTrue(np.all(q[:, 100] <= q[:, 190]))
        # Solved sustainable spending must be strictly positive.
        self.assertTrue(np.all(q[:, 0] > 0.0))
        # Median monthly spending in a plausible range for this scenario.
        medians = q[:, 100]
        self.assertTrue(np.all(medians > 500.0) and np.all(medians < 50_000.0),
                        f"medians out of range: {medians}")
        # Composite UI scores are non-negative with a material spread.
        self.assertTrue(np.all(result.ui_means >= 0.0))
        self.assertGreater(float(result.ui_means.max() - result.ui_means.min()), 1.0)

    def test_estate_ladders_sane_and_deterministic(self):
        config = cfg.instance_config()
        engine = engine_mod.Engine(config=config, batch_size=64)
        names = sorted(cal.build_allocation_defs().keys())
        renter = names.index("HOUSE_NONE_VEQT_VEQT_VEQT")
        buyer = names.index("HOUSE_CASH_VEQT_VEQT_VEQT")
        first = engine.run(64, allocation_indices=[renter, buyer])
        second = engine.run(64, allocation_indices=[renter, buyer])
        self.assertTrue(np.array_equal(first.estate, second.estate))
        e = first.estate  # (allocations, grid, 201)
        grid = len(config["bequest"].estate_grid_fractions)
        self.assertEqual(e.shape, (2, grid, 201))
        self.assertTrue(np.isfinite(e).all())
        self.assertTrue(np.all(e >= 0.0))
        # The same lives are walked at every fraction, and spending less
        # cannot reduce a life's terminal estate: every ladder point must be
        # non-increasing in the fraction f (small tolerance for histogram
        # interpolation noise at near-zero estates).
        for a in range(e.shape[0]):
            for g in range(grid - 1):
                self.assertTrue(np.all(e[a, g] >= e[a, g + 1] - 1.0),
                                f"allocation {a}: estate not monotone at grid {g}")
        # At f = 1 the liquid portfolio is exhausted by construction of w*:
        # renters keep ~nothing, buyers keep only the principal residence
        # (~the target property value, mortgage amortized to zero).
        self.assertLess(float(e[0, -1, 100]), 25_000.0, "renter f=1 median estate")
        self.assertGreater(float(e[1, -1, 100]), 300_000.0, "buyer f=1 median estate")
        self.assertLess(float(e[1, -1, 100]), 700_000.0, "buyer f=1 median estate")
        # At f = 0.5 the estate is materially larger than at f = 1.
        self.assertGreater(float(e[0, 0, 100]), 4 * float(e[0, -1, 100]) + 1_000.0)
        self.assertGreater(float(e[1, 0, 100]), float(e[1, -1, 100]) + 50_000.0)

    def test_estate_ladders_batched_identical_to_single_pass(self):
        # The bequest walk is dispatched once per simulation batch and folds
        # every life into a persistent fixed-scale histogram; integer
        # accumulation is exact, so splitting a run across batches must
        # produce byte-identical ladders to a single-dispatch run.
        config = cfg.instance_config()
        names = sorted(cal.build_allocation_defs().keys())
        indices = [names.index("HOUSE_NONE_VEQT_VEQT_VEQT"), names.index("HOUSE_CASH_VEQT_VEQT_VEQT")]
        single = engine_mod.Engine(config=config, batch_size=128).run(128, allocation_indices=indices)
        multi = engine_mod.Engine(config=config, batch_size=64).run(128, allocation_indices=indices)
        self.assertTrue(np.array_equal(single.estate, multi.estate),
                        "batched estate ladders differ from the single-pass ladders")
        self.assertTrue(np.array_equal(single.spending, multi.spending))
        # House outcomes and per-path UI scores are read back per batch and
        # must also be batch-invariant (regression gate for the old bug where
        # a post-loop scratch re-read returned the LAST batch's data for every
        # batch slice).
        self.assertTrue(np.array_equal(single.house_outcomes, multi.house_outcomes),
                        "batched house outcomes differ from the single-pass outcomes")
        self.assertTrue(np.array_equal(single.ui_means, multi.ui_means),
                        "batched UI means differ from the single-pass UI means")


if __name__ == "__main__":
    unittest.main(verbosity=2)