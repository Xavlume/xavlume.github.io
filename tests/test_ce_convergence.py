"""Convergence of the Certainty-Equivalent (CEQ) estimator with simulation count.

Focused study on one strategy:

    HOUSE_VBAL_VEQT_VEQT+CASH_VEQT   ("HOUSE_VBAL + VEQT_VEQT+CASH_VEQT")

The CEQ metric is the tool's default (gamma = 4.0, theta = 0, lambda = 0,
kappa smile-adjusted) — identical to ``tests/test_mcse.py``.

Design
------
Two questions, one experiment:

1. **Is the 100k-sim run itself stable?** Run M = 10 distinct seeds at
   N = 100,000 and report the Monte Carlo Standard Error of the CEQ estimator
   at that N (MCSE = sample std across the M seeds).

2. **How do the small runs behave?** Run the same M = 10 seeds at each of
   N = 100, 1,000, 10,000 and, per level, log:
   - the per-seed CEQ values,
   - MCSE(N) = std across the M seeds (pure sampling error at that N),
   - RMSE(N) = sqrt( mean_m (CE_N,m - CE_target)^2 ) — total error of the
     N-sim estimator against the refined 100k target, combining bias and
     variance,
   - bias = mean(CE_N) - CE_target and its z-score bias / MCSE(N),
   - the 1/sqrt(N) scaling check: MCSE(N) compared with
     MCSE(100k) * sqrt(100000 / N).

3. **Estimator comparison (same runs, no extra GPU cost).** Every run is
   scored with two CEQ estimators of the *same* asymptotic target:
   - ``ladder`` — the tool's metric: the 201-point GPU histogram ladder
     (``ce_for_quantiles``),
   - ``ecdf`` — the same 199-point CEQ formula on linearly interpolated
     order statistics of the N raw lives (``ce_for_raw_spending``); it
     removes the histogram ladder's lower-tail binning bias at low N.

4. **Sampler choice.** ``py -3.14 tests/test_ce_convergence.py --rqmc`` swaps
   the uniform source to the digital-shift scrambled Sobol (RQMC) stream
   (``engine.run(rqmc=True)``, coordinate map in ``calibration``); it keeps
   the exact-prefix property and converges to the same CEQ with lower
   sampling error at a given N. The deployed page defaults to RQMC and can
   opt back to Threefry with ``?sampler=threefry``.

Because the WGSL streams are keyed on (seed, global simulation index) only,
an N-sim run with a given seed draws exactly the first N lives of the
100k run with the same seed (``returns.wgsl``: counter = global_sim *
total_months + month; the RQMC Sobol base-2 sequence is extensible and its
per-coordinate shift depends only on the seed). The small-N estimates are
therefore nested prefixes of the reference sample, and RMSE is a pure
estimator-convergence measure.

Run (full experiment). Results print to the console only:

    py -3.14 tests/test_ce_convergence.py [--simulations 100000] [--seeds 10]
                                          [--lower 100,1000,10000]
                                          [--batch-size 1024] [--rqmc]

Fast unit smoke tests (no heavy run):

    py -3.14 tests/test_ce_convergence.py --smoke
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import calibration as cal
import config as cfg
import engine as engine_mod

PRICE_PATH = ROOT / "downloaded_prices.csv"

# The strategy under test.
STRATEGY = "HOUSE_VBAL_VEQT_VEQT+CASH_VEQT"

# Simulation counts for the deep small-N study (besides the 100k target).
LOW_COUNTS = (100, 1_000, 10_000)

SEED_BASE = 42          # the engine's canonical seed
SEED_STEP = 1000        # step between the M distinct seeds


def _has_wgpu() -> bool:
    try:
        import wgpu  # noqa: F401
        return True
    except ImportError:
        return False


def _strategy_index(config) -> int:
    names = sorted(cal.build_allocation_defs().keys())
    return names.index(STRATEGY)


def compute_kappa(gamma: float, tables) -> float:
    """Smile kappa; mirrors the browser's ``computeKappa``."""
    exponent = 1.0 - gamma
    smile = np.maximum(tables["smile"], 1e-6)
    weights = tables["cpm_weights"]
    if abs(exponent) < 1e-4:
        return float(np.exp(np.sum(weights * np.log(smile))))
    return float(np.sum(weights * smile ** exponent) ** (1.0 / exponent))


def ce_for_quantiles(quantiles: np.ndarray, gamma: float, tables) -> float:
    """Base CEQ from a 201-point ladder; mirrors the browser's ``ceForQuantiles``."""
    exponent = 1.0 - gamma
    interior = np.maximum(quantiles[1:200], 1e-6)
    if abs(exponent) < 1e-4:
        base = float(np.exp(np.mean(np.log(interior))))
    else:
        base = float(np.mean(interior ** exponent) ** (1.0 / exponent))
    return base * compute_kappa(gamma, tables)


def ce_for_raw_spending(spending_annual: np.ndarray, gamma: float, tables) -> float:
    """CEQ from the raw per-life spending values (ECDF quantile ladder).

    Identical metric definition and identical asymptotic target to the GPU
    201-point ladder (same 199 interior grid probabilities p = i / 200), but
    the quantiles are linearly interpolated order statistics of the N actual
    lives instead of the GPU histogram's log-bin interpolation. Both
    estimators converge to the same CEQ; at finite N the ECDF version removes
    the histogram ladder's small lower-tail bias at no GPU cost (see the
    estimator comparison table in the experiment output).
    """
    monthly = np.asarray(spending_annual, dtype=np.float64) / 12.0
    probs = np.arange(1, 200, dtype=np.float64) / 200.0
    ladder = np.empty(201, dtype=np.float64)
    ladder[0] = monthly.min()
    ladder[200] = monthly.max()
    ladder[1:200] = np.quantile(monthly, probs, method="linear")
    return ce_for_quantiles(ladder, gamma, tables)


def timed_run(engine, index: int, simulations: int, rqmc: bool = False,
              owen: bool = False, direct_w: bool = False) -> tuple:
    """Run the GPU pipeline once; return (CEQ-relevant result, wall seconds)."""
    start = time.perf_counter()
    result = engine.run(simulations, allocation_indices=[index], include_bequest=False,
                        rqmc=rqmc, owen=owen, direct_w=direct_w)
    return result, time.perf_counter() - start


def ceqs_for_seed(engine, index: int, simulations: int, gamma: float, tables,
                  rqmc: bool = False, owen: bool = False, direct_w: bool = False) -> tuple:
    """One seed, one strategy: run the pipeline once and score every estimator.

    Returns ``({"ladder": ..., "ecdf": ...}, wall_seconds)`` — the GPU-ladder
    CEQ (the tool's metric) and the ECDF-from-raw-lives CEQ, both computed
    from the same run, so the estimator comparison costs nothing extra on
    the GPU. ``rqmc=True`` selects the digital-shift scrambled Sobol stream
    (engine.run(rqmc=True)); ``owen=True`` switches to the Owen scramble and
    ``direct_w=True`` to the direct-chi2-LUT scale variant (research
    variants; the deployed page always uses Threefry).
    """
    result, elapsed = timed_run(engine, index, simulations, rqmc=rqmc, owen=owen, direct_w=direct_w)
    return {
        "ladder": ce_for_quantiles(result.quantiles[0], gamma, tables),
        "ecdf": ce_for_raw_spending(result.spending[0], gamma, tables),
    }, elapsed


class TestCeqConvergenceSmoke(unittest.TestCase):
    """Small-scale correctness smoke tests (no heavy convergence run)."""

    @unittest.skipUnless(_has_wgpu(), "wgpu package not installed")
    def setUp(self):
        self.config = cfg.instance_config()
        if not PRICE_PATH.exists():
            self.skipTest("downloaded_prices.csv missing (run py -3.14 download.py)")
        self.tables = cal.build_model_tables(self.config)
        self.gamma = self.config["simulation"].gamma
        self.index = _strategy_index(self.config)
        self.engine = engine_mod.Engine(config=self.config, price_path=str(PRICE_PATH), batch_size=256)

    def test_ce_deterministic_per_seed(self):
        first = self.engine.run(2000, allocation_indices=[self.index], include_bequest=False)
        second = self.engine.run(2000, allocation_indices=[self.index], include_bequest=False)
        self.assertTrue(np.array_equal(first.quantiles, second.quantiles))
        self.assertEqual(
            ce_for_quantiles(first.quantiles[0], self.gamma, self.tables),
            ce_for_quantiles(second.quantiles[0], self.gamma, self.tables),
        )

    def test_distinct_seeds_yield_distinct_ce(self):
        self.engine.seed = SEED_BASE
        q_a = self.engine.run(2000, allocation_indices=[self.index], include_bequest=False)
        self.engine.seed = SEED_BASE + SEED_STEP
        q_b = self.engine.run(2000, allocation_indices=[self.index], include_bequest=False)
        self.assertFalse(np.array_equal(q_a.quantiles, q_b.quantiles))

    def test_nested_prefixes_small_in_large(self):
        # With the same seed, an N-run is the first N lives of a larger run:
        # the N=100 simulation lives used by a 100-sim run must exist inside
        # the same seed's 10,000-sim run, ordered identically.
        self.engine.seed = SEED_BASE
        small = self.engine.run(100, allocation_indices=[self.index], include_bequest=False)
        large = self.engine.run(10_000, allocation_indices=[self.index], include_bequest=False)
        self.assertTrue(np.array_equal(small.spending[0], large.spending[0, :100]),
                        "N-run must be the prefix of the larger run for the same seed")

    def test_ce_sane_at_default_gamma(self):
        self.engine.seed = SEED_BASE
        result = self.engine.run(2000, allocation_indices=[self.index], include_bequest=False)
        ce = ce_for_quantiles(result.quantiles[0], self.gamma, self.tables)
        self.assertTrue(np.isfinite(ce) and ce > 0.0)
        median = float(result.quantiles[0, 100])
        self.assertLess(ce, median * 1.05)
        self.assertGreater(ce, median * 0.5)




def run_experiment(
    target_sims: int,
    seeds_m: int,
    batch_size: int,
    low_counts: tuple,
    rqmc: bool = False,
    owen: bool = False,
    direct_w: bool = False,
) -> None:
    """Run the convergence experiment and print the results to the console.

    Results are console-only (the JSON dumps were removed — regenerate with
    ``> file.json`` redirection if a machine-readable record is ever wanted).
    """
    config = cfg.instance_config()
    if not PRICE_PATH.exists():
        raise SystemExit("downloaded_prices.csv missing (run: py -3.14 download.py)")
    tables = cal.build_model_tables(config)
    gamma = config["simulation"].gamma
    index = _strategy_index(config)

    engine = engine_mod.Engine(config=config, price_path=str(PRICE_PATH), batch_size=batch_size)
    print(f"WebGPU adapter: {engine.adapter_info}")
    print(f"Strategy: {STRATEGY}")
    if rqmc:
        variant = "Owen scramble" if owen else "digital-shift"
        if direct_w:
            variant += " + direct chi2 LUT scale"
        print(f"Sampler: RQMC ({variant} Sobol)")
    else:
        print("Sampler: Threefry (default)")
    print(f"CEQ metric: gamma={gamma}, theta=0, lambda=0 (defaults), kappa={compute_kappa(gamma, tables):.6f}")
    print(f"Seeds (M={seeds_m}): {[SEED_BASE + k * SEED_STEP for k in range(seeds_m)]}")
    print(f"Target: N = {target_sims:,} per seed (same {seeds_m} seeds); "
          f"deep study at N = {', '.join(f'{n:,}' for n in low_counts)}\n")

    seeds = [SEED_BASE + k * SEED_STEP for k in range(seeds_m)]

    # --- Reference: CEQ at the target N (100k), M seeds ---------------------
    target_ce = {est: np.empty(seeds_m, dtype=np.float64) for est in ("ladder", "ecdf")}
    target_times = np.empty(seeds_m, dtype=np.float64)
    for m, seed in enumerate(seeds):
        engine.seed = seed
        per_seed, elapsed = ceqs_for_seed(engine, index, target_sims, gamma, tables,
                                        rqmc=rqmc, owen=owen, direct_w=direct_w)
        for est in target_ce:
            target_ce[est][m] = per_seed[est]
        target_times[m] = elapsed
    ce_target = {est: float(v.mean()) for est, v in target_ce.items()}
    mcse_target = {est: float(v.std(ddof=1)) for est, v in target_ce.items()}
    print(f"=== Target run: N = {target_sims:,} (M = {seeds_m} seeds) ===")
    for est in ("ladder", "ecdf"):
        print(f"  [{est}] per-seed CEQ: " + "  ".join(f"{v:,.2f}" for v in target_ce[est]))
        print(f"  [{est}] mean CE = ${ce_target[est]:,.2f}   MCSE = ${mcse_target[est]:,.2f} "
              f"({mcse_target[est] / ce_target[est] * 100:.2f}% relative)")
    print(f"  run time per seed: min {target_times.min():.3f}s   "
          f"max {target_times.max():.3f}s   mean {target_times.mean():.3f}s\n")

    # --- Deep study: N = 100 / 1k / 10k, same M seeds -----------------------
    print(f"=== Convergence study vs the 100k target [{STRATEGY}] ===")
    rows = {est: [] for est in ("ladder", "ecdf")}
    level_per_seed = {est: {} for est in ("ladder", "ecdf")}
    for n in low_counts:
        per_seed = {est: np.empty(seeds_m, dtype=np.float64) for est in ("ladder", "ecdf")}
        times = np.empty(seeds_m, dtype=np.float64)
        for m, seed in enumerate(seeds):
            engine.seed = seed
            ce, elapsed = ceqs_for_seed(engine, index, n, gamma, tables,
                                      rqmc=rqmc, owen=owen, direct_w=direct_w)
            for est in per_seed:
                per_seed[est][m] = ce[est]
            times[m] = elapsed
        t_min, t_max, t_mean = float(times.min()), float(times.max()), float(times.mean())
        for est in ("ladder", "ecdf"):
            mean = float(per_seed[est].mean())
            mcse = float(per_seed[est].std(ddof=1))
            bias = mean - ce_target[est]
            rmse = float(np.sqrt(np.mean((per_seed[est] - ce_target[est]) ** 2)))
            z_bias = bias / mcse if mcse > 0 else float("inf")
            scaled = mcse_target[est] * math.sqrt(target_sims / n)
            scaling_ok = 0.5 <= mcse / scaled <= 2.0
            rows[est].append({
                "n": n, "mean_ce": mean, "bias": bias, "mcse": mcse, "rmse": rmse,
                "z_bias": z_bias, "scaled_mcse": scaled, "scaling_ok": bool(scaling_ok),
                "time_min_s": t_min, "time_max_s": t_max, "time_mean_s": t_mean,
                "per_seed_time_s": [round(float(v), 4) for v in times],
            })
            level_per_seed[est][n] = per_seed[est]
            print(f"  [{est}] {n:>8,} {mean:>10,.2f} {bias:>+9,.2f} {mcse:>9,.2f} "
                  f"{mcse / mean * 100:>8.2f}% {rmse:>9,.2f} {rmse / ce_target[est] * 100:>8.2f}% "
                  f"{z_bias:>+8.2f} "
                  f"{t_min:>9.3f} {t_max:>9.3f} {t_mean:>10.3f}")
        print(f"  per-seed run time (s): " + "  ".join(f"{v:.4f}" for v in times))

    print()
    print("Notes:")
    print("  MCSE(N) = std of the M per-seed CEQ at N (sampling error of the N-sim estimator).")
    print("  RMSE(N) = sqrt(mean((CE_N,m - CE_100k_target)^2)) (bias + variance vs the refined target).")
    print("  'scaling ok' = MCSE(N) within 0.5x-2x of MCSE(100k) * sqrt(100k/N) (the 1/sqrt(N) law).")
    print("  run times are full GPU-pipeline wall seconds for the whole N-sim run (one value per seed).")
    print("  estimators: [ladder] = the tool's GPU 201-point histogram ladder (ce_for_quantiles);")
    print("  [ecdf] = same 199-point CEQ formula but quantiles are linearly interpolated order")
    print("  statistics of the N raw lives (ce_for_raw_spending) - same asymptotic target, no")
    print("  histogram binning bias at low N, no extra GPU cost (scored from the same run).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--lower", default="100,1000,10000",
                        help="comma-separated small-N levels (default 100,1000,10000)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rqmc", action="store_true",
                        help="use the digital-shift scrambled Sobol (RQMC) sampler "
                             "instead of Threefry for every run")
    parser.add_argument("--owen", action="store_true",
                        help="with --rqmc, use the Owen (nested) scramble instead of "
                             "the rigid digital shift")
    parser.add_argument("--direct-w", dest="direct_w", action="store_true",
                        help="with --rqmc, draw the chi2 scale directly from one "
                             "LUT-inverted uniform (research variant)")
    args = parser.parse_args()
    if args.simulations < 1 or args.seeds < 2:
        parser.error("--simulations must be positive and --seeds at least 2")
    low_counts = tuple(int(x) for x in args.lower.split(",") if x.strip())
    if not low_counts:
        parser.error("--lower must list at least one simulation count")
    run_experiment(args.simulations, args.seeds, args.batch_size, low_counts,
                   args.rqmc, args.owen, args.direct_w)


if __name__ == "__main__":
    if "--smoke" in sys.argv[1:]:
        sys.argv = [sys.argv[0]]
        unittest.main(verbosity=2)
    else:
        main()