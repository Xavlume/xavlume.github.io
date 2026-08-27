"""Python WebGPU compute harness for the retirement engine.

This module drives the exact WGSL pipeline that the browser build runs, using
the `wgpu` package: returns and layoffs are generated on-chip from the seed,
the five house strategies x seven accumulation paths accumulate into 35
retirement states per simulation, every (house, accumulation, bridge, post)
allocation is solved by parallel bisection on the GPU, Composite Ulcer Index
scores are tracked per path, and the 201-point quantile ladders are reduced on
the GPU by the same two-pass histogram shader the browser uses.

It shares the shader sources, the 144-byte params buffer layout, the packed
model buffer and the allocation metadata with :mod:`build_html`, so the Python
engine is a numerically equivalent reference for the standalone page.

Install the optional runtime with:

    py -3.14 -m pip install wgpu

Example:

    py -3.14 engine.py --simulations 1000 --allocations 5040
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import calibration
import config as cfg

ROOT = Path(__file__).resolve().parent
SHADER_DIR = ROOT / "shaders"
DEFAULT_PRICE_PATH = ROOT / "downloaded_prices.csv"

# Pass files concatenated in this order form the main compute module.
MAIN_SHADER_PASSES = ("common.wgsl", "returns.wgsl", "accumulation.wgsl", "solver.wgsl", "drawdown_ui.wgsl")
QUANTILES_SHADER = "quantiles.wgsl"
BEQUEST_SHADER = "bequest.wgsl"

BISECTION_STEPS = 24
SEED = 42
SOLVER_LOW = 300.0
SOLVER_HIGH = 10_000_000.0


@dataclass
class SimulationResult:
    """Everything the engine computed for one run."""

    names: List[str]
    spending: np.ndarray          # (allocations, simulations) annual net spending
    quantiles: np.ndarray         # (allocations, 201) monthly spending quantiles
    ui_means: np.ndarray          # (allocations,) mean Composite Ulcer Index
    house_outcomes: np.ndarray    # (houses, simulations, 2) (bought, buy_month)
    estate: np.ndarray            # (allocations, grid, 201) terminal-estate quantiles
    elapsed: float                # wall-clock seconds
    adapter: Dict[str, object] = field(default_factory=dict)


def load_shader_sources() -> Dict[str, str]:
    """Read every WGSL pass file once; the engine and the HTML builder share it."""
    sources = {}
    for name in MAIN_SHADER_PASSES:
        sources[name] = (SHADER_DIR / name).read_text(encoding="ascii")
    sources[QUANTILES_SHADER] = (SHADER_DIR / QUANTILES_SHADER).read_text(encoding="ascii")
    sources[BEQUEST_SHADER] = (SHADER_DIR / BEQUEST_SHADER).read_text(encoding="ascii")
    return sources


def _main_shader(sources: Dict[str, str]) -> str:
    return "\n".join(sources[name] for name in MAIN_SHADER_PASSES)


def make_params(
    config,
    n_simulations: int,
    n_allocations: int,
    batch_sims: int,
    sim_offset: int,
    seed: int = SEED,
    columns_per_workgroup: int = 1,
    generate_leveraged: bool = True,
) -> bytes:
    """Build the 144-byte params buffer; layout identical to the browser's makeParams."""
    lc = config["lifecycle"]
    tax = config["tax"]
    cma = config["cma"]
    co = config["career"]
    tables = calibration.build_model_tables(config)

    total_months = tables["accum_months"] + tables["retire_months"]
    m75_start = int((75 - lc.retirement_age) * 12)
    post_wedge_month = int((lc.pension_start_age - lc.retirement_age) * 12)

    params = np.zeros(144, dtype=np.uint8)
    u32 = params.view(np.uint32)
    f32 = params.view(np.float32)

    u32[0:4] = [n_simulations, n_allocations, total_months, tables["accum_months"]]
    u32[4:8] = [tables["retire_months"], len(cfg.FUNDS), config["simulation"].bisection_steps, m75_start]
    u32[8:12] = [post_wedge_month, lc.current_age, lc.career_start_age, lc.retirement_age]
    f32[12:16] = [
        tax.annual_distribution_yield / 12.0,
        tax.tax_on_distributions,
        (1.0 + cma.hisa_annual_real_return) ** (1.0 / 12.0) - 1.0,
        tax.capital_gains_inclusion_rate,
    ]
    f32[16:20] = [
        tax.capital_gains_tax_rate,
        cma.cash_wedge_years,
        tax.meltdown_bracket_annual / 12.0,
        tax.oas_clawback_threshold / 12.0,
    ]
    f32[20:24] = [
        tax.oas_clawback_rate,
        co.employer_match_rate,
        co.employer_match_percent,
        float(cfg.PATH_COUNT),
    ]
    u32[24:28] = [seed, config["calibration"].skew_degrees_freedom, batch_sims, sim_offset]
    f32[28:32] = [
        cma.real_borrow_rate_annual / 12.0,
        cma.extra_mer_15 / 12.0,
        cma.extra_mer_20 / 12.0,
        co.layoff_annual_probability,
    ]
    # dispatch.z: 1 = generate the leveraged return series (VEQT1.5/VEQT2);
    # the Python engine always runs the full 5,040-strategy space.
    u32[32:36] = [columns_per_workgroup, 0, 1 if generate_leveraged else 0, 0]
    return params.tobytes()


class Engine:
    """Persistent WebGPU device, pipelines and buffers for the full pipeline."""

    def __init__(
        self,
        config=None,
        price_path: Optional[str] = None,
        batch_size: int = 1024,
        seed: int = SEED,
    ):
        self.config = config if config is not None else cfg.instance_config()
        self.price_path = str(price_path or DEFAULT_PRICE_PATH)
        self.batch_size = batch_size
        self.seed = seed

        import wgpu
        from wgpu.utils import get_default_device

        self.wgpu = wgpu
        self.device = get_default_device()
        sources = load_shader_sources()
        self.sources = sources

        visibility = wgpu.ShaderStage.COMPUTE
        layout_entries = []
        for binding in range(7):
            layout_entries.append(
                {
                    "binding": binding,
                    "visibility": visibility,
                    "buffer": {"type": "storage" if binding in (1, 6) else "read-only-storage"},
                }
            )
        self.bind_group_layout = self.device.create_bind_group_layout(entries=layout_entries)
        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )
        self.shader_module = self.device.create_shader_module(
            code=_main_shader(sources)
        )
        self.pipelines = {
            name: self.device.create_compute_pipeline(
                layout=self.pipeline_layout,
                compute={"module": self.shader_module, "entry_point": name},
            )
            for name in (
                "generate_returns",
                "generate_layoffs",
                "accumulate",
                "solve",
                "track_drawdowns",
            )
        }

        # Quantile module: separate 3-binding layout.
        quantile_layout_entries = [
            {"binding": 0, "visibility": visibility, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": visibility, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": visibility, "buffer": {"type": "storage"}},
        ]
        self.quantile_layout = self.device.create_bind_group_layout(entries=quantile_layout_entries)
        self.quantile_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.quantile_layout]
        )
        self.quantile_module = self.device.create_shader_module(
            code=sources[QUANTILES_SHADER]
        )
        self.quantile_pipeline = self.device.create_compute_pipeline(
            layout=self.quantile_pipeline_layout,
            compute={"module": self.quantile_module, "entry_point": "quantiles"},
        )

        # Bequest module: separate 7-binding layout (0-4 read-only, 5-6 the
        # read-write estate ladder output and the persistent accumulation
        # histogram - two read-write bindings, within the AMD/D3D12 limit; the
        # per-simulation inputs are packed into one sim_data buffer to stay
        # within the max-8-storage-buffers-per-stage limit). Three entry
        # points mirror the solver's batching so no single dispatch outlives
        # the Windows TDR watchdog: bequest_reset zeroes the persistent
        # histogram, bequest_walk accumulates one batch of lives per dispatch,
        # bequest_final reduces the histogram to the 201-point ladders.
        bequest_layout_entries = [
            {"binding": b, "visibility": visibility, "buffer": {"type": "read-only-storage"}}
            for b in range(5)
        ] + [
            {"binding": 5, "visibility": visibility, "buffer": {"type": "storage"}},
            {"binding": 6, "visibility": visibility, "buffer": {"type": "storage"}},
        ]
        self.bequest_layout = self.device.create_bind_group_layout(entries=bequest_layout_entries)
        self.bequest_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bequest_layout]
        )
        self.bequest_module = self.device.create_shader_module(
            code=sources[BEQUEST_SHADER]
        )
        self.bequest_pipelines = {
            name: self.device.create_compute_pipeline(
                layout=self.bequest_pipeline_layout,
                compute={"module": self.bequest_module, "entry_point": name},
            )
            for name in ("bequest_reset", "bequest_walk", "bequest_final")
        }

    @property
    def adapter_info(self) -> Dict[str, object]:
        try:
            return dict(self.device.adapter.info)
        except Exception:  # pragma: no cover - adapter info is best-effort
            return {}

    def _buffer(self, data: np.ndarray, usage: int):
        return self.device.create_buffer_with_data(data=data, usage=usage)

    # -- public API ---------------------------------------------------------
    def run_returns(self, simulations: int, months: Optional[int] = None) -> np.ndarray:
        """Run only the return-generation pass; returns (simulations, months, 5) f32.

        Used by the parity test to compare the WGSL Threefry skew-t sampler
        against the NumPy CPU reference.
        """
        tables = calibration.build_model_tables(self.config)
        total_months = months or (tables["accum_months"] + tables["retire_months"])
        params = self._buffer(
            np.frombuffer(
                make_params(self.config, simulations, 1, simulations, 0, seed=self.seed),
                dtype=np.uint8,
            ),
            self.wgpu.BufferUsage.STORAGE,
        )
        scratch = self.device.create_buffer(
            size=simulations * total_months * len(cfg.FUNDS) * 4,
            usage=self.wgpu.BufferUsage.STORAGE | self.wgpu.BufferUsage.COPY_SRC,
        )
        model_buffer = self._buffer(
            calibration.build_model_buffer(self.config, self.price_path),
            self.wgpu.BufferUsage.STORAGE,
        )
        dummy_ro = self.device.create_buffer(size=4, usage=self.wgpu.BufferUsage.STORAGE)
        dummy_rw = self.device.create_buffer(size=4, usage=self.wgpu.BufferUsage.STORAGE)
        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": index, "resource": {"buffer": buffer}}
                for index, buffer in enumerate(
                    [params, scratch, dummy_ro, model_buffer, dummy_ro, dummy_ro, dummy_rw]
                )
            ],
        )
        encoder = self.device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(self.pipelines["generate_returns"])
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups((simulations * total_months + 63) // 64, 1, 1)
        pass_.end()

        readback = self.device.create_buffer(
            size=simulations * total_months * len(cfg.FUNDS) * 4,
            usage=self.wgpu.BufferUsage.MAP_READ | self.wgpu.BufferUsage.COPY_DST,
        )
        encoder.copy_buffer_to_buffer(
            scratch, 0, readback, 0, simulations * total_months * len(cfg.FUNDS) * 4
        )
        self.device.queue.submit([encoder.finish()])
        readback.map_sync(self.wgpu.MapMode.READ)
        data = np.frombuffer(
            readback.read_mapped(0, simulations * total_months * len(cfg.FUNDS) * 4),
            dtype=np.float32,
        ).copy()
        readback.unmap()
        return data.reshape(simulations, total_months, len(cfg.FUNDS))

    def run(
        self,
        simulations: int,
        allocation_indices: Optional[List[int]] = None,
        columns_per_workgroup: int = 1,
    ) -> SimulationResult:
        """Run the complete on-chip pipeline for the requested strategies.

        columns_per_workgroup mirrors the deployed page's dispatch shaping
        (default 1 = the reference one-column-per-thread shape); results are
        byte-identical for any value.
        """
        if simulations < 1:
            raise ValueError("simulations must be positive")
        wgpu = self.wgpu
        tables = calibration.build_model_tables(self.config)
        total_months = tables["accum_months"] + tables["retire_months"]
        career_years = tables["career_years"]
        path_count = cfg.PATH_COUNT
        house_count = cfg.HOUSE_COUNT

        metadata, names = calibration.allocation_metadata(self.config, allocation_indices)
        n_allocations = len(names)
        # The params buffer carries columns_per_workgroup (1 by default - the
        # reference shape): solve/track_drawdowns run that many allocation
        # columns per thread. The engine always dispatches one workgroup row
        # per allocation, so at columns = 1 the grid covers the allocation
        # space exactly once; at columns > 1 the shader's stride loop re-solves
        # some allocations (idempotent writes, byte-identical results). The
        # browser instead shrinks the grid to ceil(allocationCount / columns),
        # which is why the shipped default is 1 in both.
        model_values = calibration.build_model_buffer(self.config, self.price_path)

        storage_usage = wgpu.BufferUsage.STORAGE
        allocation_buffer = self._buffer(metadata, storage_usage)
        model_buffer = self._buffer(model_values, storage_usage)
        spending_buffer = self.device.create_buffer(
            size=n_allocations * simulations * 4,
            usage=storage_usage | wgpu.BufferUsage.COPY_SRC,
        )
        readback = self.device.create_buffer(
            size=n_allocations * simulations * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        dummy_ro = self.device.create_buffer(size=4, usage=storage_usage)

        # Packed scratch buffer sized for the worst-case (full) batch.
        scratch = self.device.create_buffer(
            size=(
                self.batch_size * total_months * len(cfg.FUNDS)
                + self.batch_size * career_years
                + house_count * path_count * self.batch_size * 4
                + house_count * self.batch_size * 2
                + self.batch_size * n_allocations
                + self.batch_size * path_count  # memoized accumulation-phase UI
            )
            * 4,
            usage=storage_usage | wgpu.BufferUsage.COPY_SRC,
        )

        spending = np.empty((n_allocations, simulations), dtype=np.float32)
        house_outcomes = np.zeros((house_count, simulations, 2), dtype=np.float32)
        ui_scores = np.zeros((simulations, n_allocations), dtype=np.float32)
        started = time.perf_counter()

        # Global packed copy of the per-batch scratch data needed by the
        # bequest estate pass: monthly returns, retirement states and house
        # outcomes, all indexed by the GLOBAL simulation id, in one buffer
        # (mirrors the main module's scratch packing; keeps the bequest
        # module within the max-8-storage-buffers-per-stage limit). The
        # bequest walk itself is dispatched per batch (like the solver) and
        # accumulates into the persistent histogram, so no single dispatch
        # outlives the TDR watchdog; only the tiny reset/final passes run
        # once.
        estate_grid = len(self.config["bequest"].estate_grid_fractions)
        estate_hist_words = n_allocations * estate_grid * 514  # 512 bins + min + max
        sim_data_words = (
            simulations * total_months * len(cfg.FUNDS)
            + house_count * path_count * simulations * 4
            + house_count * simulations * 2
        )
        sim_data_buffer = self.device.create_buffer(
            size=sim_data_words * 4,
            usage=storage_usage | wgpu.BufferUsage.COPY_DST,
        )
        estate_buffer = self.device.create_buffer(
            size=n_allocations * estate_grid * 201 * 4,
            usage=storage_usage | wgpu.BufferUsage.COPY_SRC,
        )
        estate_hist_buffer = self._buffer(
            np.zeros(estate_hist_words, dtype=np.uint32), storage_usage
        )
        estate_readback = self.device.create_buffer(
            size=n_allocations * estate_grid * 201 * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        # Per-batch readbacks of house outcomes and per-path UI scores. The
        # copies happen INSIDE the batch loop below: scratch is overwritten by
        # every subsequent batch, so it must never be re-read after the loop.
        house_readback = self.device.create_buffer(
            size=house_count * self.batch_size * 2 * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        ui_readback = self.device.create_buffer(
            size=self.batch_size * n_allocations * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        offset = 0
        while offset < simulations:
            count = min(self.batch_size, simulations - offset)
            params = self._buffer(
                np.frombuffer(
                    make_params(self.config, simulations, n_allocations, count, offset, seed=self.seed, columns_per_workgroup=columns_per_workgroup),
                    dtype=np.uint8,
                ),
                storage_usage,
            )
            bind_group = self.device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {"binding": index, "resource": {"buffer": buffer}}
                    for index, buffer in enumerate(
                        [params, scratch, allocation_buffer, model_buffer, dummy_ro, dummy_ro, spending_buffer]
                    )
                ],
            )
            bequest_bind_group = self.device.create_bind_group(
                layout=self.bequest_layout,
                entries=[
                    {"binding": index, "resource": {"buffer": buffer}}
                    for index, buffer in enumerate(
                        [
                            params, spending_buffer, sim_data_buffer,
                            allocation_buffer, model_buffer, estate_buffer, estate_hist_buffer,
                        ]
                    )
                ],
            )
            encoder = self.device.create_command_encoder()
            # Zero the persistent bequest histograms before the first batch.
            if offset == 0:
                pass_ = encoder.begin_compute_pass()
                pass_.set_pipeline(self.bequest_pipelines["bequest_reset"])
                pass_.set_bind_group(0, bequest_bind_group)
                pass_.dispatch_workgroups(n_allocations * max(estate_grid, 1), 1, 1)
                pass_.end()
            for name, workgroups in (
                ("generate_returns", (count * total_months + 63) // 64),
                ("generate_layoffs", (count * career_years + 63) // 64),
                ("accumulate", (count + 63) // 64),
                ("solve", (count + 63) // 64),
                ("track_drawdowns", (count + 63) // 64),
            ):
                pass_ = encoder.begin_compute_pass()
                pass_.set_pipeline(self.pipelines[name])
                pass_.set_bind_group(0, bind_group)
                if name == "accumulate":
                    pass_.dispatch_workgroups(workgroups, path_count * house_count, 1)
                else:
                    pass_.dispatch_workgroups(workgroups, n_allocations if name in ("solve", "track_drawdowns") else 1, 1)
                pass_.end()
            # Persist this batch's returns / states / houses for the bequest
            # estate pass (scratch itself is overwritten by the next batch).
            returns_bytes = count * total_months * len(cfg.FUNDS) * 4
            states_offset = count * total_months * len(cfg.FUNDS) + count * career_years
            houses_offset = states_offset + house_count * path_count * count * 4
            global_states_offset = simulations * total_months * len(cfg.FUNDS)
            encoder.copy_buffer_to_buffer(
                scratch, 0, sim_data_buffer, offset * total_months * len(cfg.FUNDS) * 4, returns_bytes
            )
            for block in range(house_count * path_count):
                encoder.copy_buffer_to_buffer(
                    scratch, (states_offset + block * count * 4) * 4,
                    sim_data_buffer, (global_states_offset + block * simulations * 4 + offset * 4) * 4,
                    count * 16,
                )
            for h in range(house_count):
                encoder.copy_buffer_to_buffer(
                    scratch, (houses_offset + h * count * 2) * 4,
                    sim_data_buffer,
                    (global_states_offset + house_count * path_count * simulations * 4 + h * simulations * 2 + offset * 2) * 4,
                    count * 8,
                )
            # Bequest walk for THIS batch only (w = f * w*, estates folded
            # into the persistent histogram) - bounded per-dispatch cost like
            # the solver, so the Windows TDR watchdog is never hit.
            pass_ = encoder.begin_compute_pass()
            pass_.set_pipeline(self.bequest_pipelines["bequest_walk"])
            pass_.set_bind_group(0, bequest_bind_group)
            pass_.dispatch_workgroups(n_allocations * max(estate_grid, 1), 1, 1)
            pass_.end()
            # Persist this batch's house outcomes and drawdown scores for the
            # CPU readback below (scratch is overwritten by the next batch).
            encoder.copy_buffer_to_buffer(
                scratch, houses_offset * 4, house_readback, 0, house_count * count * 2 * 4
            )
            encoder.copy_buffer_to_buffer(
                scratch, (houses_offset + house_count * count * 2) * 4,
                ui_readback, 0, count * n_allocations * 4,
            )
            self.device.queue.submit([encoder.finish()])
            # Read this batch's house outcomes and Composite UI scores back
            # while the readback buffers still hold this batch's data.
            house_readback.map_sync(wgpu.MapMode.READ)
            house_data = np.frombuffer(
                house_readback.read_mapped(0, house_count * count * 2 * 4), dtype=np.float32
            ).copy()
            house_readback.unmap()
            ui_readback.map_sync(wgpu.MapMode.READ)
            ui_data = np.frombuffer(
                ui_readback.read_mapped(0, count * n_allocations * 4), dtype=np.float32
            ).copy()
            ui_readback.unmap()
            house_outcomes[:, offset:offset + count, :] = house_data.reshape(house_count, count, 2)
            ui_scores[offset:offset + count, :] = ui_data.reshape(count, n_allocations)
            offset += count

        # Spending readback (allocation-major).
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(spending_buffer, 0, readback, 0, n_allocations * simulations * 4)
        self.device.queue.submit([encoder.finish()])
        readback.map_sync(wgpu.MapMode.READ)
        spending = np.frombuffer(
            readback.read_mapped(0, n_allocations * simulations * 4), dtype=np.float32
        ).copy()
        readback.unmap()
        spending = spending.reshape(n_allocations, simulations)

        # House outcomes and per-path UI scores were already read back inside
        # the batch loop (scratch is batch-local and overwritten every batch,
        # so a post-loop re-read would see only the LAST batch's data).
        ui_means = ui_scores.mean(axis=0)
        quantiles = self._quantiles_on_gpu(spending, simulations, n_allocations)
        estate = self._estate_final_on_gpu(
            spending_buffer, sim_data_buffer,
            allocation_buffer, model_buffer, estate_buffer, estate_hist_buffer,
            estate_readback, simulations, n_allocations, estate_grid,
        )
        elapsed = time.perf_counter() - started
        return SimulationResult(
            names=names,
            spending=spending,
            quantiles=quantiles,
            ui_means=ui_means,
            house_outcomes=house_outcomes,
            estate=estate,
            elapsed=elapsed,
            adapter=self.adapter_info,
        )

    # -- GPU quantile reduction ---------------------------------------------
    def _quantiles_on_gpu(self, spending: np.ndarray, simulations: int, n_allocations: int) -> np.ndarray:
        """Reduce the spending buffer to 201 quantiles per allocation on the GPU."""
        wgpu = self.wgpu
        storage_usage = wgpu.BufferUsage.STORAGE
        spending_buffer = self._buffer(spending.reshape(-1), storage_usage | wgpu.BufferUsage.COPY_SRC)
        quantile_buffer = self.device.create_buffer(
            size=n_allocations * 201 * 4, usage=storage_usage | wgpu.BufferUsage.COPY_SRC
        )
        quantile_readback = self.device.create_buffer(
            size=n_allocations * 201 * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        params = self._buffer(
            np.frombuffer(
                make_params(self.config, simulations, n_allocations, simulations, 0, seed=self.seed),
                dtype=np.uint8,
            ),
            storage_usage,
        )
        bind_group = self.device.create_bind_group(
            layout=self.quantile_layout,
            entries=[
                {"binding": index, "resource": {"buffer": buffer}}
                for index, buffer in enumerate([params, spending_buffer, quantile_buffer])
            ],
        )
        encoder = self.device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(self.quantile_pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups(n_allocations, 1, 1)
        pass_.end()
        encoder.copy_buffer_to_buffer(quantile_buffer, 0, quantile_readback, 0, n_allocations * 201 * 4)
        self.device.queue.submit([encoder.finish()])
        quantile_readback.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(
            quantile_readback.read_mapped(0, n_allocations * 201 * 4), dtype=np.float32
        ).copy()
        quantile_readback.unmap()
        return data.reshape(n_allocations, 201)

    # -- GPU terminal-estate reduction (bequest ladders) ---------------------
    def _estate_final_on_gpu(
        self,
        spending_buffer,
        sim_data_buffer,
        allocation_buffer,
        model_buffer,
        estate_buffer,
        estate_hist_buffer,
        estate_readback,
        simulations: int,
        n_allocations: int,
        estate_grid: int,
    ) -> np.ndarray:
        """Reduce the accumulated estate histogram to (allocations, grid, 201) quantiles.

        The per-batch bequest_walk dispatches (inside ``run``) already folded
        every life's estate into the persistent histogram; this tiny final
        pass locates the 201 quantiles and copies the ladders back to CPU.
        """
        wgpu = self.wgpu
        storage_usage = wgpu.BufferUsage.STORAGE
        params = self._buffer(
            np.frombuffer(
                make_params(self.config, simulations, n_allocations, simulations, 0, seed=self.seed),
                dtype=np.uint8,
            ),
            storage_usage,
        )
        bind_group = self.device.create_bind_group(
            layout=self.bequest_layout,
            entries=[
                {"binding": index, "resource": {"buffer": buffer}}
                for index, buffer in enumerate(
                    [
                        params, spending_buffer, sim_data_buffer,
                        allocation_buffer, model_buffer, estate_buffer, estate_hist_buffer,
                    ]
                )
            ],
        )
        encoder = self.device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(self.bequest_pipelines["bequest_final"])
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups(n_allocations * max(estate_grid, 1), 1, 1)
        pass_.end()
        encoder.copy_buffer_to_buffer(
            estate_buffer, 0, estate_readback, 0, n_allocations * estate_grid * 201 * 4
        )
        self.device.queue.submit([encoder.finish()])
        estate_readback.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(
            estate_readback.read_mapped(0, n_allocations * estate_grid * 201 * 4),
            dtype=np.float32,
        ).copy()
        estate_readback.unmap()
        return data.reshape(n_allocations, estate_grid, 201)


def _allocation_selection(count: int) -> List[int]:
    """Select a strided subset of strategy indices (matches the browser URL param)."""
    names = sorted(calibration.build_allocation_defs().keys())
    total = len(names)
    if count < 1 or count > total:
        raise ValueError(f"allocations must be between 1 and {total}")
    if count == total:
        return list(range(total))
    return np.linspace(0, total - 1, count, dtype=int).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--allocations", type=int, default=len(calibration.build_allocation_defs()))
    parser.add_argument("--price-path", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.simulations < 1:
        parser.error("simulations and batch-size must be positive")

    try:
        engine = Engine(
            price_path=args.price_path,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        result = engine.run(
            args.simulations,
            allocation_indices=_allocation_selection(args.allocations),
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    best = int(np.argmax(result.quantiles[:, 100]))
    print(f"WebGPU adapter: {engine.adapter_info}")
    print(f"Completed {len(result.names):,} allocations x {args.simulations:,} simulations in {result.elapsed:.2f}s.")
    print(f"Top allocation by median monthly spending: {result.names[best]}")
    print(f"Median monthly spending: ${result.quantiles[best, 100]:,.1f}")

    if args.output:
        with open(args.output, "w", encoding="ascii") as output_file:
            json.dump(
                {
                    "strategies": [
                        {
                            "name": name,
                            "quantiles": np.round(result.quantiles[index], 1).tolist(),
                            "ui": round(float(result.ui_means[index]), 3),
                        }
                        for index, name in enumerate(result.names)
                    ],
                    "meta": {"backend": "WebGPU", "n_strategies": len(result.names)},
                },
                output_file,
                separators=(",", ":"),
            )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
