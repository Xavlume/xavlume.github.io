# AGENTS.md — Wealth & Lifetime Allocation Engine

Guidance for AI coding agents working in this repository. Read this before
changing anything. The README is written for humans; this file is written for
agents that will edit the code.

---

## 1. What this project is

A **client-side, GPU-accelerated lifetime asset-allocation and retirement
simulation** for a Quebec (Montreal) resident. The entire application ships as
one self-contained static HTML file (`index.html`) that runs **100% in the
browser via WebGPU** and deploys to GitHub Pages. There is no backend.

It simulates **5,040 strategies** (every combination of 5 housing plans × 7
accumulation paths × 12 bridge-phase options × 12 post-pension-phase options),
runs each simulated life through the full Quebec fiscal lifecycle on the GPU,
solves the **maximum sustainable annual net spending** `w*` by 24-step parallel
bisection, and ranks strategies by **Certainty Equivalent** with an instant
(no-GPU) drawdown-adjusted re-ranking in pure JavaScript.

The code is largely AI-generated (see README). The owner does not review every
line, so **write conservatively, keep changes minimal and surgical, and lean on
the test suites** — they are the safety net.

---

## 2. Repository map

| Path | Role | Agents may edit? |
|---|---|---|
| `config.py` | **Single source of truth**: typed dataclasses for every fiscal rule, CMA, career/housing/tax number + human formatting helpers | ✅ yes — source |
| `calibration.py` | Skew-t calibration, packed model buffer, derived lifecycle tables, allocation metadata, JSON payload, **NumPy CPU reference sampler** | ✅ yes — source |
| `engine.py` | Python `wgpu` harness running the *same* WGSL passes (numerical reference for the deployed page) | ✅ yes — source |
| `build_html.py` | Template + shaders + runtime JS → `index.html`. **Contains the entire browser runtime (`RUNTIME_JS`)** | ✅ yes — source |
| `shaders/*.wgsl` | The five GPU compute passes + quantile reduction | ✅ yes — source |
| `template_readonly.html` | UI markup/CSS foundation + mock dataset script, ingested at build time | ✅ **yes for agents — see §4** |
| `index.html` | **Generated** deployable application (tracked in git) | ❌ never hand-edit — rebuild |
| `download.py` | Price-history download & backfill pipeline (yfinance + regressions) | ✅ yes — source |
| `downloaded_prices.csv` | Calibrated monthly real price history (**gitignored — not in the repo**) | ❌ regenerate via `download.py` |
| `canadian_cpi.csv`, `dexcaus.csv` | Input data for the download pipeline (tracked) | ✅ yes — data |
| `tests/` | Unit, parity and E2E suites | ✅ yes — source |
| `README.md`, `LICENSE`, `.gitignore` | Docs / MIT / ignores | ✅ yes |

### Suggested reading order for a new agent

1. `README.md` — human-oriented overview, math summary.
2. `config.py` — every parameter lives here.
3. `shaders/common.wgsl` — buffer contracts and memory layout (the "API" of the GPU side).
4. `calibration.py` — how the payload and model buffer are built.
5. `build_html.py` — how the page is assembled and where the JS runtime lives.
6. `tests/` — what correctness means in this project.

---

## 3. Commands (Python 3.14)

> **Interpreter invocation:** the project pins Python 3.14. `py -3.14` is the
> Windows Python Launcher; on Linux/macOS (CI containers, Codespaces, Docker)
> use `python3.14` or a 3.14 virtualenv's `python` instead. Do not silently
> fall back to an older interpreter — the code uses `X | None` typing and
> 3.14-era stdlib behavior.

```bash
# Build the standalone application (needs downloaded_prices.csv present)
py -3.14 build_html.py                          # writes index.html
py -3.14 build_html.py --price-path downloaded_prices.csv --output index.html

# Run the Python WebGPU engine directly (optional wgpu runtime)
py -3.14 -m pip install wgpu
py -3.14 engine.py --simulations 1000 --allocations 5040 [--output out.json]

# Regenerate the price history if downloaded_prices.csv is missing (network)
py -3.14 -m pip install yfinance scipy pandas numpy
py -3.14 download.py                            # writes downloaded_prices.csv

# Tests (each is a standalone unittest file; run from anywhere)
py -3.14 tests/test_calibration.py              # unit tests (no GPU)
py -3.14 tests/test_parity.py                   # CPU vs WebGPU parity + determinism (needs wgpu + GPU)
py -3.14 tests/test_e2e_selenium.py             # headless-browser E2E (needs Chrome/Edge + selenium)
py -3.14 tests/test_e2e_selenium.py --simulations 1000 --headed
```

Dependencies: `numpy` + `pandas` are required for the build; `scipy`/`yfinance`
only for `download.py`; `wgpu` only for `engine.py`/`test_parity.py`; `selenium`
only for the E2E test. `build_html.py` does **not** need `wgpu` (it is imported
lazily inside `engine.Engine.__init__`).

### Test feedback loop (run fastest first)

Iterate on the cheap checks before the heavy ones:

1. `tests/test_calibration.py` — pure CPU unit tests; seconds, no GPU.
2. `build_html.py` — verifies asset assembly and that `RUNTIME_JS`/template
   markers are still structurally valid; seconds.
3. `tests/test_parity.py` — WGSL vs NumPy parity + determinism; needs `wgpu`
   and a GPU; seconds to a minute.
4. `tests/test_e2e_selenium.py --simulations 1000` — full browser run
   (WebGPU + UI + formula gates); slowest, needs Chrome/Edge **and a
   freshly built `index.html`**. Run last, only after 1–3 pass.

---

## 4. `template_readonly.html` — the read-only boundary (important)

`template_readonly.html` is the UI foundation: all markup, CSS, and the mock
dataset script. The name means **read-only *to the build pipeline* — not to
you**:

- The **build pipeline must never write to it**. `build_html.py` only ever
  *reads* it (`TEMPLATE_PATH.read_text(...)`). Do not change the pipeline so
  that any script mutates or regenerates `template_readonly.html`. It is a
  hand-maintained source file, an input to the build.
- **AI agents may edit it directly.** Changing the HTML structure or CSS here is
  how you change the UI — the changes flow into `index.html` on the next build.
- If you touch its `<script>` block, keep the `// Mock Quantitative Dataset`
  comment intact: `_replace_mock_script()` regex-requires it and the build
  raises `RuntimeError` if it is missing.
- Understand what the build replaces: the template's **entire** `<script>`
  block (mock dataset + static UI JS) is swapped for the real model-data
  `<script>` tag plus `RUNTIME_JS`. **JavaScript you write inside the
  template's script block does NOT survive the build.** The browser runtime
  lives in `RUNTIME_JS` inside `build_html.py`. Rule of thumb:
  - HTML / CSS → edit the template.
  - Runtime JS logic → edit `RUNTIME_JS` in `build_html.py`.
  - DOM ids/classes referenced by runtime JS must exist in the template.

### Modifying `RUNTIME_JS` safely (string trap)

`RUNTIME_JS` is pure JavaScript held in a **raw Python string**
(`RUNTIME_JS = r"""…"""`) inside `build_html.py`. Common agent failure modes:

- **Never put `"""`** (three consecutive double-quotes) in the JS — it
  terminates the Python string early and the module stops parsing.
- **A backslash immediately before the closing `"""`** escapes a quote and
  breaks the module (SyntaxError at import). Watch trailing backslashes at the
  end of the string.
- Backslashes are *literal* in a raw string (that is what the JS needs:
  `"\n"`, `"\t"`, `"\uXXXX"` stay as-is) — do **not** add extra escaping, and
  do not switch the string to a non-raw variant.
- `build_html.py` importing cleanly only proves the *Python* string is intact;
  syntactically invalid *JavaScript* inside it still builds and only explodes
  in the browser. After editing `RUNTIME_JS`:
  1. Run the build — catches Python-level corruption immediately.
  2. If `node` is available, syntax-check the extracted JS:

     ```python
     # node --check the JS inside RUNTIME_JS (fast, no browser needed)
     import pathlib, re, subprocess
     src = pathlib.Path("build_html.py").read_text(encoding="utf-8")
     js = re.search(r'RUNTIME_JS = r"""(.*?)"""', src, re.S).group(1)
     pathlib.Path("_runtime_check.js").write_text(js, encoding="utf-8")
     subprocess.run(["node", "--check", "_runtime_check.js"])  # then delete the temp file
     ```
  3. Run the E2E suite — it fails on SEVERE console errors, i.e. JS parse
     failures in the real browser.

---

## 5. Design principles (invariants to preserve)

From the README; these are what make the project coherent. **Violating any of
them breaks parity tests or the E2E suite.**

1. **One source of truth.** Every fiscal rule, CMA, real-estate number and
   calibration control is defined once, in the typed dataclasses of `config.py`
   (`LifecycleConfig`, `CareerConfig`, `HousingConfig`, `TaxFiscalConfig`,
   `CMAConfig`, `ModelCalibrationConfig`, `SmileConfig`, `SimulationConfig`).
   Never hardcode a parameter in a shader, in `RUNTIME_JS`, or in
   `calibration.py`. To add/change a parameter: update `config.py`, then wire it
   through `calibration.py` (`build_model_tables`, `lifecycle_constants`,
   `model_defaults`) and — if it is editable in the settings window — the
   `INPUT_SCHEMA` table in `build_html.py` (schema-driven binding between
   human formatting like `$88,000` / `4.30% / yr` and raw decimals).

2. **Structured buffer contracts.** Every storage-buffer region is declared and
   documented in `shaders/common.wgsl` (and `quantiles.wgsl`). There are no
   undocumented magic offsets. If you move something in a buffer, update the
   layout docs and every consumer.

3. **Exact parity — three synchronized implementations of the same math.**
   The numerical formulas exist in three places that must stay identical:
   - Python: `calibration.py` (and `engine.py`)
   - JavaScript: `RUNTIME_JS` in `build_html.py` (`buildDynamicModel`,
     `calibrateReturnModel`, `monthlyTax`, `pensionAmounts`, `ceForQuantiles`,
     `computeKappa`, …)
   - WGSL: `shaders/*.wgsl` (`monthly_tax`, `net_monthly`, `interp_tax`,
     `test_solvency`, `drawdown_ui`, …)
   A formula change in one place **must** be mirrored in the other two.
   Enforcement: `test_parity.py` (Python vs GPU) and `test_e2e_selenium.py`
   (formula gates: `CE == CE_base × exp(−λ·UI)`, CMA toggle, schema binding).

4. **Determinism.** Fixed seed `42` (`SEED` in `engine.py`, `defaultSeed` in the
   payload). Paths come from the Threefry-2x32-20 counter-based PRNG, with a
   NumPy mirror (`calibration.returns_cpu` / `threefry_uniforms`) that must
   produce byte-equivalent streams to the WGSL sampler. Don't change seeds,
   counters, or the PRNG without updating **both** samplers.

5. **No parameter duplication.** If you find yourself typing the same number in
   two files, you are doing it wrong — put it in `config.py` instead.

### Zero-tolerance rules (NEVER)

1. **NEVER** add a third read-write storage-buffer binding in WGSL — breaks
   AMD/D3D12 dispatch (§6.2).
2. **NEVER** hand-edit `index.html` — always edit the sources and run
   `build_html.py`.
3. **NEVER** change the PRNG, its counters, or the seed in WGSL without
   updating the NumPy mirror in `calibration.py` (and any JS port) — the parity
   test proves exact CPU/GPU agreement.
4. **NEVER** hardcode raw fiscal/market numbers in shaders, `RUNTIME_JS` or
   `calibration.py` — add them to `config.py` dataclasses.
5. **NEVER** change the params-buffer or model-buffer layout in only one of the
   implementations — mirror it in Python, JS and WGSL together (§6.3/§6.4).
6. **NEVER** make any script write to `template_readonly.html` — the pipeline
   only reads it (§4).
7. **NEVER** commit `downloaded_prices.csv` — it is intentionally gitignored;
   regenerate it with `download.py` instead.
8. **NEVER** `git commit` (or stage-then-commit) without the **user's explicit
   permission**. Leave all changes in the working tree, unstaged or staged, for
   the user to review and commit themselves. "Definition of done" does NOT
   include committing — if a task description or checklist seems to say so,
   the user's permission still wins. When you finish a task, report exactly
   which files changed, but do not commit.

---

## 6. Architecture reference (what the code actually does)

### 6.1 GPU pipeline

The main compute module is the concatenation of these files, **in this fixed
order** (`MAIN_SHADER_PASSES` in `engine.py`):

```
common.wgsl → returns.wgsl → accumulation.wgsl → solver.wgsl → drawdown_ui.wgsl
```

`quantiles.wgsl` is a separate, self-contained module (own `Params`, own
3-binding layout) run after the main pipeline.

Passes and dispatch:

| Pass | Entry point | Work | Dispatch |
|---|---|---|---|
| 1 | `generate_returns` | Threefry skew-t monthly returns for 5 funds | `⌈sims×months/64⌉` |
| 2 | `generate_layoffs` | annual layoff flags (0.5× salary) | `⌈sims×careerYears/64⌉` |
| 3 | `accumulate` | 35 retirement states (5 houses × 7 paths) | `⌈sims/64⌉ × (7×5)` |
| 4 | `solve` | 24-step bisection → sustainable spending w* | `⌈sims/64⌉ × allocations` |
| 5 | `track_drawdowns` | Composite Ulcer Index per path | `⌈sims/64⌉ × allocations` |
| 6 | `quantiles` | 201-point spending quantile ladder | `allocations` workgroups |
| 7 | `bequest_reset` | zero the persistent estate histogram | `allocations × grid` workgroups (tiny) |
| 8 | `bequest_walk` | estate at w = f·w* per (allocation, grid fraction) — **one dispatch per simulation batch** | `allocations × grid` workgroups |
| 9 | `bequest_final` | 201-point estate ladder from the accumulated histogram | `allocations × grid` workgroups (tiny) |

Passes 7–9 live in the separate `bequest.wgsl` module (own 7-binding layout;
two read-write bindings; the per-simulation inputs are packed into one
`sim_data` buffer so the module stays within the max-8-storage-buffers-per-
stage limit). `bequest_walk` is batched like `solve` so no single dispatch
outlives the Windows TDR watchdog (~2 s): each batch's lives are folded into
a persistent fixed-scale histogram (`estate_hist`), and only the tiny
reset/final dispatches run once per simulation.

### 6.2 Bindings (main module)

- `0` params (read) — the **144-byte params buffer**
- `1` scratch (**read-write**) — packed regions: returns, layoffs, 35 states,
  house outcomes, drawdown scores (offsets computed in `common.wgsl`)
- `2` allocations metadata (read) — 12 u32 per strategy
- `3` model_values (read) — the packed model buffer
- `4`, `5` unused read-only placeholders (keep the 7-entry layout)
- `6` spending_results (**read-write**) — (allocation × simulation) w*

⚠️ **AMD/D3D12 quirk (do not regress):** Chrome's D3D12 backend on AMD drivers
silently drops writes beyond **two read-write storage buffers per stage** —
hence the scratch-packing design and why only bindings 1 and 6 are read-write.
Do not add a third read-write binding.

### 6.3 The 144-byte params buffer

Layout is 9 × `vec4` (`dimensions`, `solver`, `calendar`, `constants0-2`,
`generate`, `generate1`, `dispatch`) — documented field-by-field in
`common.wgsl`'s `Params`. The `dispatch.x` field is the
allocation-columns-per-workgroup count from
`SimulationConfig.columns_per_workgroup` (16 by default in the deployed page,
1 in the Python engine): `solve`/`track_drawdowns` loop over that many
allocation columns per thread so the grid stays ONE dispatch per pass
(splitting the allocation space into multiple dispatches silently corrupts
results on some AMD D3D12 drivers). Keep the grid small enough that a single
dispatch never runs longer than the Windows TDR watchdog (~2 s) on GPUs
without mid-dispatch compute preemption. The two knobs that bound
per-dispatch wall time are `SimulationConfig.batch_size` (sims per batch)
and `columns_per_workgroup`, and they interact: at small batch sizes a large
`columns_per_workgroup` leaves the GPU under-occupied (threads ≈
batch × allocations / columns), so wall time collapses into the per-thread
serial chain (batch 100 × columns 128 measured ~0.87 s/dispatch vs ~0.09 s at
columns 8). Defaults (250 sims, 16 columns) keep dispatches ~0.2 s on a
modern dGPU with plenty of threads — roughly 5× the TDR headroom of the
previous 1,024-sim/128-column default while still being faster. There are
**three synchronized implementations**:

- `engine.make_params` (Python, `engine.py`)
- `makeParams` (`RUNTIME_JS` in `build_html.py`)
- the `Params` structs in `common.wgsl` **and** `quantiles.wgsl`

Any change to the layout must be mirrored in **all four places**.

### 6.4 The packed model buffer (`calibration.build_model_buffer`)

f32 words in this order (layout documented in both `common.wgsl` and
`calibration.py`):

```
[0, careerYears×6)     career rows: (net retirement stream, net house stream,
                       tax rate, cumulative TFSA room, cumulative RRSP room, salary)
[*, +retireMonths×4)   month0: (smile/12, healthcare/12, gross pension, net pension)
[*, +retireMonths×4)   month1: (RRIF factor, OAS max, 0, 0)
[*, +54)               tax tail: gross/net interpolation grids; [44..47] monthly
                       thresholds; slot 48 is an UNUSED PAD; [49..53] tax rates
[*, +18)               skew-t constants: xi[3], omega[3], delta[3], Cholesky[9]
[*, +11)               house constants (target capital, mortgage principal/rate,
                       taxes, rent, FHSA, HBP, house count, target property
                       value at index 10 — read only by the bequest pass)
[*, +1+grid)           bequest estate-grid tail: [grid count, spending
                       fractions...], consumed only by bequest.wgsl
```

Mirrors: `buildDynamicModel` in `RUNTIME_JS` and the offset math in
`common.wgsl` (`career_years()*6 + retire*8 + 54 + 18`). **Do not change the
layout in only one of the three.**

### 6.5 Allocation metadata & strategy space

- **5,040 strategies** = 5 house plans × 7 accumulation paths × 12 bridge × 12
  post. Names: `HOUSE_X_ACCUM_BRIDGE_POST`, e.g.
  `HOUSE_NONE_VEQT_VEQT+CASH_VEQT`.
- Fund codes: `VEQT=0, VEQT1.5=1, VEQT2=2, VGRO=3, VBAL=4, DECLINING=5, RISING=6`.
- Phase options (12): each of the 5 funds, its `+CASH` variant, plus the two
  glidepaths. Glidepaths **cannot** be combined with `+CASH`
  (`allocation_phase_code` raises).
- GPU metadata row: 12 u32 = `[accumCode, bridgeCode, postCode, flags,
  accumGlide.xy, bridgeGlide.xy, postGlide.xy, 0, 0]`; flags = bit0 cash bridge,
  bit1 cash post, bits2-4 house code. The HTML payload embeds a compact 4-u32
  form that the browser expands (`selectedAllocationBuffer`). The strided
  subset selection for `?allocations=N` must stay identical between
  `_allocation_selection` (engine) and `selectedAllocationIndices` (JS).

### 6.6 Key math (see README for the full write-up)

- **Skew-t returns**: `ST(ξ, ω, δ, Σ, ν)`, ν = 5 fixed; `b_ν` evaluated exactly
  for integer ν; sampled on-chip from Threefry uniforms (|N(0,1)| skew +
  χ²ν/ν scale + 3 normals + Cholesky rotation), clipped at −95%.
- **Leverage**: `r1.5 = max(−0.95, 1.5·r − 0.5·r_borrow − fee1.5)`,
  `r2.0 = max(−0.95, 2.0·r − r_borrow − fee2.0)`.
- **CE**: `CE(γ) = [Σ_t w_t·s_t^(1−γ)]^(1/(1−γ)) · κ(smile, γ)` with
  mortality-adjusted, discounted intertemporal weights (CPM2014-like table in
  `calibration.py`, 50% mortality reduction, 1% annual discount).
- **Re-ranking**: `CE_adj = CE(γ) × exp(−λ × Composite UI)` — pure JS from
  cached 201-point quantile ladders, never a GPU re-simulation.
- **Composite UI**: drawdowns on cumulative return indices,
  `UI_phase = √(mean(D_t²))`, weights `(0.60 bridge, 0.25 post, 0.15 accum)`,
  renormalized to `(0, 0.65, 0.35)` when the bridge is empty.
- **Solver**: fixed 24-step bisection over `w ∈ [300, 10,000,000]`; the solved
  w* is **net** disposable lifestyle spending (owning vs renting rank on the
  same scale).
- **Quantiles**: 201-point ladder (P0, P0.5, …, P100), GPU 2-pass histogram;
  default floor percentile = 10 (`P10`).

---

## 7. Common tasks (recipes)

**"Change a fiscal parameter"** (e.g. OAS clawback rate):
1. Edit the field in `config.py` (e.g. `TaxFiscalConfig.oas_clawback_rate`).
2. If it is editable in the UI, add/update the entry in `INPUT_SCHEMA`
   (`build_html.py`) — indices are the flat position of the input inside the
   tab's `.form-grid-2 .input-field .form-input` containers in DOM order. The
   E2E "SCHEMA INPUT BINDING" section is a regression gate for this.
3. If the shaders consume it, it must reach the model buffer or params buffer
   (§6.3/§6.4) — update `build_model_tables`/`make_params` (Python), the
   matching JS, and the WGSL offset docs together.
4. Rebuild and re-run all three test suites.

**"Change the UI layout/colors"**: edit `template_readonly.html` only (markup +
CSS). Rebuild → `index.html`. Don't touch runtime JS there; see §4.

**"Change simulation logic"** (a fiscal mechanic in retirement): find it in the
WGSL (`solver.wgsl` `test_solvency`, `accumulation.wgsl`, `drawdown_ui.wgsl`),
mirror in `buildDynamicModel`/JS helpers, and if the Python engine/parity tests
cover it, mirror there too. Update the layout docs in `common.wgsl` if offsets
change. Run parity + E2E.

**"Add a fund / phase option"**: update `config.py` (`FUNDS`, `PHASE_OPTIONS`,
`GLIDEPATH_OPTIONS` — counts are derived, so mostly automatic), check
`_phase_funds`/`allocation_phase_code`/`phase_fund` (WGSL) handle the new code,
add a badge in `CHIP_META` (both the template's copy — mock-only — and
`RUNTIME_JS`'s copy), and keep `fund_indices` order consistent everywhere. E2E
asserts the badge chip colors.

**"Change the PRNG"**: update `common.wgsl` `threefry_u32`/`threefry_uniforms`,
the NumPy mirror (`calibration.py`), and the JS port if one exists. Then run
`test_parity.py` — it proves CPU/GPU agreement to float32 precision.

### Adding a new settings input (worked example)

`INPUT_SCHEMA` indices are the **flat** 0-based position of an input across
*all* `.form-grid-2` containers in a tab, in DOM order — not per-grid. For
example, `tab-re` currently binds 13 inputs (indices 0–12). If you add a new
`<div class="input-field"><input class="form-input" …></div>` as the 4th input
of the tab (flat position 3):

1. New row in `INPUT_SCHEMA`: `["tab-re", 3, "someModelField", "money"]`.
2. Every existing `tab-re` entry with index ≥ 3 shifts up by +1 (3→4, 4→5, …).
3. If the E2E test references positional indices in that tab (the "SCHEMA INPUT
   BINDING" section uses `s['tab-re:6']` and `s['tab-re:8']`), update those
   references too, or they will read the wrong field.
4. Add the field to `model_defaults`/`build_model_tables` (`calibration.py`)
   and, if the shaders consume it, to the model/params buffer in all three
   implementations (§6.3/§6.4).
5. Rebuild, then run the E2E "SCHEMA INPUT BINDING" gate, which re-reads every
   tab through `schemaInputs()`.

---

## 8. Gotchas

- **`index.html` is committed** (GitHub Pages serves it). After any change that
  affects the page, rebuild `index.html` so the working tree stays deployable.
  Never edit it by hand — your edits will be overwritten by the next build.
  Whether the rebuild gets committed is the user's call (see zero-tolerance
  rule 8).
- **`downloaded_prices.csv` is gitignored.** A fresh clone has no price history
  and `build_html.py` will fail until `download.py` has been run. If you see a
  missing-CSV error, regenerate it (network required).
- **`build_html.py` fails hard** (explicit `RuntimeError`s) if the template is
  missing the mock-script marker, the CMA-toggle anchor, or the House Savings
  Escalation field. If you rename/restructure those template regions, update
  the builders' `_inject_*` functions and `_replace_mock_script`.
- **Settings schema indices are positional.** Reordering inputs inside a
  template `.form-grid-2` silently rebinds fields; keep `INPUT_SCHEMA` in sync
  or the E2E schema gate catches you.
- **Run pacing**: default batch 250 sims (10,000 sims = 40 batches; each batch
  is one submit whose `solve` dispatch must stay well under the Windows TDR
  watchdog — see §6.3 for the batch/columns trade-off), slider 500–10,000;
  `?allocations=N`
  caps strategies for huge runs; GPU buffer preflight rejects runs that exceed
  `maxStorageBufferBindingSize`; a 120-second watchdog flags hung runs; device
  loss surfaces in the status bar and console. Console logs are prefixed
  `ENGINE` and `dumpDiagnostics()` prints a full snapshot.
- **Headless WebGPU** (E2E/CI): Chrome with
  `--enable-unsafe-webgpu --use-angle=d3d11`.
- **Python 3.14** is the pinned interpreter (`py -3.14`); the code uses modern
  typing (`X | None`) with `from __future__ import annotations`.
- Tests self-bootstrap their import path (`sys.path.insert` of the repo root)
  — run them directly as scripts, they do not need pytest.

## 9. Definition of done (checklist)

- [ ] `py -3.14 tests/test_calibration.py` passes.
- [ ] If GPU math touched: `py -3.14 tests/test_parity.py` passes.
- [ ] `py -3.14 build_html.py` succeeds and `index.html` is rebuilt.
- [ ] If the page is affected: `py -3.14 tests/test_e2e_selenium.py
      --simulations 1000` passes with **zero SEVERE console errors**.
- [ ] No hardcoded parameter duplicating `config.py`; no buffer layout changed
      in only one implementation; no read-write binding added.
- [ ] `template_readonly.html` untouched by any script; `index.html` never
      hand-edited; regenerated `index.html` present in the working tree; **no
      commit made** — committing happens only with the user's explicit
      permission (zero-tolerance rule 8).
