# Introduction

Hello. This is a little project I've been working on. It is the culmination of about 1 year into the subject of Retirement allocation testing.

It all started with the paper ["Beyond the Status Quo: A Critical Assessment of Lifecycle Investment Advice"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4590406), which is a paper that challenges the idea of reducing your equity allocation when nearing retirement. It advocates for a 100% equity globally diversified portfolio through working years, up to, and including retirement. 

I was exposed to this paper by the following YouTube video by Ben Felix: ["The Most Controversial Paper in Finance"](https://www.youtube.com/watch?v=-nPon8Ad_Ug), whose videos I love watching.

This got me started down a retirement backtesting rabbit hole, looking at block bootstrap methods and t distributions and the such. I have long had a passion for quantitative finance and investing, and this is yet another step in my journey.

What started as a single python script quickly grew in proportion, and I felt like I was spending more time refreshing myself on my python skills versus learning about lifetime allocation. Therefore, I started using generative AI to write the code for me, and I have not written a single line of code for this project. However, I am still in control of how it works in the backend and what techniques were used and why...

This repo contains AI generated code from models such as `Deepseek-V4-Flash` and `Gemini-3.7-Flash`. I am putting it on the internet in the case that this might be of interest to anyone out there. Note that I have not looked upon all the lines of this code, and so there might be some sneaky bugs I haven't found yet.

What follows is the AI generated readme file explaining what this tool is and how to use it:

# Wealth & Lifetime Allocation Engine

A client-side, GPU-accelerated lifetime asset-allocation and retirement
simulation for a Quebec resident, built to run 100% in the browser
via **WebGPU** and to deploy as a single static file on **GitHub Pages**.

`index.html` is the final standalone application. It embeds the
calibrated heavy-tailed market model, five WGSL compute passes and a
Wealthsimple-style institutional light UI in one self-contained file — no
backend, no local file dependencies (fonts come from the Google Fonts CDN).

```
.
├── config.py               # Single source of truth: typed dataclasses for every parameter
├── calibration.py          # Multivariate skew-t calibration, model buffers, CPU reference sampler
├── engine.py               # Python wgpu compute harness (exact WGSL execution parity)
├── build_html.py           # Standalone HTML builder (template + shaders + runtime -> index.html)
├── index.html              # Generated: the deployable GitHub Pages application
├── template_readonly.html  # Institutional light-mode UI foundation (read-only, ingested at build time)
├── download.py             # Price history download & backfill pipeline (writes downloaded_prices.csv)
├── downloaded_prices.csv   # Calibrated monthly real price history (VEQT / VGRO / VBAL)
├── canadian_cpi.csv        # CPI input for the download pipeline
├── dexcaus.csv             # USD/CAD FX input for the download pipeline
├── README.md
├── LICENSE
├── .gitignore
├── shaders/
│   ├── common.wgsl         # Shared buffer schemas, memory layout, PRNG and tax helpers
│   ├── returns.wgsl        # Threefry-2x32-20 PRNG & multivariate skew-t return generator
│   ├── accumulation.wgsl   # 5 house strategies x 7 paths, stochastic buy, FHSA/HBP
│   ├── solver.wgsl         # 24-step parallel bisection for sustainable spending w*
│   ├── drawdown_ui.wgsl    # Composite Ulcer Index tracking across lifecycle phases
│   └── quantiles.wgsl      # GPU-native 2-pass histogram reduction (201 quantiles/allocation)
└── tests/
    ├── test_calibration.py # Calibration unit tests
    ├── test_parity.py      # Python CPU vs. WebGPU numerical parity tests
    └── test_e2e_selenium.py# Automated headless-browser verification (5,040 strategies)
```

## Quick start

```bash
# 1. Build the standalone application (reads downloaded_prices.csv)
py -3.14 build_html.py

# 2. Open it: just double-click index.html (or host it anywhere,
#    including GitHub Pages — it is a single self-contained file)

# 3. Run the test suites
py -3.14 tests/test_calibration.py
py -3.14 tests/test_parity.py
py -3.14 tests/test_e2e_selenium.py          # needs Chrome/Edge + selenium
```

Optional Python runtime for the engine and parity tests: `py -3.14 -m pip install wgpu`.

## What the app does

The page simulates **5,040 strategies** — every combination of

* **5 housing plans** (renter, or buyer funded through a CASH / VBAL / VGRO /
  VEQT house fund),
* **7 accumulation paths** (VEQT, VEQT1.5, VEQT2, VGRO, VBAL, DECLINING
  glidepath, RISING glidepath),
* **12 bridge phase options** and **12 post-pension phase options** (each
  pure fund, its `+CASH` wedge variant, and the two glidepaths).

Each strategy runs the full Quebec fiscal lifecycle on the GPU: stochastic
layoffs, employer DC matching, TFSA/RRSP/FHSA/HBP room mechanics, stochastic
home purchase at the down-payment target, mortgage amortizing to zero at
retirement, RRSP meltdown, RRIF minimums, OAS clawback and enhanced QPP
deferral. For every simulated path the engine solves the **maximum sustainable
annual net spending** `w*` by 24-step parallel bisection, then reduces the
201-point monthly spending quantile ladder (P0…P100) on the GPU.

The dashboard ranks strategies by **Certainty Equivalent** (CRRA utility),
with instant post-processing: moving the Risk Aversion `γ` or Drawdown
Aversion `λ` sliders and pressing *Update Table & Re-Rank* re-ranks all 5,040
strategies in pure JavaScript from the cached quantiles, applying the exact
formula

```
CE_adj = CE(γ) × exp(−λ × Composite UI)
```

No GPU re-simulation is needed for any slider, filter, search or sort.

## The mathematical model

### Return calibration (multivariate skew-t)

Monthly simple returns for VEQT / VGRO / VBAL are computed from the price
history (`downloaded_prices.csv`). The engine calibrates a multivariate
skew-t distribution `ST(ξ, ω, δ, Σ, ν)`:

| parameter | meaning | calibration |
|---|---|---|
| `ξ` | location (mode) vector | `ξ = μ − ω δ b_ν`, with forward-looking CMA means `μ` |
| `ω` | scale vector | `ω_i = √(Σ_ii / (ν/(ν−2) − δ_i² b_ν²))` |
| `δ` | skewness vector | moment estimator `δ_i = cap·tanh(skew_i/3)`, clamped by `δᵀΣ⁻¹δ ≤ 0.99` |
| `Σ` | correlation | `Σ_ij = ((ν−2)/ν)·(cov_ij/(ω_i ω_j) + b_ν² δ_i δ_j)`, unit diagonal |
| `ν` | degrees of freedom | fixed at 5 (heavy tails) |

The settings window's **"Use expected returns above — off uses historical"**
toggle switches the mean vector `μ` between the forward-looking CMAs listed in
the CMAs tab and the historical sample means of the price history.

with `b_ν = √(ν/π)·Γ((ν−1)/2)/Γ(ν/2)` evaluated **exactly** for integer `ν`
(no Lanczos approximation). The shader samples `ST` on-chip through a
Threefry-2x32-20 counter-based PRNG: a `|N(0,1)|` skew component, a
χ²_ν/ν scale, three independent normals, and a Cholesky rotation of the
residual correlation `Σ − δδᵀ`. The parity test proves the NumPy reference
and the WGSL sampler agree to float32 precision.

The two leveraged funds are deterministic transforms of VEQT:
`r₁.₅ = max(−0.95, 1.5·r − 0.5·r_borrow − fee₁.₅)`,
`r₂.₀ = max(−0.95, 2.0·r − r_borrow − fee₂.₀)`, all clipped at −95%.

### Spending smile & intertemporal utility

Retirement spending follows a life-stage "smile" schedule (flat → declining
→ care phase), and the certainty equivalent is computed with mortality-aware,
discounted intertemporal weights (CPM2014 survival curve, 50% mortality
reduction, 1% annual discount):

```
CE(γ) = [ Σ_t w_t · s_t^(1−γ) ]^(1/(1−γ)) · κ(smile, γ)
```

### Composite Ulcer Index

Drawdowns are measured on **cumulative return indices** (never on net liquid
wealth, so planned decumulation cannot masquerade as a drawdown), with
`UI_phase = √(mean(D_t²))` and
`UI_comp = 0.60·UI_bridge + 0.25·UI_post + 0.15·UI_accum`
(re-normalized to `(0, 0.65, 0.35)` when the bridge phase is empty).

## Design principles

* **One source of truth.** Every fiscal rule, CMA, real-estate number and
  calibration control lives in the typed dataclasses of `config.py`; the
  Python engine, the calibration payload and the browser runtime all derive
  from the same defaults.
* **Structured buffer contracts.** Every WGSL storage-buffer region is
  declared and documented in `shaders/common.wgsl` (and
  `shaders/quantiles.wgsl`); there are no undocumented magic offsets. The
  128-byte params buffer is byte-identical between the Python runner and the
  browser's `makeParams`.
* **No parameter duplication.** The old engine hardcoded defaults across four
  files and raw WGSL offsets; this tree defines each number once.
* **Exact parity.** `engine.py` drives the *same* WGSL pass files the browser
  runs, from the *same* calibration code path, so the Python harness is a
  trustworthy numerical reference for the deployed page.

## Configuration

`config.py` exposes strongly-typed dataclasses (`LifecycleConfig`,
`CareerConfig`, `HousingConfig`, `TaxFiscalConfig`, `CMAConfig`,
`ModelCalibrationConfig`, `SmileConfig`, `SimulationConfig`) plus
human-friendly formatting/parsing helpers (`20.0%`, `$88,000`,
`4.30% / yr`) that the settings window uses to present and read the same
values the engine stores as raw decimals (`0.2`, `88000`, `0.043`).

## Troubleshooting

If the app reports that WebGPU is unavailable or a run does nothing:

1. Open the browser console (F12) — every step of the pipeline logs
   **`ENGINE`**-prefixed lines (adapter selection, buffer sizes, batch
   progress, quantile reduction) and every failure logs full context plus a
   diagnostics snapshot.
2. Run `dumpDiagnostics()` in the console to print a complete snapshot:
   browser version, adapter list, engine state and device-loss reason.
3. GPU device loss mid-run (crash/`TDR` on Windows) surfaces immediately in
   the console and in the status bar; a 120-second watchdog flags hung runs.
4. For headless/CI use, launch Chrome with
   `--enable-unsafe-webgpu --use-angle=d3d11`.

> Tip: for very large runs (e.g. 30k paths), load with
> `?allocations=1000` to cap GPU memory to a subset of strategies.

## License

MIT — use it, study it, fork it.
