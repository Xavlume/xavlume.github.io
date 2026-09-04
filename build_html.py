"""Build the standalone GitHub Pages HTML for the lifetime allocation engine.

The builder reads the institutional light-mode UI foundation
(``template_readonly.html``), replaces its mock dataset script with the full
WebGPU runtime, injects the calibrated model payload (``__MODEL_JSON__``) and
the WGSL shader sources, and writes ``index.html`` — a 100%
client-side, zero-backend application that runs entirely in the browser.

Usage:

    py -3.14 build_html.py [--price-path downloaded_prices.csv] [--output index.html]

The generated file is fully standalone: all styles, SVG, JavaScript and
shaders are inline; fonts come from the Google Fonts CDN.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import calibration
import config as cfg
import engine

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "template_readonly.html"
DEFAULT_PRICE_PATH = ROOT / "downloaded_prices.csv"
DEFAULT_OUTPUT_PATH = ROOT / "index.html"

# ---------------------------------------------------------------------------
# Markers replaced in the runtime JS and the template.
# ---------------------------------------------------------------------------
MODEL_MARKER = "__MODEL_JSON__"
SHADER_MARKER = "__SHADER_JSON__"
QUANTILES_MARKER = "__QUANTILES_JSON__"
BEQUEST_MARKER = "__BEQUEST_JSON__"
SOBOL_TABLE_MARKER = "__SOBOL_TABLE_B64__"
SOBOL_DIMS_MARKER = "__SOBOL_TABLE_DIMS__"

RUNTIME_JS = r"""
"use strict";
// ===========================================================================
// WebGPU runtime for the Wealth & Lifetime Allocation Engine.
//
// This script replaces the template's mock dataset with the real engine:
//  1. It reads the calibrated model payload from the #model-data script tag.
//  2. It builds the parameter buffers and runs the five WGSL compute passes
//     (returns -> layoffs -> accumulation -> solver -> drawdowns) plus the
//     GPU quantile reduction and the terminal-estate (bequest) ladders.
//  3. Risk Aversion (gamma), Drawdown Aversion (lambda), Bequest Intensity
//     (theta) and Bequest Curvature (k) re-rank the table INSTANTLY from the
//     cached 201-point quantile ladders (pure JavaScript, no GPU re-simulation):
//         CE_adj = CE(gamma, theta, k) x exp(-lambda x Composite UI)
//     where theta = k = 0 reduces CE(gamma, theta, k) to the base CE exactly.
// ===========================================================================

const MODEL = JSON.parse(document.getElementById("model-data").textContent);
const SHADER_SOURCE = __SHADER_JSON__;
const QUANTILES_SHADER_SOURCE = __QUANTILES_JSON__;
const BEQUEST_SHADER_SOURCE = __BEQUEST_JSON__;
const C = MODEL.constants;
const DEFAULTS = MODEL.defaults;
const TOTAL_ALLOCATIONS = MODEL.allocations.count;
const RUN_ALLOCATION_COUNT = Number(new URLSearchParams(location.search).get("allocations")) || TOTAL_ALLOCATIONS;
// The five underlying return series (VEQT, VEQT1.5, VEQT2, VGRO, VBAL).
// DECLINING/RISING accumulation glidepaths are monthly switching schedules,
// not return series, so they are sampled on-chip by the accumulate pass.
const RETURN_FUND_COUNT = 5;
// Upper bound (ms) for the per-batch progress-paint yield in the batch loop;
// see the comment at the await site.
const BATCH_YIELD_MS = 16;
const ALLOCATION_NAMES = MODEL.allocations.names;
const ALLOCATION_METADATA = new Uint32Array(MODEL.allocations.metadata);
// The "Leverage" checkbox in the settings window gates whether leveraged
// strategies (VEQT1.5 = code 1, VEQT2 = code 2) participate in a run. The
// compact metadata row is [accumCode, bridgeCode, postCode, flags] with fund
// codes VEQT=0..RISING=6 (see allocation_phase_code), so excluding codes 1/2
// from all three phase slots shrinks the strategy space from 5 houses x 7
// accumulation paths x 12 bridge x 12 post = 5,040 down to
// 5 x 5 x 8 x 8 = 1,600 strategies.
const LEVERAGED_FUND_CODES = [1, 2];

const state = {
  results: null,        // per-strategy simulation results (quantiles, ui, ...)
  dynamic: null,        // built dynamic model of the last run
  applied: null,        // control snapshot used for the last render
  sort: { column: "ce", ascending: false, active: false },
  deviceContext: null,
  devicePromise: null,
  activeRun: null,
  adapterText: "pending"
};
let activeStrategy = null;

function byId(id) { return document.getElementById(id); }
function money(value) { return value == null ? "—" : "$" + Math.round(value).toLocaleString("en-US"); }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch])); }
function f2(value) { return Number(value).toFixed(1); }
function setText(id, text) { byId(id).textContent = text; }

// ---------------------------------------------------------------------------
// Console diagnostics: every failure path logs an "ENGINE" line with context,
// and window.dumpDiagnostics() prints a full snapshot for bug reports.
// ---------------------------------------------------------------------------
function engineLog(level, ...args) {
  try {
    const prefix = "%cENGINE%c";
    const styles = ["background:#0f172a;color:#fff;border-radius:3px;padding:1px 6px;font-weight:700;", ""];
    console[level](prefix, ...styles, ...args);
  } catch (_) { /* console unavailable */ }
}
const logInfo = (...args) => engineLog("info", ...args);
const logDebug = (...args) => engineLog("debug", ...args);
const logWarn = (...args) => engineLog("warn", ...args);
const logError = (...args) => engineLog("error", ...args);

window.addEventListener("error", (event) => {
  logError("Uncaught error:", event.message, "at", event.filename + ":" + event.lineno,
           event.error && event.error.stack ? "\n" + event.error.stack : "");
});
window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason || {};
  logError("Unhandled promise rejection:", reason && reason.message ? reason.message : String(reason),
           reason && reason.stack ? "\n" + reason.stack : "");
});

async function dumpDiagnostics() {
  logInfo("=== ENGINE DIAGNOSTICS ===");
  logInfo("User agent:", navigator.userAgent);
  logInfo("WebGPU API present:", !!navigator.gpu);
  logInfo("URL:", location.href);
  logInfo("State:", JSON.stringify({
    results: state.results ? state.results.length : null,
    deviceReady: !!state.deviceContext,
    adapterText: state.adapterText,
    activeRun: !!state.activeRun,
    deviceLostReason: state.deviceLostReason || null,
    samplerRequested: new URLSearchParams(location.search).get("sampler") || null,
    sobolTableBuiltIn: sobolTableAvailable(),
    totals: {allocations: TOTAL_ALLOCATIONS, runAllocations: RUN_ALLOCATION_COUNT}
  }));
  if (navigator.gpu) {
    try {
      if (typeof navigator.gpu.requestAdapterInfo === "function") {
        const infos = await navigator.gpu.requestAdapterInfo();
        logInfo("Available adapters (" + infos.length + "):",
                infos.map(i => ({vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description})));
      } else {
        logWarn("navigator.gpu.requestAdapterInfo() not available on this browser version; adapter enumeration skipped.");
      }
    } catch (err) {
      logError("requestAdapterInfo() failed:", err);
    }
  }
  logInfo("=== END ENGINE DIAGNOSTICS ===");
}
window.dumpDiagnostics = dumpDiagnostics;

function uiSeverity(ui) {
  // Severity coloring for the mean Composite Ulcer Index: green = mild
  // (VBAL/VGRO), amber = moderate (100% equity), red = severe (leveraged).
  if (ui < 12) return "var(--brand-green)";
  if (ui < 20) return "var(--brand-amber)";
  return "#dc2626";
}

function setStatus(message, error) {
  const element = byId("gpu-status");
  element.textContent = message;
  element.className = "engine-chip " + (error ? "bad" : "ok");
}

// ---------------------------------------------------------------------------
// Simulation path-count slider (simple linear range, 500..10,000).
// ---------------------------------------------------------------------------
const SIM_MIN = 500, SIM_MAX = 10000, SIM_STEP = 500;
function simulationCountFromSlider() {
  const value = Number(byId("slider-sim-count").value);
  return Math.min(SIM_MAX, Math.max(SIM_MIN, Math.round(value / SIM_STEP) * SIM_STEP));
}
function simulationSliderPosition(count) {
  return Math.min(SIM_MAX, Math.max(SIM_MIN, Math.round(Number(count) / SIM_STEP) * SIM_STEP));
}

// ---------------------------------------------------------------------------
// Editable model inputs: schema-driven binding between the settings window's
// natural human formatting (20.0%, $88,000, 4.30% / yr) and the engine's raw
// decimals (0.2, 88000, 0.043).
// ---------------------------------------------------------------------------
function parseMoney(text) { return Number(String(text).replace(/[^0-9.\-]/g, "")); }
function parsePercent(text) { return Number(String(text).replace(/[^0-9.\-]/g, "")) / 100; }
function formatMoney(value) { return String(Math.round(value)); }
function formatPercent(value) { return (value * 100).toFixed(2); }

// Each entry: [tab id, input index within that tab, model field, kind]
// The input index is the flat position of the field inside the tab's
// .form-grid-2 containers, in DOM order. Kinds: years | money | pct | int | plain.
const INPUT_SCHEMA = [
  ["tab-career", 0, "currentAge", "years"],
  ["tab-career", 1, "careerStartAge", "years"],
  ["tab-career", 2, "retirementAge", "years"],
  ["tab-career", 3, "pensionStartAge", "years"],
  ["tab-career", 4, "deathAge", "years"],
  ["tab-career", 5, "startingSalary", "money"],
  ["tab-career", 6, "promotionPhase0", "pct"],
  ["tab-career", 7, "promotionPhase1", "pct"],
  ["tab-career", 8, "promotionPhase2", "pct"],
  ["tab-career", 9, "promotionPhase3", "pct"],
  ["tab-career", 10, "retirementSavingsStartAnnual", "money"],
  ["tab-career", 11, "retirementSavingsEscalationRate", "pct"],
  ["tab-career", 12, "savingsMaxFraction", "pct"],
  ["tab-career", 13, "employerMatchRate", "pct"],
  ["tab-career", 14, "employerMatchPercent", "pct"],
  ["tab-career", 15, "layoffAnnualProbability", "pct"],

  ["tab-re", 0, "propertyValue", "money"],
  ["tab-re", 1, "downPaymentFraction", "pct"],
  ["tab-re", 2, "closingCosts", "money"],
  ["tab-re", 3, "realMortgageRateAnnual", "pct"],
  ["tab-re", 4, "monthlyPropertyTaxesCondo", "money"],
  ["tab-re", 5, "monthlyMarketRent", "money"],
  ["tab-re", 6, "houseSavingsStartAnnual", "money"],
  ["tab-re", 7, "houseSavingsEscalationRate", "pct"],
  ["tab-re", 8, "houseSavingsMaxFraction", "pct"],
  ["tab-re", 9, "fhsaAnnualLimit", "money"],
  ["tab-re", 10, "fhsaMaxBalance", "money"],
  ["tab-re", 11, "hbpMaxWithdrawal", "money"],
  ["tab-re", 12, "hbpRepaymentYears", "years"],

  ["tab-tax", 0, "meltdownBracketAnnual", "money"],
  ["tab-tax", 1, "oasClawbackThreshold", "money"],
  ["tab-tax", 2, "capitalGainsInclusionRate", "pct"],
  ["tab-tax", 3, "capitalGainsTaxRate", "pct"],
  ["tab-tax", 4, "maxQppAge65", "money"],
  ["tab-tax", 5, "maxOasAge65", "money"],
  ["tab-tax", 6, "qppMaximumAnnual", "money"],
  ["tab-tax", 7, "qppMaximumMSGA", "money"],
  ["tab-tax", 8, "qppDeferralAnnual", "pct"],
  ["tab-tax", 9, "oasDeferralAnnual", "pct"],

  ["tab-cma", 0, "cmaVEQT", "pct"],
  ["tab-cma", 1, "cmaVGRO", "pct"],
  ["tab-cma", 2, "cmaVBAL", "pct"],
  ["tab-cma", 3, "hisaAnnualRealReturn", "pct"],
  ["tab-cma", 4, "annualDistributionYield", "pct"],
  ["tab-cma", 5, "taxOnDistributions", "pct"],
  ["tab-cma", 6, "realBorrowRateAnnual", "pct"],
  ["tab-cma", 7, "extraMer15", "pct"],
  ["tab-cma", 8, "extraMer20", "pct"],
  ["tab-cma", 9, "cashWedgeFraction", "pct"],
    // Glide fields interleave Declining (left column) / Rising (right column)
    // row by row so each grid row pairs the same fund: VEQT, then VGRO, then
    // VBAL — Declining share on the left, Rising share on the right.
    ["tab-cma", 10, "glidepathDeclining0", "pct"],
    ["tab-cma", 11, "glidepathRising0", "pct"],
    ["tab-cma", 12, "glidepathDeclining1", "pct"],
    ["tab-cma", 13, "glidepathRising1", "pct"],
    ["tab-cma", 14, "glidepathDeclining2", "pct"],
    ["tab-cma", 15, "glidepathRising2", "pct"],

  ["tab-spend", 0, "smilePhase0", "pct"],
  ["tab-spend", 1, "smilePhase1", "pct"],
  ["tab-spend", 2, "smilePhase2", "pct"],
  ["tab-spend", 3, "smilePhase3", "pct"],
  ["tab-spend", 4, "skewDegreesFreedom", "int"],
  ["tab-spend", 5, "deltaCap", "plain"],
  ["tab-spend", 6, "mortalityReductionFactor", "plain"],
  ["tab-spend", 7, "discountRateAnnual", "pct"],
];

function schemaInputs() {
  const inputs = {};
  for (const tabId of ["tab-career", "tab-re", "tab-tax", "tab-cma", "tab-spend"]) {
    const tab = byId(tabId);
    const list = Array.from(tab.querySelectorAll(".form-grid-2 .input-field .form-input"));
    list.forEach((element, index) => { inputs[tabId + ":" + index] = element; });
  }
  return inputs;
}

function formatForKind(kind, value) {
  if (kind === "pct") return formatPercent(value);
  if (kind === "money") return formatMoney(value);
  if (kind === "plain") return Number(value).toFixed(2);
  return String(value);  // years / int
}

function parseForKind(kind, text) {
  if (kind === "pct") return parsePercent(text);
  if (kind === "money") return parseMoney(text);
  return Number(text);
}

// Glidepath + cash-tent percentage fields round-trip through the same pct
// pipeline (50.0 <-> 0.50), so no special formatting is needed beyond making
// sure the parsed values land on the config arrays (readModelInputs) and
// formatting keeps two decimals (applyModelToInputs).

function applyModelToInputs(config) {
  const inputs = schemaInputs();
  INPUT_SCHEMA.forEach(([tabId, inputIndex, field, kind]) => {
    const element = inputs[tabId + ":" + inputIndex];
    if (!element) return;
    element.value = formatForKind(kind, modelValue(config, field));
  });
  const cmaToggle = byId("inp-use-forward-cmas");
  if (cmaToggle) cmaToggle.checked = !!config.useForwardLookingCmas;
}

function modelValue(config, field) {
  if (field.startsWith("promotionPhase")) return config.promotionPhases[Number(field.slice(-1))].growth;
  if (field.startsWith("smilePhase")) return config.smileSchedule[Number(field.slice(-1))].change;
  if (field.startsWith("cma")) return config.cmas[field.slice(3)];
  if (field.startsWith("glidepathDeclining")) return config.glidepathDeclining[Number(field.slice(-1))];
  if (field.startsWith("glidepathRising")) return config.glidepathRising[Number(field.slice(-1))];
  return config[field];
}

function readModelInputs() {
  const input = JSON.parse(JSON.stringify(MODEL.inputs));
  const inputs = schemaInputs();
  for (let index = 0; index < INPUT_SCHEMA.length; index++) {
    // NOTE: the schema stores the per-tab input index separately from the
    // global schema position — the lookup key MUST use the input index.
    const [tabId, inputIndex, field, kind] = INPUT_SCHEMA[index];
    const element = inputs[tabId + ":" + inputIndex];
    if (!element) continue;
    const value = parseForKind(kind, element.value);
    if (!Number.isFinite(value)) throw new Error("Invalid number in settings field " + field + ": '" + element.value + "'");
    if (field.startsWith("promotionPhase")) input.promotionPhases[Number(field.slice(-1))].growth = value;
    else if (field.startsWith("smilePhase")) input.smileSchedule[Number(field.slice(-1))].change = value;
    else if (field.startsWith("cma")) input.cmas[field.slice(3)] = value;
    else if (field.startsWith("glidepathDeclining")) input.glidepathDeclining[Number(field.slice(-1))] = value;
    else if (field.startsWith("glidepathRising")) input.glidepathRising[Number(field.slice(-1))] = value;
    else input[field] = value;
  }
  // The first promotion tier always begins at the career start age so the
  // salary trajectory and the tier labels stay consistent.
  input.promotionPhases[0].start = input.careerStartAge;
  // Expected-return source: ON = the CMAs listed in the settings, OFF = the
  // historical sample means of the calibrated price history.
  input.useForwardLookingCmas = byId("inp-use-forward-cmas").checked;
  const ages = [input.currentAge, input.careerStartAge, input.retirementAge, input.pensionStartAge, input.deathAge];
  if (ages.some(age => !Number.isInteger(age)) || input.currentAge >= input.careerStartAge || input.careerStartAge >= input.retirementAge || input.retirementAge >= input.pensionStartAge || input.pensionStartAge >= input.deathAge) {
    throw new Error("Ages must be ordered current < career start < retirement < pension < death.");
  }
  if (input.startingSalary <= 0 || input.skewDegreesFreedom <= 2 || input.skewDegreesFreedom % 1 !== 0) {
    throw new Error("Starting salary must be positive and skew degrees of freedom must be an integer greater than 2.");
  }
  // Glidepath shares: each pair must sum to ~100% and no single fund share
  // may exceed the max-share cap (mirrors config.glidepath_max_share).
  const maxShare = C.glidepathMaxShare || 0.5;
  const epsilon = 1e-6;
  for (const [label, shares] of [["DECLINING", input.glidepathDeclining], ["RISING", input.glidepathRising]]) {
    if (shares.some(v => !Number.isFinite(v) || v < 0)) throw new Error(label + " glidepath shares must be non-negative percentages.");
    if (Math.abs(shares[0] + shares[1] + shares[2] - 1.0) > epsilon) {
      throw new Error(label + " glidepath shares must sum to 100%.");
    }
    if (shares.some(v => v > maxShare + epsilon)) {
      throw new Error(label + " glidepath shares: no single fund may exceed " + (maxShare * 100).toFixed(0) + "%.");
    }
  }
  if (!Number.isFinite(input.cashWedgeFraction) || input.cashWedgeFraction < 0 || input.cashWedgeFraction > 1) {
    throw new Error("Cash wedge must be a percentage between 0% and 100% of the retirement span.");
  }
  return input;
}

// ---------------------------------------------------------------------------
// Fiscal & calibration helpers (exact ports of the engine's math).
// ---------------------------------------------------------------------------
// (The single-regime runtime calibration — inverse3/cholesky3/calibrateReturnModel —
// was retired with the two-state Markov model: the regime fit happens in
// Python at build time; only the CMA mean-shift is recomputed live, below.)

function exactLnGammaInteger(n) {
  let total = 0;
  for (let i = 1; i < n; i++) total += Math.log(i);
  return total;
}
function exactLnGammaHalfInteger(k) {
  let total = 0.5 * Math.log(Math.PI);
  for (let i = 1; i <= 2 * k; i++) total += Math.log(i);
  total -= k * Math.log(4);
  for (let i = 1; i <= k; i++) total -= Math.log(i);
  return total;
}
function exactLogGamma(value) {
  const rounded = Math.round(value);
  const isHalf = Math.abs(value - rounded) > 1e-9;
  const base = Math.floor(value);
  return isHalf ? exactLnGammaHalfInteger(base) : exactLnGammaInteger(rounded);
}
function exactBNu(nu) {
  if (nu <= 1) return 0;
  return Math.sqrt(nu / Math.PI) * Math.exp(exactLogGamma((nu - 1) / 2) - exactLogGamma(nu / 2));
}

// Live CMA mean-shift for the two-state Markov return model.  The regime
// fit (HMM + per-state skew-t sets) is calibrated in Python at build time and
// embedded in the payload; the toggle and the CMA inputs change the DRIFT only,
// exactly as in calibration.calibrate_two_state_markov: the stationary
// (regime-weighted) mean of the embedded states is shifted by the common
// constant that re-targets it on
//   forward-looking: log1p(CMA)/12 + cov_ii/2   (moment-matched log-mean)
//   historical:      the sample mean of the embedded price history
// so only xi changes — omega/delta/Cholesky/p00/p11 stay build-time fixed.
function applyCmaMeanShift(rm, config) {
  const values = MODEL.historicalReturns;
  const count = MODEL.historicalReturnCount;
  const covariance = new Array(3).fill(0);
  const means = [0, 0, 0];
  for (let row = 0; row < count; row++) for (let column = 0; column < 3; column++) means[column] += values[row * 3 + column];
  for (let column = 0; column < 3; column++) means[column] /= count;
  for (let row = 0; row < count; row++) for (let column = 0; column < 3; column++) covariance[column] += (values[row * 3 + column] - means[column]) * (values[row * 3 + column] - means[column]);
  for (let column = 0; column < 3; column++) covariance[column] /= count - 1;
  const targetMean = config.useForwardLookingCmas
    ? [Math.log1p(config.cmas.VEQT) / 12 + 0.5 * covariance[0],
       Math.log1p(config.cmas.VGRO) / 12 + 0.5 * covariance[1],
       Math.log1p(config.cmas.VBAL) / 12 + 0.5 * covariance[2]]
    : means;
  const prior0 = rm.prior0;
  const s0 = rm.states[0], s1 = rm.states[1];
  for (let column = 0; column < 3; column++) {
    // xi = mu - omega*delta*b_nu  =>  mu = xi + omega*delta*b_nu.
    const weightedMu = prior0 * (s0.xi[column] + s0.omega[column] * s0.delta[column] * exactBNu(MODEL.returnModel.nu))
      + (1 - prior0) * (s1.xi[column] + s1.omega[column] * s1.delta[column] * exactBNu(MODEL.returnModel.nu));
    const shift = targetMean[column] - weightedMu;
    s0.xi[column] += shift; s1.xi[column] += shift;
  }
  return rm;
}

function monthlyTax(gross, config) {
  const brackets = [0].concat(config.taxThresholdsAnnual.map(value => value / 12));
  let tax = 0;
  for (let index = 0; index < 5; index++) tax += Math.max(0, Math.min(gross, index < 4 ? brackets[index + 1] : gross) - brackets[index]) * config.taxRates[index];
  return tax;
}

function annualTax(gross, config) {
  const incomeTax = monthlyTax(gross / 12, config) * 12;
  const qppTier1 = Math.max(0, Math.min(gross, config.qppMaximumAnnual) - config.qppBasicAnnual) * config.qppRate;
  const qppTier2 = Math.max(0, Math.min(gross, config.qppMaximumMSGA || 81200) - config.qppMaximumAnnual) * (config.qppRateTier2 || 0.04);
  const ei = Math.min(gross, config.eiMaximumAnnual) * config.eiRate;
  const rqap = Math.min(gross, config.rqapMaximumAnnual) * config.rqapRate;
  return incomeTax + qppTier1 + qppTier2 + ei + rqap;
}

function netTaxableIncome(gross, age, config, oas7074, oas75) {
  const oasEligible = age >= config.pensionStartAge && age >= 65;
  const oasMax = oasEligible ? (age >= 75 ? oas75 : oas7074) / 12 : 0;
  const clawback = oasMax > 0 ? Math.min(oasMax, Math.max(0, gross - config.oasClawbackThreshold / 12) * config.oasClawbackRate) : 0;
  return gross - monthlyTax(gross, config) - clawback;
}

function pensionAmounts(config) {
  const careerYears = config.retirementAge - config.careerStartAge;
  let salarySum = 0;
  let salary = config.startingSalary;
  for (let i = 0; i < careerYears; i++) {
    const age = config.careerStartAge + i;
    if (i > 0) {
      const phase = config.promotionPhases.find(item => age >= item.start && age < item.end);
      salary *= 1 + (phase ? phase.growth : 0);
    }
    salarySum += Math.min(1.0, salary / config.qppMaximumAnnual);
  }
  const qppEarningsRatio = careerYears > 0 ? (salarySum / careerYears) : 0;
  const baseQpp = config.maxQppAge65 * Math.min(1, careerYears / 40) * qppEarningsRatio;
  const effectiveQppAge = Math.min(72, Math.max(60, config.pensionStartAge));
  const qppMultiplier = effectiveQppAge >= 65
    ? 1 + (effectiveQppAge - 65) * config.qppDeferralAnnual
    : 1 - (65 - effectiveQppAge) * (config.qppEarlyPenaltyAnnual || 0.06);
  const qpp = baseQpp * qppMultiplier;

  const effectiveOasAge = Math.min(70, Math.max(65, config.pensionStartAge));
  const oasMultiplier = Math.min(config.oasDeferralCap, 1 + (effectiveOasAge - 65) * config.oasDeferralAnnual);
  const oas7074 = config.maxOasAge65 * oasMultiplier;
  return {cpp: qpp, oas7074, oas75: oas7074 * config.oas75Increase};
}

function uniqueSorted(values) { return Array.from(new Set(values.map(value => Number(value.toFixed(6))))).sort((a, b) => a - b); }

function buildDynamicModel(config) {
  const constants = {
    currentAge: config.currentAge, careerStartAge: config.careerStartAge, retirementAge: config.retirementAge,
    pensionStartAge: config.pensionStartAge, deathAge: config.deathAge,
    accumMonths: (config.retirementAge - config.currentAge) * 12,
    bridgeMonths: (config.pensionStartAge - config.retirementAge) * 12,
    retireMonths: (config.deathAge - config.retirementAge) * 12,
    totalMonths: (config.retirementAge - config.currentAge + config.deathAge - config.retirementAge) * 12,
    careerYears: config.retirementAge - config.careerStartAge, funds: C.funds,
    annualDistributionYield: config.annualDistributionYield, taxOnDistributions: config.taxOnDistributions,
    capitalGainsInclusion: config.capitalGainsInclusionRate, capitalGainsTaxRate: config.capitalGainsTaxRate,
    hisaMonthly: Math.pow(1 + config.hisaAnnualRealReturn, 1 / 12) - 1, cashWedgeFraction: config.cashWedgeFraction,
    glidepathDeclining: config.glidepathDeclining || C.glidepathDeclining, glidepathRising: config.glidepathRising || C.glidepathRising,
    meltdownMonthly: config.meltdownBracketAnnual / 12, oasThresholdMonthly: config.oasClawbackThreshold / 12, oasClawbackRate: config.oasClawbackRate,
    employerMatchRate: config.employerMatchRate, employerMatchPercent: config.employerMatchPercent,
    realBorrowRateAnnual: config.realBorrowRateAnnual, extraMer15: config.extraMer15, extraMer20: config.extraMer20,
    layoffAnnualProbability: config.layoffAnnualProbability,
    bisectionSteps: Number(MODEL.constants.bisectionSteps) || 24,
    m75Start: Math.round((75 - config.retirementAge) * 12), postWedgeMonth: Math.round((config.pensionStartAge - config.retirementAge) * 12),
    seed: MODEL.defaultSeed, skewDegreesFreedom: config.skewDegreesFreedom,
    targetHouseCapital: config.propertyValue * config.downPaymentFraction + config.closingCosts,
    mortgagePrincipal: config.propertyValue * (1 - config.downPaymentFraction),
    mortgageMonthlyRate: Math.pow(1 + config.realMortgageRateAnnual, 1 / 12) - 1,
    monthlyPropertyTaxesCondo: config.monthlyPropertyTaxesCondo,
    monthlyMarketRent: config.monthlyMarketRent,
    fhsaAnnualLimit: config.fhsaAnnualLimit,
    fhsaMaxBalance: config.fhsaMaxBalance,
    hbpMaxWithdrawal: config.hbpMaxWithdrawal,
    hbpRepaymentYears: config.hbpRepaymentYears,
    propertyValue: config.propertyValue,
    estateGridFractions: MODEL.constants.estateGridFractions || [],
  };
  const career = new Float32Array(constants.careerYears * 6);
  const salaries = new Array(constants.careerYears);
  let salary = config.startingSalary;
  for (let index = 0; index < constants.careerYears; index++) {
    const age = config.careerStartAge + index;
    if (index > 0) {
      const phase = config.promotionPhases.find(item => age >= item.start && age < item.end);
      salary *= 1 + (phase ? phase.growth : 0);
    }
    salaries[index] = salary;
  }
  const netSalaries = salaries.map(value => value - annualTax(value, config));
  const retirementStartRate = config.retirementSavingsStartAnnual / netSalaries[0];
  const houseStartRate = config.houseSavingsStartAnnual / netSalaries[0];
  let cumulativeRrsp = 0;
  for (let index = 0; index < constants.careerYears; index++) {
    const salaryTaxRate = config.taxThresholdsAnnual.findIndex(threshold => salaries[index] < threshold);
    const rate = salaryTaxRate < 0 ? config.taxRates[4] : config.taxRates[salaryTaxRate];
    const retirementRate = Math.min(retirementStartRate + index * config.retirementSavingsEscalationRate, config.savingsMaxFraction);
    const houseRate = Math.min(houseStartRate + index * config.houseSavingsEscalationRate, config.houseSavingsMaxFraction);
    const tfsaRoom = (config.careerStartAge - 18 + 1) * config.tfsaAnnualLimit - config.otherTfsa + index * config.tfsaAnnualLimit;
    career.set([retirementRate * netSalaries[index], houseRate * netSalaries[index], rate, tfsaRoom, cumulativeRrsp, salaries[index]], index * 6);
    cumulativeRrsp += Math.min(config.rrspContributionRate * salaries[index], config.rrspMaxContribution);
  }
  const pension = pensionAmounts(config);
  const grossPre = [0].concat(config.taxThresholdsAnnual.map(value => value / 12), [1_000_000 / 12]);
  const netPre = grossPre.map(value => netTaxableIncome(value, config.pensionStartAge - 1, config, pension.oas7074, pension.oas75));
  function postGrid(oas) {
    const oasMonthly = oas / 12, threshold = config.oasClawbackThreshold / 12;
    return uniqueSorted([0, config.taxThresholdsAnnual[0] / 12, config.taxThresholdsAnnual[1] / 12, threshold, config.taxThresholdsAnnual[2] / 12, threshold + oasMonthly / config.oasClawbackRate, config.taxThresholdsAnnual[3] / 12, 1_000_000 / 12]);
  }
  const gross70 = postGrid(pension.oas7074), gross75 = postGrid(pension.oas75);
  const net70 = gross70.map(value => netTaxableIncome(value, Math.max(70, config.pensionStartAge), config, pension.oas7074, pension.oas75));
  const net75 = gross75.map(value => netTaxableIncome(value, 75, config, pension.oas7074, pension.oas75));
  const taxValues = new Float32Array(54);
  taxValues.set(grossPre, 0); taxValues.set(netPre, 6); taxValues.set(gross70, 12); taxValues.set(net70, 20); taxValues.set(gross75, 28); taxValues.set(net75, 36);
  taxValues.set(config.taxThresholdsAnnual.map(value => value / 12), 44); taxValues.set(config.taxRates, 49);
  const month0 = new Float32Array(constants.retireMonths * 4), month1 = new Float32Array(constants.retireMonths * 4), smile = new Float32Array(constants.retireMonths);
  let smileCurrent = 1;
  for (let month = 0; month < constants.retireMonths; month++) {
    const age = config.retirementAge + month / 12;
    const smilePhase = config.smileSchedule.find(item => age >= item.start && age < item.end);
    smile[month] = smileCurrent; if (smilePhase && smilePhase.change !== 0) smileCurrent *= 1 + smilePhase.change / 12;
    const qppMonthly = (age >= config.pensionStartAge) ? (pension.cpp / 12) : 0;
    const oasEligible = age >= config.pensionStartAge && age >= 65;
    const oasMonthly = oasEligible ? ((age < 75 ? pension.oas7074 : pension.oas75) / 12) : 0;
    const grossPension = qppMonthly + oasMonthly;
    month0.set([
      smile[month] / 12,
      age < config.healthcareEndAge ? config.post50HealthcareAnnual / 12 : 0,
      grossPension,
      0
    ], month * 4);
    month0[month * 4 + 3] = grossPension > 0 ? netTaxableIncome(grossPension, age, config, pension.oas7074, pension.oas75) : 0;
    const wholeAge = Math.floor(age);
    const rrif = wholeAge < 71 ? 0 : wholeAge >= 95 ? 0.2 : Number(config.rrifFactors[String(wholeAge)] || 0.2);
    const oasMax = oasEligible ? ((age >= 75 ? pension.oas75 : pension.oas7074) / 12) : 0;
    month1.set([rrif, oasMax, 0, 0], month * 4);
  }
  const cpmWeights = new Float32Array(constants.retireMonths);
  const adjustedMortality = MODEL.mortalityAnnualProbability.map(value => Math.pow(value, config.mortalityReductionFactor));
  let weightSum = 0;
  for (let month = 0; month < constants.retireMonths; month++) {
    const position = month * (adjustedMortality.length - 1) / Math.max(1, constants.retireMonths - 1), low = Math.floor(position), high = Math.ceil(position);
    const survival = adjustedMortality[low] + (adjustedMortality[high] - adjustedMortality[low]) * (position - low);
    cpmWeights[month] = Math.pow(1 + config.discountRateAnnual / 12, -month) * survival; weightSum += cpmWeights[month];
  }
  for (let month = 0; month < constants.retireMonths; month++) cpmWeights[month] /= weightSum;
  // Two-state Markov switching return model: regime fit calibrated in Python
  // at build time and embedded in the payload (MODEL.returnModel). Append the
  // 36-word two-state skew-t block (2 x [xi, omega, delta, row-major
  // Cholesky]) plus the two transition probabilities p00/p11 after the tax
  // tail, then the 11 house constants (index 10 = target property value, read
  // only by the bequest estate pass) and the bequest estate-grid tail [grid
  // count, fractions...] (mirror of calibration.build_model_buffer /
  // bequest.wgsl estate_grid_offset()). The CMA toggle + CMA inputs retarget
  // the drift live via applyCmaMeanShift (a fresh copy each run — never
  // mutate the embedded payload).
  const rm = MODEL.returnModel;
  const returnModel = {
    kind: rm.kind || "two-state-markov",
    nu: rm.nu, observations: rm.observations,
    p00: rm.p00, p11: rm.p11, prior0: rm.prior0,
    states: [
      { xi: rm.states[0].xi.slice(), omega: rm.states[0].omega, delta: rm.states[0].delta, cholesky: rm.states[0].cholesky },
      { xi: rm.states[1].xi.slice(), omega: rm.states[1].omega, delta: rm.states[1].delta, cholesky: rm.states[1].cholesky }
    ]
  };
  applyCmaMeanShift(returnModel, config);
  const s0 = returnModel.states[0], s1 = returnModel.states[1];
  const returnModelArray = [].concat(s0.xi, s0.omega, s0.delta, s0.cholesky,
                                     s1.xi, s1.omega, s1.delta, s1.cholesky,
                                     [returnModel.p00, returnModel.p11]);
  const houseConstantsArray = [
    constants.targetHouseCapital, constants.mortgagePrincipal, constants.mortgageMonthlyRate,
    constants.monthlyPropertyTaxesCondo, constants.monthlyMarketRent,
    constants.fhsaAnnualLimit, constants.fhsaMaxBalance, constants.hbpMaxWithdrawal,
    constants.hbpRepaymentYears, C.houseCount, constants.propertyValue,
    constants.estateGridFractions.length, ...constants.estateGridFractions
  ];
  const staticValues = new Float32Array(career.length + month0.length + month1.length + taxValues.length + returnModelArray.length + houseConstantsArray.length);
  staticValues.set(career, 0); staticValues.set(month0, career.length); staticValues.set(month1, career.length + month0.length); staticValues.set(taxValues, career.length + month0.length + month1.length); staticValues.set(returnModelArray, career.length + month0.length + month1.length + taxValues.length); staticValues.set(houseConstantsArray, career.length + month0.length + month1.length + taxValues.length + returnModelArray.length);
  return {constants, career, month0, month1, taxValues, staticValues, smile, cpmWeights, returnModel, pension};
}

// ---------------------------------------------------------------------------
// RQMC (Sobol) sampler — the DEFAULT; ?sampler=threefry opts out to legacy.
// The page embeds the Joe-Kuo direction table truncated to its TOP 14 bits
// (bit-packed, base64); 14 bits cover simulation indices < 2^14 = 16,384
// (the UI caps runs at 10,000 paths per seed).  The WGSL load-shifts each
// stored word back to its full 32-bit position (<< (32 - bits)), which
// reconstructs the exact direction number for every k <= 14, so the
// browser's stream is byte-identical to the Python engine's 20-bit
// digital-shift Sobol path.  The default is ALWAYS Threefry; ?sampler=rqmc
// is the only way to switch.  Non-default ages/horizons (coordinates
// beyond the embedded table) and runs >= 2^14 paths fall back to Threefry
// with a console warning.
// ---------------------------------------------------------------------------
const SOBOL_RQMC_BITS = 14;
const SOBOL_TABLE_DIMS = __SOBOL_TABLE_DIMS__;
const SOBOL_TABLE_B64 = "__SOBOL_TABLE_B64__";
function sobolTableAvailable() { return SOBOL_TABLE_B64.length > 8 && SOBOL_TABLE_DIMS > 0; }
let _sobolTableWords = null;
function sobolTableWords() {
  if (_sobolTableWords) return _sobolTableWords;
  const raw = Uint8Array.from(atob(SOBOL_TABLE_B64), c => c.charCodeAt(0));
  const total = (raw.length * 8 / SOBOL_RQMC_BITS) | 0;
  const words = new Uint32Array(total);
  let bitPos = 0;
  for (let i = 0; i < total; i++, bitPos += SOBOL_RQMC_BITS) {
    const byteIdx = bitPos >> 3, shift = bitPos & 7;
    let val = raw[byteIdx] >>> shift;
    if (shift + SOBOL_RQMC_BITS > 8) val |= raw[byteIdx + 1] << (8 - shift);
    if (shift + SOBOL_RQMC_BITS > 16) val |= raw[byteIdx + 2] << (16 - shift);
    words[i] = val & ((1 << SOBOL_RQMC_BITS) - 1);
  }
  _sobolTableWords = words;
  return words;
}
function rqmcSamplerEnabled(dynamic, simulations) {
  // RQMC digital-shift Sobol is the DEFAULT (lower single-seed error at low
  // simulation counts, same converged CEQ); ?sampler=threefry opts out to
  // the legacy counter-based Threefry stream.
  if (!sobolTableAvailable()) { logWarn("RQMC sampler unavailable: the build was made without the Sobol table (sobol_dirs_u32.npy / scipy missing at build time). Continuing with Threefry."); return false; }
  const url = new URLSearchParams(location.search);
  if (url.get("sampler") === "threefry") return false;
  const dims = dynamic.constants.totalMonths * 10 + dynamic.constants.careerYears;
  if (dims > SOBOL_TABLE_DIMS) {
    logWarn("RQMC disabled: this configuration needs", dims, "Sobol coordinates but the embedded table only has", SOBOL_TABLE_DIMS, "(non-default ages/horizons). Continuing with Threefry.");
    return false;
  }
  if (simulations >= (1 << SOBOL_RQMC_BITS)) {
    logWarn("RQMC disabled: the run needs", simulations, "paths but the embedded 14-bit table covers up to", (1 << SOBOL_RQMC_BITS) - 1, ". Continuing with Threefry.");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// WebGPU pipeline
// ---------------------------------------------------------------------------
function makeParams(dynamic, simulations, allocations, batchSims, simOffset, columnsPerWorkgroup = 1, generateLeveraged = true, rqmcBits = 0) {
  const dimensions = dynamic.constants;
  const buffer = new ArrayBuffer(160);
  new Uint32Array(buffer, 0, 4).set([simulations, allocations, dimensions.totalMonths, dimensions.accumMonths]);
  new Uint32Array(buffer, 16, 4).set([dimensions.retireMonths, RETURN_FUND_COUNT, dimensions.bisectionSteps, dimensions.m75Start]);
  new Uint32Array(buffer, 32, 4).set([dimensions.postWedgeMonth, dimensions.currentAge, dimensions.careerStartAge, dimensions.retirementAge]);
  new Float32Array(buffer, 48, 4).set([dimensions.annualDistributionYield / 12, dimensions.taxOnDistributions, dimensions.hisaMonthly, dimensions.capitalGainsInclusion]);
  // constants1.y: the cash-wedge FRACTION of the retirement span — the
  // shaders multiply it by the retirement month count (solver.x).
  new Float32Array(buffer, 64, 4).set([dimensions.capitalGainsTaxRate, dimensions.cashWedgeFraction, dimensions.meltdownMonthly, dimensions.oasThresholdMonthly]);
  new Float32Array(buffer, 80, 4).set([dimensions.oasClawbackRate, dimensions.employerMatchRate, dimensions.employerMatchPercent, dimensions.funds.length]);
  new Uint32Array(buffer, 96, 4).set([dimensions.seed, dimensions.skewDegreesFreedom, batchSims, simOffset]);
  new Float32Array(buffer, 112, 4).set([dimensions.realBorrowRateAnnual / 12, dimensions.extraMer15 / 12, dimensions.extraMer20 / 12, dimensions.layoffAnnualProbability]);
  // dispatch.y: 1 under ?sampler=rqmc (digital-shift Sobol), else Threefry.
  // dispatch.z: 1 = generate the leveraged return series (VEQT1.5/VEQT2),
  // 0 = skip them (leverage-off runs never read funds 1/2).
  // dispatch.w: RQMC direction-bit width (14); 0 = RQMC off.
  new Uint32Array(buffer, 128, 4).set([columnsPerWorkgroup, rqmcBits > 0 ? 1 : 0, generateLeveraged ? 1 : 0, rqmcBits]);
  // glide: DECLINING .xy = (VEQT share, VGRO share), RISING .zw = (VBAL
  // share, VGRO share); the third fund takes the remainder. One schedule for
  // every glidepath phase (accumulation, bridge, post-pension).
  new Float32Array(buffer, 144, 4).set([dimensions.glidepathDeclining[0], dimensions.glidepathDeclining[1], dimensions.glidepathRising[0], dimensions.glidepathRising[1]]);
  return buffer;
}

function staticBuffer(device, data) {
  const size = Math.max(4, data.byteLength);
  const buffer = device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function createDeviceContext() {
  if (state.deviceContext) return state.deviceContext;
  if (state.devicePromise) return state.devicePromise;
  const promise = (async () => {
    if (!navigator.gpu) {
      const message = "WebGPU is unavailable in this browser. Use a current Chrome or Edge (113+) with hardware acceleration enabled (chrome://settings/system), or launch with --enable-unsafe-webgpu.";
      logError(message);
      await dumpDiagnostics();
      throw new Error(message);
    }
    logInfo("Requesting WebGPU adapter (falling back across power preferences)...");
    let adapter = null;
    let lastAdapterError = null;
    for (const powerPreference of ["high-performance", "low-power", undefined]) {
      try {
        const options = powerPreference ? {powerPreference} : undefined;
        logDebug("navigator.gpu.requestAdapter(", powerPreference || "default", ")");
        adapter = await navigator.gpu.requestAdapter(options);
        if (adapter) {
          logInfo("Adapter found with powerPreference =", powerPreference || "default");
          break;
        }
      } catch (err) {
        lastAdapterError = err;
        logWarn("requestAdapter(", powerPreference || "default", ") threw:", err);
      }
    }
    if (!adapter) {
      const message = "No WebGPU adapter found. Possible causes: disabled hardware acceleration, an outdated or failing GPU driver, or WebGPU blocked by a browser flag. See the diagnostics above and try chrome://gpu.";
      logError(message, lastAdapterError || "");
      await dumpDiagnostics();
      throw new Error(message);
    }
    const info = adapter.info || {};
    state.adapterText = info.device || info.description || info.vendor || "high-performance adapter";
    logInfo("Adapter selected:", state.adapterText, JSON.stringify(info));
    setText("adapter-meta", state.adapterText);
    logDebug("Adapter limits:", {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension
    });
    // Large runs (30k paths x 5,040 strategies) need storage buffers above
    // Chrome's default caps, so request the adapter's own maxima.
    let device;
    try {
      device = await adapter.requestDevice({
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
        }
      });
    } catch (err) {
      logError("adapter.requestDevice() failed:", err);
      throw err;
    }
    // Surface GPU-side failures instead of letting them fail silently.
    device.addEventListener("uncapturederror", (event) => {
      const error = event.error || {};
      logError("GPU uncaptured error:", error.message, error.stack ? "\n" + error.stack : "");
    });
    device.lost.then((lostInfo) => {
      state.deviceLostReason = lostInfo.reason + (lostInfo.message ? " (" + lostInfo.message + ")" : "");
      logError("WebGPU device lost:", state.deviceLostReason);
      setStatus("WebGPU device lost: " + state.deviceLostReason, true);
      setText("run-message", "The GPU device was lost (" + state.deviceLostReason + "). Reload the page; if this repeats, check your GPU driver.");
    });
    const visibility = GPUShaderStage.COMPUTE;
    // Main module: seven storage bindings (0-6). The Chrome D3D12 backend on
    // AMD drivers silently drops dispatch writes beyond two read-write storage
    // buffers per stage, so returns/layoffs/states share one packed scratch
    // buffer (binding 1) and spending is the only other read-write binding (6).
    let shader;
    try {
      shader = device.createShaderModule({code: SHADER_SOURCE});
    } catch (err) {
      logError("createShaderModule failed for the main pipeline:", err);
      throw err;
    }
    const layout = device.createBindGroupLayout({entries: Array.from({length: 7}, (_, binding) => ({
      binding, visibility, buffer: {type: binding === 1 || binding === 6 ? "storage" : "read-only-storage"}
    }))});
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [layout]});
    const generateReturns = device.createComputePipeline({layout: pipelineLayout, compute: {module: shader, entryPoint: "generate_returns"}});
    const generateLayoffs = device.createComputePipeline({layout: pipelineLayout, compute: {module: shader, entryPoint: "generate_layoffs"}});
    const accumulate = device.createComputePipeline({layout: pipelineLayout, compute: {module: shader, entryPoint: "accumulate"}});
    const solve = device.createComputePipeline({layout: pipelineLayout, compute: {module: shader, entryPoint: "solve"}});
    const trackDrawdowns = device.createComputePipeline({layout: pipelineLayout, compute: {module: shader, entryPoint: "track_drawdowns"}});
    // Quantile module: separate 3-binding layout with a single read-write
    // binding.
    let quantilesShader;
    try {
      quantilesShader = device.createShaderModule({code: QUANTILES_SHADER_SOURCE});
    } catch (err) {
      logError("createShaderModule failed for the quantile pipeline:", err);
      throw err;
    }
    const quantileLayout = device.createBindGroupLayout({entries: [
      {binding: 0, visibility, buffer: {type: "read-only-storage"}},
      {binding: 1, visibility, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility, buffer: {type: "storage"}}
    ]});
    const quantilePipelineLayout = device.createPipelineLayout({bindGroupLayouts: [quantileLayout]});
    const quantiles = device.createComputePipeline({layout: quantilePipelineLayout, compute: {module: quantilesShader, entryPoint: "quantiles"}});
    // Bequest module: separate 7-binding layout (0-4 read-only, 5-6 the
    // read-write estate ladder output and the persistent accumulation
    // histogram — two read-write bindings, within the AMD/D3D12 limit; the
    // per-simulation inputs are packed into one sim_data buffer to stay
    // within the max-8-storage-buffers-per-stage limit). Three entry points
    // mirror the solver's batching so no single dispatch outlives the
    // Windows TDR watchdog: bequest_reset zeroes the persistent histogram,
    // bequest_walk accumulates one batch of lives per dispatch,
    // bequest_final reduces the histogram to the 201-point ladders.
    let bequestShader;
    try {
      bequestShader = device.createShaderModule({code: BEQUEST_SHADER_SOURCE});
    } catch (err) {
      logError("createShaderModule failed for the bequest pipeline:", err);
      throw err;
    }
    const bequestLayout = device.createBindGroupLayout({entries: Array.from({length: 7}, (_, binding) => ({
      binding, visibility, buffer: {type: binding === 5 || binding === 6 ? "storage" : "read-only-storage"}
    }))});
    const bequestPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bequestLayout]});
    const bequestReset = device.createComputePipeline({layout: bequestPipelineLayout, compute: {module: bequestShader, entryPoint: "bequest_reset"}});
    const bequestWalk = device.createComputePipeline({layout: bequestPipelineLayout, compute: {module: bequestShader, entryPoint: "bequest_walk"}});
    const bequestFinal = device.createComputePipeline({layout: bequestPipelineLayout, compute: {module: bequestShader, entryPoint: "bequest_final"}});
    const context = {device, layout, quantileLayout, generateReturns, generateLayoffs, accumulate, solve, trackDrawdowns, quantiles, bequestLayout, bequestReset, bequestWalk, bequestFinal, limits: device.limits};
    state.deviceContext = context;
    setStatus("WebGPU ready: " + state.adapterText, false);
    logInfo("WebGPU device context ready on", state.adapterText);
    return context;
  })();
  state.devicePromise = promise;
  try { return await promise; }
  catch (err) {
    logError("createDeviceContext failed:", err);
    throw err;
  }
  finally { if (state.devicePromise === promise) state.devicePromise = null; }
}

function leverageEnabled() {
  const toggle = byId("chk-leverage");
  return !toggle || toggle.checked;
}

function allocationPool(leverage) {
  // Source indices into ALLOCATION_NAMES/METADATA for a run. With leverage
  // ON the pool is every strategy (5,040); OFF it drops every strategy whose
  // accumulation, bridge or post phase is VEQT1.5 or VEQT2 (1,600 remain).
  if (leverage) return Array.from({length: TOTAL_ALLOCATIONS}, (_, i) => i);
  const pool = [];
  for (let i = 0; i < TOTAL_ALLOCATIONS; i++) {
    const codes = ALLOCATION_METADATA.subarray(i * 4, i * 4 + 3);
    if (LEVERAGED_FUND_CODES.includes(codes[0]) || LEVERAGED_FUND_CODES.includes(codes[1]) || LEVERAGED_FUND_CODES.includes(codes[2])) continue;
    pool.push(i);
  }
  return pool;
}

function effectiveAllocationCount(leverage) {
  // The ?allocations=N URL cap applies to the leverage-filtered pool.
  return Math.min(RUN_ALLOCATION_COUNT, allocationPool(leverage).length);
}

function selectedAllocationIndices(count, leverage) {
  const pool = allocationPool(leverage);
  if (count === pool.length) return pool;
  // count - 1 is the stride divisor below, so a single-allocation cap must be
  // handled explicitly (pool[NaN] would silently zero the metadata row).
  if (count <= 1) return [pool[0]];
  return Array.from({length: count}, (_, i) => pool[Math.floor(i * (pool.length - 1) / (count - 1))]);
}

function glidepathBoundaries(code, months, constants) {
  // Mirrors calibration._glidepath_boundaries exactly: the first two legs of
  // the glidepath take their share of `months` (rounded to nearest month,
  // remaining months reserved for the last leg). The shares come from
  // dynamic.constants (the editable settings), one schedule for every
  // glidepath phase.
  const dec = constants.glidepathDeclining, ris = constants.glidepathRising;
  const roundMonths = share => Math.round(Math.max(0, Math.min(1, share)) * months);
  if (code === 5) {
    const first = roundMonths(dec[0]);
    return [first, first + roundMonths(dec[1])];
  }
  if (code === 6) {
    const first = roundMonths(ris[0]);
    return [first, first + roundMonths(ris[1])];
  }
  return [0, 0];
}

function selectedAllocationBuffer(count, constants, leverage) {
  const indices = selectedAllocationIndices(count, leverage);
  const data = new Uint32Array(count * 12);
  for (let i = 0; i < count; i++) {
    const metadata = ALLOCATION_METADATA.subarray(indices[i] * 4, indices[i] * 4 + 4);
    const offset = i * 12;
    data.set(metadata, offset);
    data.set(glidepathBoundaries(metadata[0], constants.accumMonths, constants), offset + 4);
    data.set(glidepathBoundaries(metadata[1], constants.bridgeMonths, constants), offset + 6);
    data.set(glidepathBoundaries(metadata[2], constants.retireMonths - constants.bridgeMonths, constants), offset + 8);
  }
  return {indices, data};
}

function setProgress(done, total, detail) {
  byId("progress-fill").style.width = (total ? (100 * done / total) : 0) + "%";
  setText("progress-detail", detail);
}

async function simulate(settings, run) {
  const context = await createDeviceContext();
  const dynamic = buildDynamicModel(settings.model);
  const device = context.device;
  const batchSize = DEFAULTS.batchSize;
  const leverage = leverageEnabled();
  const allocationCount = effectiveAllocationCount(leverage);
  // Dispatch shaping: each solve/track_drawdowns thread serially walks
  // `columnsPerWorkgroup` allocation columns (stride = dispatch_y), keeping
  // the grid ONE dispatch per pass - splitting the allocation space into
  // multiple dispatches silently corrupts results on some AMD D3D12 drivers.
  // The shaders mirror the stride math, so results are byte-identical for
  // any column count. Measured (batch 250): at the full 5,040-strategy space
  // the solve dispatch is throughput-bound and takes ~0.18 s regardless of
  // the column count, while at small spaces (e.g. leverage off = 1,600)
  // fewer columns give strictly shorter dispatches (61 ms at 1 column vs
  // 121 ms at 16). The configured default is therefore 1 column per thread
  // (maximum parallelism, shortest dispatches, most TDR headroom).
  const columnsPerWorkgroup = Math.max(1, DEFAULTS.columnsPerWorkgroup | 0);
  const dispatchAllocations = Math.max(1, Math.ceil(allocationCount / columnsPerWorkgroup));
  const totalSims = settings.simulations;
  const rqmcBits = rqmcSamplerEnabled(dynamic, totalSims) ? SOBOL_RQMC_BITS : 0;
  logInfo("Simulation start:", {simulations: totalSims, allocations: allocationCount, leverage, batchSize,
          columnsPerWorkgroup, dispatchAllocations,
          totalMonths: dynamic.constants.totalMonths, careerYears: dynamic.constants.careerYears,
          pathCount: dynamic.constants.funds.length, houseCount: C.houseCount,
          retirementAge: settings.model.retirementAge, pensionStartAge: settings.model.pensionStartAge,
          sampler: rqmcBits ? "rqmc-sobol (default)" : "threefry (legacy)"});
  // Preflight: the spending and drawdown buffers are both
  // allocationCount x totalSims x 4 bytes; a run that needs more than the
  // GPU's storage-buffer binding limit would fail with an opaque WebGPU
  // validation error (silently dropped dispatches -> zero results).
  const neededBytes = allocationCount * totalSims * 4;
  const bufferCap = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (neededBytes > bufferCap) {
    throw new Error(
      "This run needs " + Math.round(neededBytes / 1048576) + " MB per result buffer, but the GPU caps storage buffers at " +
      Math.round(bufferCap / 1048576) + " MB. Reduce the simulation count (or load with fewer allocations, e.g. ?allocations=1000)."
    );
  }
  const pathCount = dynamic.constants.funds.length;
  const totalMonths = dynamic.constants.totalMonths;
  const careerYears = dynamic.constants.careerYears;
  const houseCount = C.houseCount;
  // The estate grid is a config constant (the fractions live in the model
  // buffer tail), independent of (theta, k).
  const estateGrid = Math.max(1, (dynamic.constants.estateGridFractions || []).length);
  // Packed global per-simulation data for the bequest pass (returns, states,
  // house outcomes) - sized here so the buffer log below can reference it.
  const simDataWords = totalSims * totalMonths * RETURN_FUND_COUNT + houseCount * pathCount * totalSims * 4 + houseCount * totalSims * 2;
  const selected = selectedAllocationBuffer(allocationCount, dynamic.constants, leverage);
  const allocationBuffer = staticBuffer(device, selected.data);
  const modelBuffer = staticBuffer(device, dynamic.staticValues);
  const scratchSize = (batchSize * totalMonths * RETURN_FUND_COUNT + batchSize * careerYears + houseCount * pathCount * batchSize * 4 + houseCount * batchSize * 2 + batchSize * allocationCount + pathCount * batchSize) * 4;
  logDebug("GPU buffers (MB):", {
    scratch: (scratchSize / 1048576).toFixed(1),
    spending: (allocationCount * totalSims * 4 / 1048576).toFixed(1),
    drawdownReadback: (allocationCount * totalSims * 4 / 1048576).toFixed(1),
    quantileOutput: (allocationCount * 201 * 4 / 1048576).toFixed(1),
    estateOutput: (allocationCount * estateGrid * 201 * 4 / 1048576).toFixed(1),
    estateHist: (allocationCount * estateGrid * 514 * 4 / 1048576).toFixed(1),
    estateInputs: (simDataWords * 4 / 1048576).toFixed(1),
    limit: (Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize) / 1048576).toFixed(0)
  });
  const scratchBuffer = device.createBuffer({size: scratchSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const spendingBuffer = device.createBuffer({size: allocationCount * totalSims * 4, usage: GPUBufferUsage.STORAGE});
  const quantileBuffer = device.createBuffer({size: allocationCount * 201 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const quantileReadback = device.createBuffer({size: allocationCount * 201 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const drawdownReadback = device.createBuffer({size: allocationCount * totalSims * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const houseReadback = device.createBuffer({size: houseCount * batchSize * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const houseOutcomes = {bought: [], buyMonth: []};
  for (let h = 0; h < houseCount; h++) {
    houseOutcomes.bought.push(new Uint8Array(totalSims));
    houseOutcomes.buyMonth.push(new Float32Array(totalSims));
  }
  const dummyA = device.createBuffer({size: 4, usage: GPUBufferUsage.STORAGE});
  const dummyB = device.createBuffer({size: 4, usage: GPUBufferUsage.STORAGE});
  // Binding 4: the truncated Sobol direction table under ?sampler=rqmc (the
  // shader only reads it when dispatch.y != 0); the 4-byte dummy otherwise.
  const sobolBuffer = rqmcBits ? staticBuffer(device, sobolTableWords()) : null;
  const batchBindings = rqmcBits
    ? [scratchBuffer, allocationBuffer, modelBuffer, sobolBuffer, dummyB, spendingBuffer]
    : [scratchBuffer, allocationBuffer, modelBuffer, dummyA, dummyB, spendingBuffer];
  // Global packed copy of the per-batch scratch data for the terminal-estate
  // (bequest) pass: monthly returns, retirement states and house outcomes,
  // all indexed by the GLOBAL simulation id, in one buffer (mirrors the main
  // module's scratch packing; keeps the bequest module within the max-8-
  // storage-buffers-per-stage limit).
  const simDataBuffer = device.createBuffer({size: simDataWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  const estateBuffer = device.createBuffer({size: allocationCount * estateGrid * 201 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const estateReadback = device.createBuffer({size: allocationCount * estateGrid * 201 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  // Persistent per-(allocation, grid) accumulation histogram: 512 fixed
  // log2(estate+1) bins + exact min/max words. Zeroed once by bequest_reset,
  // folded batch by batch by bequest_walk, reduced by bequest_final. The
  // walk is batched exactly like the solver so no single dispatch outlives
  // the Windows TDR watchdog.
  const estateHistBuffer = device.createBuffer({size: allocationCount * estateGrid * 514 * 4, usage: GPUBufferUsage.STORAGE});
  const totalBatches = Math.ceil(totalSims / batchSize);
  let offset = 0;
  try {
    for (let batchNumber = 0; batchNumber < totalBatches; batchNumber++) {
      if (run.cancelled) throw new Error("__CANCELLED__");
      const count = Math.min(batchSize, totalSims - offset);
      const batchStarted = performance.now();
      logDebug("Batch", (batchNumber + 1) + "/" + totalBatches, "sims", offset, "..", offset + count - 1);
      const params = device.createBuffer({size: 160, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(params, 0, makeParams(dynamic, totalSims, allocationCount, count, offset, columnsPerWorkgroup, leverage, rqmcBits));
      const bindGroup = device.createBindGroup({layout: context.layout, entries: [params, ...batchBindings].map((buffer, binding) => ({binding, resource: {buffer}}))});
      const bequestBindGroup = device.createBindGroup({layout: context.bequestLayout, entries: [params, spendingBuffer, simDataBuffer, allocationBuffer, modelBuffer, estateBuffer, estateHistBuffer].map((buffer, binding) => ({binding, resource: {buffer}}))});
      let encoder = device.createCommandEncoder();
      // Zero the persistent bequest histograms before the first batch.
      if (batchNumber === 0) {
        let resetPass = encoder.beginComputePass();
        resetPass.setPipeline(context.bequestReset); resetPass.setBindGroup(0, bequestBindGroup);
        resetPass.dispatchWorkgroups(allocationCount * estateGrid, 1, 1); resetPass.end();
      }
      let pass = encoder.beginComputePass();
      pass.setPipeline(context.generateReturns); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count * totalMonths / 64), 1, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.generateLayoffs); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count * careerYears / 64), 1, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.accumulate); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64), pathCount * houseCount, 1); pass.end();
      // solve and track_drawdowns cover all allocations via the shader-side
      // k-loop (params.dispatch.x = columnsPerWorkgroup); one dispatch each.
      pass = encoder.beginComputePass();
      pass.setPipeline(context.solve); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64), dispatchAllocations, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.trackDrawdowns); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64), dispatchAllocations, 1); pass.end();
      // Persist this batch's returns / states / houses for the bequest estate
      // pass (scratch is overwritten by the next batch). The scratch layout
      // matches common.wgsl: returns [0, RET), states after layoffs, houses
      // after states — all batch-local, hence the block copies into the
      // global, totalSims-strided buffers.
      encoder.copyBufferToBuffer(scratchBuffer, 0, simDataBuffer, offset * totalMonths * RETURN_FUND_COUNT * 4, count * totalMonths * RETURN_FUND_COUNT * 4);
      const statesOffset = count * totalMonths * RETURN_FUND_COUNT + count * careerYears;
      const housesOffset = statesOffset + houseCount * pathCount * count * 4;
      const globalStatesOffset = totalSims * totalMonths * RETURN_FUND_COUNT;
      for (let block = 0; block < houseCount * pathCount; block++) {
        encoder.copyBufferToBuffer(scratchBuffer, (statesOffset + block * count * 4) * 4, simDataBuffer, (globalStatesOffset + block * totalSims * 4 + offset * 4) * 4, count * 16);
      }
      for (let h = 0; h < houseCount; h++) {
        encoder.copyBufferToBuffer(scratchBuffer, (housesOffset + h * count * 2) * 4, simDataBuffer, (globalStatesOffset + houseCount * pathCount * totalSims * 4 + h * totalSims * 2 + offset * 2) * 4, count * 8);
      }
      // Bequest walk for THIS batch only (w = f * w*, estates folded into the
      // persistent histogram): bounded per-dispatch cost like the solver, so
      // the Windows TDR watchdog is never hit even at 10k paths.
      let walkPass = encoder.beginComputePass();
      walkPass.setPipeline(context.bequestWalk); walkPass.setBindGroup(0, bequestBindGroup);
      walkPass.dispatchWorkgroups(allocationCount * estateGrid, 1, 1); walkPass.end();
      const batchDrawdownOffset = count * totalMonths * RETURN_FUND_COUNT + count * careerYears + houseCount * pathCount * count * 4 + houseCount * count * 2;
      encoder.copyBufferToBuffer(scratchBuffer, batchDrawdownOffset * 4, drawdownReadback, offset * allocationCount * 4, count * allocationCount * 4);
      const batchHouseOffset = count * totalMonths * RETURN_FUND_COUNT + count * careerYears + houseCount * pathCount * count * 4;
      encoder.copyBufferToBuffer(scratchBuffer, batchHouseOffset * 4, houseReadback, 0, houseCount * count * 2 * 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      params.destroy();
      await houseReadback.mapAsync(GPUMapMode.READ);
      const houseData = new Float32Array(houseReadback.getMappedRange()).slice();
      houseReadback.unmap();
      for (let h = 0; h < houseCount; h++) {
        for (let s = 0; s < count; s++) {
          const i = (h * count + s) * 2;
          houseOutcomes.bought[h][offset + s] = houseData[i] > 0.5 ? 1 : 0;
          houseOutcomes.buyMonth[h][offset + s] = houseData[i + 1];
        }
      }
      offset += count;
      logDebug("Batch", (batchNumber + 1), "done in", (performance.now() - batchStarted).toFixed(0), "ms");
      setProgress(90 * offset / totalSims, 100, "Batch " + (batchNumber + 1) + "/" + totalBatches + " complete");
      // Yield so the progress bar can paint, but NEVER pace the pipeline by
      // the frame cost of the leaderboard DOM: after the first run the
      // 1,600-row table repaints every frame, and a plain
      // requestAnimationFrame await then resolves only after that paint,
      // stretching each batch by its paint time (~2x slower runs after the
      // first). Racing the frame against a fixed timeout keeps the vsync
      // cadence when frames are cheap and caps the wait at BATCH_YIELD_MS
      // when they are not.
      await new Promise(resolve => {
        const timeout = setTimeout(resolve, BATCH_YIELD_MS);
        requestAnimationFrame(() => { clearTimeout(timeout); resolve(); });
      });
    }
    if (run.cancelled) throw new Error("__CANCELLED__");
    setProgress(95, 100, "Computing quantiles on GPU...");
    logDebug("Quantile reduction: dispatching", allocationCount, "workgroups over", totalSims, "paths each");
    const quantileParams = device.createBuffer({size: 160, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(quantileParams, 0, makeParams(dynamic, totalSims, allocationCount, totalSims, 0, 1));
    const quantileBindGroup = device.createBindGroup({layout: context.quantileLayout, entries: [quantileParams, spendingBuffer, quantileBuffer].map((buffer, binding) => ({binding, resource: {buffer}}))});
    const encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(context.quantiles); pass.setBindGroup(0, quantileBindGroup);
    pass.dispatchWorkgroups(allocationCount, 1, 1); pass.end();
    encoder.copyBufferToBuffer(quantileBuffer, 0, quantileReadback, 0, allocationCount * 201 * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    quantileParams.destroy();
    await quantileReadback.mapAsync(GPUMapMode.READ);
    const quantileData = new Float32Array(quantileReadback.getMappedRange()).slice();
    quantileReadback.unmap();
    // Terminal-estate (bequest) ladders: every batch already folded its lives'
    // estates into the persistent histogram (bequest_walk); this tiny final
    // pass locates the 201 quantiles per (allocation, grid fraction). The
    // ladders are preference-independent — (theta, k) enter only the JS
    // re-rank later — so they are cached like the spending quantiles and
    // never touch the solver.
    setProgress(97, 100, "Computing terminal-estate ladders on GPU...");
    const bequestParams = device.createBuffer({size: 160, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(bequestParams, 0, makeParams(dynamic, totalSims, allocationCount, totalSims, 0, 1));
    const bequestFinalBindGroup = device.createBindGroup({layout: context.bequestLayout, entries: [bequestParams, spendingBuffer, simDataBuffer, allocationBuffer, modelBuffer, estateBuffer, estateHistBuffer].map((buffer, binding) => ({binding, resource: {buffer}}))});
    const bequestEncoder = device.createCommandEncoder();
    let bequestPass = bequestEncoder.beginComputePass();
    bequestPass.setPipeline(context.bequestFinal); bequestPass.setBindGroup(0, bequestFinalBindGroup);
    bequestPass.dispatchWorkgroups(allocationCount * estateGrid, 1, 1); bequestPass.end();
    bequestEncoder.copyBufferToBuffer(estateBuffer, 0, estateReadback, 0, allocationCount * estateGrid * 201 * 4);
    device.queue.submit([bequestEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    bequestParams.destroy();
    await estateReadback.mapAsync(GPUMapMode.READ);
    const estateData = new Float32Array(estateReadback.getMappedRange()).slice();
    estateReadback.unmap();
    await drawdownReadback.mapAsync(GPUMapMode.READ);
    const drawdownData = new Float32Array(drawdownReadback.getMappedRange());
    const uiMeans = new Array(allocationCount).fill(0);
    for (let sim = 0; sim < totalSims; sim++) {
      const base = sim * allocationCount;
      for (let index = 0; index < allocationCount; index++) uiMeans[index] += drawdownData[base + index];
    }
    drawdownReadback.unmap();
    for (let index = 0; index < allocationCount; index++) uiMeans[index] /= totalSims;
    const names = selected.indices.map(index => ALLOCATION_NAMES[index]);
    // Per-house purchase statistics: median buy age and median initial
    // mortgage payment over the paths where the house was actually bought.
    const houseStats = [];
    for (let h = 0; h < houseCount; h++) {
      const months = [];
      for (let s = 0; s < totalSims; s++) if (houseOutcomes.bought[h][s]) months.push(houseOutcomes.buyMonth[h][s]);
      if (months.length === 0) {
        houseStats.push({buyAge: null, p90BuyAge: null, mortgage: null});
        continue;
      }
      months.sort((a, b) => a - b);
      const monthQuantile = p => {
        const pos = (months.length - 1) * p / 100;
        const i = Math.floor(pos);
        const frac = pos - i;
        return months[i] + (i + 1 < months.length ? frac * (months[i + 1] - months[i]) : 0);
      };
      const medianMonth = monthQuantile(50);
      const p90Month = monthQuantile(90);
      const nMonths = Math.max(dynamic.constants.accumMonths - medianMonth, 1);
      const growth = Math.pow(1 + dynamic.constants.mortgageMonthlyRate, nMonths);
      const mortgage = dynamic.constants.mortgagePrincipal * dynamic.constants.mortgageMonthlyRate * growth / Math.max(growth - 1, 1e-12);
      houseStats.push({buyAge: dynamic.constants.currentAge + medianMonth / 12,
                       p90BuyAge: dynamic.constants.currentAge + p90Month / 12,
                       mortgage});
    }
    const results = [];
    for (let index = 0; index < names.length; index++) {
      const quantiles = new Array(201);
      for (let j = 0; j <= 200; j++) quantiles[j] = Math.round(quantileData[index * 201 + j] * 10) / 10;
      const parts = names[index].split("_");
      const houseIndex = Math.max(0, C.houses.indexOf(parts[0] + "_" + parts[1]));
      const hs = houseStats[houseIndex];
      const estate = new Array(estateGrid);
      for (let g = 0; g < estateGrid; g++) {
        const base = (index * estateGrid + g) * 201;
        estate[g] = new Float32Array(estateData.subarray(base, base + 201));
      }
      results.push({name: names[index], quantiles, median: quantiles[100], ui: uiMeans[index], estate,
                    buyAge: hs.buyAge, p90BuyAge: hs.p90BuyAge, mortgage: hs.mortgage});
    }
    return {results, names, dynamic, sampler: rqmcBits ? "rqmc" : "threefry"};
  } catch (error) {
    logError("WebGPU simulation failed (sims = " + totalSims + ", allocations = " + allocationCount + ", batch = " + batchSize + "):",
             error && error.message ? error.message : error, error && error.stack ? "\n" + error.stack : "");
    await dumpDiagnostics();
    throw error;
  } finally {
    allocationBuffer.destroy(); modelBuffer.destroy();
    scratchBuffer.destroy(); spendingBuffer.destroy();
    quantileBuffer.destroy(); quantileReadback.destroy();
    drawdownReadback.destroy(); houseReadback.destroy();
    simDataBuffer.destroy(); estateBuffer.destroy(); estateReadback.destroy(); estateHistBuffer.destroy();
    if (sobolBuffer) sobolBuffer.destroy();
    dummyA.destroy(); dummyB.destroy();
  }
}

// ---------------------------------------------------------------------------
// Post-processing: CE, lambda-adjusted re-ranking, filters & table.
// ---------------------------------------------------------------------------
function computeKappa(gamma, dynamic) {
  const exponent = 1 - gamma;
  if (Math.abs(exponent) < 1e-4) {
    let logSum = 0;
    for (let i = 0; i < dynamic.smile.length; i++) logSum += dynamic.cpmWeights[i] * Math.log(Math.max(dynamic.smile[i], 1e-6));
    return Math.exp(logSum);
  }
  let sum = 0;
  for (let i = 0; i < dynamic.smile.length; i++) sum += dynamic.cpmWeights[i] * Math.pow(Math.max(dynamic.smile[i], 1e-6), exponent);
  return Math.pow(sum, 1 / exponent);
}

function ceForQuantiles(quantiles, gamma, dynamic) {
  const exponent = 1 - gamma;
  let base;
  if (Math.abs(exponent) < 1e-4) {
    let logSum = 0;
    for (let i = 1; i < 200; i++) logSum += Math.log(Math.max(quantiles[i], 1e-6));
    base = Math.exp(logSum / 199);
  } else {
    let powerSum = 0;
    for (let i = 1; i < 200; i++) powerSum += Math.pow(Math.max(quantiles[i], 1e-6), exponent);
    base = Math.pow(powerSum / 199, 1 / exponent);
  }
  return base * computeKappa(gamma, dynamic);
}

// ---------------------------------------------------------------------------
// Bequest-adjusted CE. The cached estate ladders (preference-independent,
// computed once per run by bequest.wgsl) hold the 201-point quantile ladder
// of the tax-adjusted terminal estate at each spending fraction f of the
// solved w*. With the De Nardi-style bequest utility added to the CRRA
// consumption utility, the strategy's base CE is the best over the grid:
//     CE_beq(f) = [ mean_i ( (f * q_i * kappa)^(1-gamma)
//                           + theta_actual * (b_f,i + k)^(1-gamma) ) ]^(1/(1-gamma))
// The theta SLIDER is normalized so 0.5 always means parity, at any gamma:
//     theta_actual = 2 * parity * theta_slider
//     parity = (estate_ref / (retireMonths * spending_ref))^(gamma - 1)
// i.e. the intensity at which the bequest term equals the consumption term
// for a reference life (a bequestParityReferenceEstate estate vs.
// bequestParityReferenceSpending of monthly spending, both in the tool's
// monthly-equivalent units). 0 = no motive, 0.5 = parity, 1 = twice parity.
// The estate is valued in monthly-spending-equivalent units (lump sum spread
// over the retirement horizon), and the luxury threshold k is clamped to a
// positive floor when theta > 0: CRRA utility of a zero estate is -infinity
// for gamma > 1 otherwise (that is exactly the role of k in De Nardi 2004).
// For theta_slider = 0 the grid term collapses to f * CE_base, whose maximum
// sits exactly at f = 1, i.e. the unadjusted CE — the default (theta = k = 0)
// is a pure identity.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fine spending-fraction search. The GPU walks the estate grid (six fixed
// fractions of w*) once per run; the JS re-rank refines the choice between
// grid points by LINEARLY interpolating each of the 201 estate quantiles in
// f. The terminal estate is empirically near-affine in f (each 0.1 step of
// spending removes roughly the same wealth), so linear interpolation is
// accurate to a few percent, preserves quantile and f monotonicity, and
// cannot overshoot into negative estates near f = 1.
// ---------------------------------------------------------------------------
function estateLadderAt(strategy, f, dynamic) {
  const fractions = dynamic.constants.estateGridFractions || [];
  const estates = strategy.estate;
  if (!estates || estates.length === 0) return null;
  if (fractions.length < 2) return estates[0];
  const fMin = fractions[0], fMax = fractions[fractions.length - 1];
  const fClamped = Math.min(fMax, Math.max(fMin, f));
  if (fClamped <= fMin) return estates[0];
  if (fClamped >= fMax) return estates[estates.length - 1];
  let g = 0;
  while (g < fractions.length - 2 && fractions[g + 1] < fClamped) g++;
  const fLow = fractions[g], fHigh = fractions[g + 1];
  const t = (fClamped - fLow) / (fHigh - fLow);
  const lo = estates[g], hi = estates[g + 1];
  const ladder = new Float32Array(201);
  for (let i = 0; i < 201; i++) ladder[i] = lo[i] + t * (hi[i] - lo[i]);
  return ladder;
}

// Bequest-adjusted CE at an arbitrary spending fraction f (interpolated
// estate ladder; exact at the GPU grid fractions).
function bequestCeAtFraction(strategy, f, params) {
  const {gamma, thetaActual, kEff, kappa, months, logMode, exponent} = params;
  const q = strategy.quantiles;
  const ladder = estateLadderAt(strategy, f, params.dynamic);
  let sum = 0;
  for (let i = 1; i < 200; i++) {
    const c = Math.max(f * q[i] * kappa, 1e-6);
    const b = Math.max(ladder[i] / months + kEff, 1e-6);
    sum += logMode ? Math.log(c) + thetaActual * Math.log(b) : Math.pow(c, exponent) + thetaActual * Math.pow(b, exponent);
  }
  return logMode ? Math.exp(sum / 199) : Math.pow(sum / 199, 1 / exponent);
}

// Returns {ce, f}: the bequest-adjusted CE maximized over f in the grid
// range. An exact scan of the GPU grid fractions (no interpolation error)
// brackets the maximum, then a parabolic fit through the three points of the
// winning window refines f continuously (the CE is the sum of a consumption
// term falling in f and a bequest term rising in f, hence smooth and
// unimodal near the peak). The result is never worse than the grid scan.
function bequestBest(strategy, gamma, theta, k, dynamic) {
  const kappa = computeKappa(gamma, dynamic);
  const exponent = 1 - gamma;
  const logMode = Math.abs(exponent) < 1e-4;
  const fractions = dynamic.constants.estateGridFractions || [];
  const months = Math.max(1, dynamic.constants.retireMonths);
  const kEff = Math.max(k, theta > 0 ? (C.minBequestCurvature || 10000) : 0) / months;
  const parity = Math.pow(C.bequestParityReferenceEstate / (months * C.bequestParityReferenceSpending), gamma - 1);
  const thetaActual = 2 * parity * theta;
  const params = {gamma, thetaActual, kEff, kappa, months, logMode, exponent, dynamic};

  // Exact scan over the GPU grid fractions.
  let bestG = 0;
  let bestCe = -Infinity;
  const ces = new Array(fractions.length);
  for (let g = 0; g < fractions.length; g++) {
    const f = fractions[g] != null ? fractions[g] : 1;
    ces[g] = bequestCeAtFraction(strategy, f, params);
    if (ces[g] > bestCe) { bestCe = ces[g]; bestG = g; }
  }
  // Parabolic refinement through the winning window [g-1, g, g+1].
  const g0 = Math.max(0, bestG - 1), g2 = Math.min(fractions.length - 1, bestG + 1);
  if (g2 - g0 >= 2) {
    const y1 = ces[g0], y2 = ces[bestG], y3 = ces[g2];
    const denom = y1 - 2 * y2 + y3;
    if (Math.abs(denom) > 1e-12) {
      const h = fractions[g2] - fractions[g0];
      const fParabola = fractions[bestG] + 0.5 * h * (y1 - y3) / denom;
      if (fParabola >= fractions[g0] && fParabola <= fractions[g2]) {
        const ceParabola = bequestCeAtFraction(strategy, fParabola, params);
        if (ceParabola > bestCe) return {ce: ceParabola, f: fParabola};
      }
    }
  }
  return {ce: bestCe, f: fractions[bestG] != null ? fractions[bestG] : 1};
}

function bequestCeForStrategy(strategy, gamma, theta, k, dynamic) {
  if (!strategy.estate || strategy.estate.length === 0) return ceForQuantiles(strategy.quantiles, gamma, dynamic);
  return bequestBest(strategy, gamma, theta, k, dynamic).ce;
}

// The spending choice the bequest-adjusted CE implies for one strategy: the
// best fraction f of the max sustainable w* and the median estate at 95
// (ladder point P50) at that fraction.
function bequestChoiceForStrategy(strategy, gamma, theta, k, dynamic) {
  if (!strategy.estate || strategy.estate.length === 0) return null;
  const best = bequestBest(strategy, gamma, theta, k, dynamic);
  const ladder = estateLadderAt(strategy, best.f, dynamic);
  return {f: best.f, medianEstate: ladder ? ladder[100] : 0};
}

function lowerBound(values, target) {
  let low = 0, high = values.length;
  while (low < high) { const middle = (low + high) >> 1; if (values[middle] < target) low = middle + 1; else high = middle; }
  return low;
}

// Base CE depends only on (gamma, theta, k, quantiles, estate ladders,
// dynamic model), so the per-row CRRA evaluation is cached per control
// snapshot on the results array. The cache is naturally invalidated when a
// new run replaces state.results. Theta = 0 keeps the exact unadjusted path.
function ceBasesForGamma(gamma, theta, k, dynamic) {
  const cache = state.results._ceCache;
  if (cache && cache.gamma === gamma && cache.theta === theta && cache.k === k) return cache.values;
  const bequestOn = theta > 0 && !!(state.results[0] && state.results[0].estate && state.results[0].estate.length);
  const values = bequestOn
    ? state.results.map(s => bequestCeForStrategy(s, gamma, theta, k, dynamic))
    : state.results.map(s => ceForQuantiles(s.quantiles, gamma, dynamic));
  state.results._ceCache = {gamma, theta, k, values};
  return values;
}

function isGlidepath(value) {
  const base = value.replace("+CASH", "");
  return base === "DECLINING" || base === "RISING";
}

function captureAppliedControls() {
  const mix = document.querySelector("#seg-mix .seg-pill.active");
  return {
    gamma: Number(byId("slider-gamma").value),
    uiLambda: Number(byId("slider-lambda").value),
    theta: Number(byId("slider-theta").value) || 0,
    k: Number(byId("slider-k").value) || 0,
    floorP: DEFAULTS.floorPercentile,
    house: byId("filter-house").value,
    accum: byId("filter-accum").value,
    mix: mix ? mix.dataset.mix : "ALL",
    search: byId("table-search").value.trim().toUpperCase()
  };
}

function displayedRows() {
  if (!state.results) return [];
  const applied = state.applied || captureAppliedControls();
  const gamma = applied.gamma;
  const uiLambda = applied.uiLambda || 0;
  const theta = applied.theta || 0;
  const k = applied.k || 0;
  const floorP = applied.floorP;
  const dynamic = state.dynamic || buildDynamicModel(readModelInputs());
  const floorIndex = Math.round(floorP * 2);
  const ceBases = ceBasesForGamma(gamma, theta, k, dynamic);

  let rows = state.results.map((strategy, index) => {
    const q = strategy.quantiles;
    const parts = strategy.name.split("_");
    const house = parts[0] + "_" + parts[1];
    const hasCash = parts.some(part => part.includes("+CASH"));
    const hasGlidepath = parts.some(isGlidepath);
    const baseCe = ceBases[index];
    const ui = strategy.ui || 0;
    return {
      rank: index,
      name: strategy.name,
      parts,
      hasCash,
      hasGlidepath,
      house,
      accum: parts[2],
      bridge: parts[3],
      post: parts[4],
      ceBase: baseCe,
      ui,
      ce: uiLambda > 0 ? baseCe * Math.exp(-uiLambda * ui) : baseCe,
      median: strategy.median,
      floor: q[floorIndex],
      p1: q[2],
      p90: q[180],
      buyAge: strategy.buyAge,
      p90BuyAge: strategy.p90BuyAge,
      mortgage: strategy.mortgage
    };
  });

  const house = applied.house;
  if (house !== "ALL") rows = rows.filter(row => row.house === house);
  const accumulation = applied.accum;
  if (accumulation !== "ALL") rows = rows.filter(row => row.parts[2] === accumulation);
  const mix = applied.mix;
  if (mix === "NO_CASH") rows = rows.filter(row => !row.hasCash);
  if (mix === "NO_GLIDEPATH") rows = rows.filter(row => !row.hasGlidepath);
  if (mix === "ASSETS_ONLY") rows = rows.filter(row => !row.hasCash && !row.hasGlidepath);
  const search = applied.search;
  if (search) rows = rows.filter(row => row.name.toUpperCase().includes(search));
  // Empty result sets must short-circuit here: the sort branch below reads
  // rows[0] and would throw a TypeError on an undefined element.
  if (!rows.length) return rows;

  // Default view is the CE leaderboard (rank 1 = highest CE). A column sort
  // (state.sort.active) re-orders the visible sub-table: descending first,
  // then ascending, then back to the CE default. Missing values (renter buy
  // age / mortgage) always sink to the bottom, and ties fall back to the CE
  // rank so the ordering stays deterministic.
  const sort = state.sort;
  if (sort.active) {
    const column = sort.column;
    const direction = sort.ascending ? 1 : -1;
    const numeric = typeof rows[0][column] === "number";
    rows.sort((a, b) => {
      const left = a[column], right = b[column];
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      const cmp = numeric ? left - right : left < right ? -1 : left > right ? 1 : 0;
      return cmp * direction || b.ce - a.ce;
    });
  } else {
    rows.sort((a, b) => b.ce - a.ce);
  }
  return rows;
}

// Column-sort cycle: click a header to sort that column biggest -> smallest,
// click it again for smallest -> biggest, click a third time to return to the
// default CE leaderboard. Works on whatever rows the current filters show.
function cycleSort(column) {
  if (state.sort.active && state.sort.column === column) {
    if (state.sort.ascending) {
      state.sort = { column: "ce", ascending: false, active: false };
    } else {
      state.sort = { column, ascending: true, active: true };
    }
  } else {
    state.sort = { column, ascending: false, active: true };
  }
  updateSortIndicators();
  renderTable(true);
}

function updateSortIndicators() {
  document.querySelectorAll("th.sortable[data-sorted]").forEach(th => th.removeAttribute("data-sorted"));
  if (state.sort.active) {
    const th = document.querySelector('th.sortable[data-col="' + state.sort.column + '"]');
    if (th) th.setAttribute("data-sorted", state.sort.ascending ? "asc" : "desc");
  }
}

// Two-tone split chips: a saturated color slug (allocation/ratio) + white label.
const CHIP_META = {
  "VEQT": ["chip-veqt", "100/0", "VEQT"],
  "VEQT1.5": ["chip-veqt15", "x1.5", "VEQT 1.5"],
  "VEQT2": ["chip-veqt2", "x2.0", "VEQT 2.0"],
  "VGRO": ["chip-vgro", "80/20", "VGRO"],
  "VBAL": ["chip-vbal", "60/40", "VBAL"],
  "CASH": ["chip-cash", "HISA", "CASH"],
  "DECLINING": ["chip-glide", "\u2193", "DECLINING"],
  "RISING": ["chip-glide", "\u2191", "RISING"],
};

function renderBadge(str) {
  if (!str) return "";
  const isCash = str.includes("+CASH");
  const base = str.replace("+CASH", "");
  const meta = CHIP_META[base] || ["chip-veqt", "", base];
  let html = '<span class="badge"><span class="badge-chip ' + meta[0] + '">' + meta[1] + '</span><span class="badge-text">' + escapeHtml(meta[2]) + "</span></span>";
  if (isCash) html += ' <span class="badge"><span class="badge-chip chip-cash">HISA</span><span class="badge-text">+CASH</span></span>';
  return html;
}

function renderHouseBadge(house) {
  if (house === "HOUSE_NONE") return '<span class="badge"><span class="badge-chip chip-rent">RE</span><span class="badge-text">Renter</span></span>';
  const fund = house.replace("HOUSE_", "");
  const meta = CHIP_META[fund] || ["chip-veqt", "", fund];
  return '<span class="badge"><span class="badge-chip ' + meta[0] + '">' + meta[1] + '</span><span class="badge-text">Buyer (' + escapeHtml(fund) + ")</span></span>";
}

// ---------------------------------------------------------------------------
// Virtualized leaderboard rendering.
//
// Only the rows near the scroll position are in the DOM. Rendering all 5,040
// rich rows at once produced a ~60k px-tall compositor layer that kept the
// GPU process busy every frame; that work contended with the WebGPU compute
// submissions and made every run after the first ~2x slower (measured: 1,600
// rich rows visible = 0.84 s vs 0.31 s with the table hidden or windowed).
// Spacer rows preserve the scrollbar geometry.
// ---------------------------------------------------------------------------
let windowedRows = [];
let tableRowHeight = 0;

function rowHtml(row, index) {
  const selected = activeStrategy && row.name === activeStrategy.name ? ' class="selected-row"' : "";
  return "<tr" + selected + ' data-name="' + escapeHtml(row.name) + '">' +
    '<td class="mono" style="font-weight:700; color:var(--brand-navy)">' + (index + 1) + "</td>" +
    "<td>" + renderHouseBadge(row.house) + "</td>" +
    '<td><div class="badge-group">' + renderBadge(row.parts[2]) + "</div></td>" +
    '<td><div class="badge-group">' + renderBadge(row.parts[3]) + "</div></td>" +
    '<td><div class="badge-group">' + renderBadge(row.parts[4]) + "</div></td>" +
    '<td class="right mono" style="color:var(--text-muted)">' + (row.buyAge == null ? "—" : row.buyAge.toFixed(1)) + "</td>" +
    '<td class="right mono" style="color:var(--text-muted)">' + (row.p90BuyAge == null ? "—" : row.p90BuyAge.toFixed(1)) + "</td>" +
    '<td class="right mono" style="color:var(--text-muted)">' + (row.mortgage == null ? "—" : money(row.mortgage)) + "</td>" +
    '<td class="right mono" style="font-weight:700; color:var(--brand-green)">' + money(row.ce) + "/mo</td>" +
    '<td class="right mono">' + money(row.median) + "/mo</td>" +
    '<td class="right mono" style="color:' + uiSeverity(row.ui) + '">' + f2(row.ui) + "</td>" +
    '<td class="right mono" style="color:var(--brand-amber); font-weight:600">' + money(row.floor) + "/mo</td>" +
    '<td class="right mono">' + money(row.p90) + "/mo</td></tr>";
}

function spacerHtml(rows) {
  return rows > 0 ? '<tr class="v-spacer" aria-hidden="true" style="border:0"><td colspan="13" style="height:' + Math.round(rows * (tableRowHeight || 38)) + 'px; padding:0; border:0; line-height:0"></td></tr>' : "";
}

function paintWindow(rows) {
  const body = byId("table-body");
  const wrap = document.querySelector(".table-wrap");
  const rowH = tableRowHeight || 38;
  const viewport = wrap && wrap.clientHeight ? wrap.clientHeight : 540;
  const scrollTop = wrap ? wrap.scrollTop : 0;
  const total = rows.length;
  const top = Math.max(0, Math.floor(scrollTop / rowH) - 4);
  const bottom = Math.min(total, top + Math.ceil(viewport / rowH) + 8);
  let html = spacerHtml(top);
  for (let i = top; i < bottom; i++) html += rowHtml(rows[i], i);
  html += spacerHtml(total - bottom);
  body.innerHTML = html;
  // Rows have uniform markup, so one sample gives the exact row height and
  // keeps the spacers and the scroll-to-row mapping true.
  const sample = body.querySelector("tr[data-name]");
  if (sample) tableRowHeight = sample.getBoundingClientRect().height || tableRowHeight;
  body.querySelectorAll("tr[data-name]").forEach(tr => {
    tr.addEventListener("click", () => {
      const name = tr.dataset.name;
      const strategy = state.results.find(item => item.name === name);
      if (strategy) selectStrategy(strategy);
    });
  });
}

function renderTable(selectTop) {
  state.applied = captureAppliedControls();
  updateSortIndicators();
  const rows = displayedRows();
  const body = byId("table-body");
  const pill = document.querySelector('#seg-mix .seg-pill[data-mix="ALL"]');
  if (pill) pill.textContent = "All (" + (state.results ? state.results.length : effectiveAllocationCount(leverageEnabled())).toLocaleString("en-US") + ")";

  if (!rows.length) {
    windowedRows = [];
    body.innerHTML = '<tr><td colspan="13" style="text-align:center; padding:32px; color:var(--text-muted); font-family:var(--font-mono)">' +
      (state.results ? "No strategies match your criteria." : "No results yet — open Settings and press Simulate.") + "</td></tr>";
    if (activeStrategy) drawDistributionChart();
    return;
  }

  windowedRows = rows;
  // Windowed rendering (see the virtualization notes above); a fresh list
  // (filters, search, sort, re-rank) starts at the top of the scroll area.
  if (selectTop) {
    const wrap = document.querySelector(".table-wrap");
    if (wrap) wrap.scrollTop = 0;
  }
  const wasUnmeasured = !tableRowHeight;
  paintWindow(rows);
  if (rows.length && wasUnmeasured) paintWindow(rows);  // one correction pass with the measured row height

  // Selection policy: filter changes (selectTop) promote the top-ranked row
  // of the visible sub-table; otherwise keep the current selection while it
  // is still visible, falling back to the top row.
  let chosen = null;
  if (selectTop) {
    if (rows.length) chosen = state.results.find(item => item.name === rows[0].name);
  } else {
    if (activeStrategy) chosen = state.results.find(item => item.name === activeStrategy.name);
    if (!chosen && rows.length) chosen = state.results.find(item => item.name === rows[0].name);
  }
  if (chosen) {
    activeStrategy = chosen;
    updateKpisAndChart(chosen, rows);
  }
}

function updateKpisAndChart(strategy, rows) {
  const list = rows || displayedRows();
  const row = list.find(item => item.name === strategy.name) || {};
  byId("kpi-ce").innerHTML = money(row.ce != null ? row.ce : strategy.median) + '<span style="font-size:14px; font-weight:500; color:var(--text-muted)">/mo</span>';
  byId("kpi-median").innerHTML = money(strategy.median) + '<span style="font-size:14px; font-weight:500; color:var(--text-muted)">/mo</span>';
  byId("kpi-floor").innerHTML = money(row.floor != null ? row.floor : strategy.quantiles[20]) + '<span style="font-size:14px; font-weight:500; color:var(--text-muted)">/mo</span>';
  byId("kpi-buy-age").textContent = strategy.buyAge != null ? "Age " + strategy.buyAge.toFixed(1) : "Renter";
  byId("kpi-mortgage").textContent = strategy.mortgage != null ? money(strategy.mortgage) + "/mo" : "None";
  // Bequest trade-off line: the spending fraction of the max sustainable w*
  // the chosen preferences imply, and the median estate at 95 it leaves.
  const beqLine = byId("kpi-bequest");
  if (beqLine) {
    const applied = captureAppliedControls();
    const theta = applied.theta || 0;
    let text = "";
    if (theta > 0 && strategy.estate && strategy.estate.length) {
      const choice = bequestChoiceForStrategy(strategy, applied.gamma, theta, applied.k, state.dynamic || buildDynamicModel(readModelInputs()));
      if (choice) {
        text = "Spends " + Math.round(choice.f * 100) + "% of max sustainable to leave a median estate of " + money(choice.medianEstate) + ".";
      }
    }
    beqLine.textContent = text;
  }
  byId("chart-strategy-name").textContent = strategy.name;
  drawDistributionChart();
}

function selectStrategy(strategy) {
  activeStrategy = strategy;
  renderTable();
}

// ---------------------------------------------------------------------------
// S-Curve chart: real 201-point quantile ladder of the selected strategy.
// ---------------------------------------------------------------------------
let hoverPercentile = null;

function quantileAt(strategy, percentile) {
  const q = strategy.quantiles;
  const position = percentile * 200;
  const low = Math.floor(position);
  const high = Math.min(200, low + 1);
  const fraction = position - low;
  return q[low] + fraction * (q[high] - q[low]);
}

function drawDistributionChart() {
  const canvas = byId("chart-canvas");
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.parentElement.clientWidth;
  const height = canvas.parentElement.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const padLeft = 45, padRight = 20, padTop = 26, padBottom = 25;
  const pWidth = width - padLeft - padRight;
  const pHeight = height - padTop - padBottom;

  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (pHeight / 4) * i;
    ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(width - padRight, y); ctx.stroke();
  }

  if (!state.results || !activeStrategy) {
    ctx.fillStyle = "#64748b";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText("Run a simulation to draw the spending distribution", padLeft, padTop + pHeight / 2);
    return;
  }
  const q = activeStrategy.quantiles;
  const minVal = q[0], maxVal = Math.max(q[200], q[0] + 1);

  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const p = i / 200;
    const x = padLeft + p * pWidth;
    const y = padTop + pHeight * (1 - (q[i] - minVal) / (maxVal - minVal));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const markers = [
    { p: 0.10, label: "P10 Floor", color: "#d97706", bgColor: "#fffbeb", borderColor: "#fde68a" },
    { p: 0.50, label: "Median (P50)", color: "#059669", bgColor: "#ecfdf5", borderColor: "#a7f3d0" },
    { p: 0.90, label: "P90 Upside", color: "#0284c7", bgColor: "#f0f9ff", borderColor: "#bae6fd" }
  ];
  markers.forEach(m => {
    const x = padLeft + m.p * pWidth;
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, padTop + 2); ctx.lineTo(x, height - padBottom); ctx.stroke();
    ctx.setLineDash([]);
    const text = m.label + ": " + money(quantileAt(activeStrategy, m.p));
    ctx.font = "700 10px Inter, sans-serif";
    const textMetrics = ctx.measureText(text);
    const badgeW = textMetrics.width + 14;
    const badgeH = 18;
    const badgeX = Math.max(padLeft, Math.min(width - padRight - badgeW, x - badgeW / 2));
    const badgeY = padTop - 20;
    ctx.fillStyle = m.bgColor;
    ctx.strokeStyle = m.borderColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = m.color;
    ctx.fillText(text, badgeX + 7, badgeY + 12);
  });

  if (hoverPercentile !== null) {
    const p = hoverPercentile;
    const v = quantileAt(activeStrategy, p);
    const x = padLeft + p * pWidth;
    const y = padTop + pHeight * (1 - (v - minVal) / (maxVal - minVal));
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = "#0284c7"; ctx.fill(); ctx.stroke();
    byId("chart-crosshair-val").textContent = "P" + Math.round(p * 100) + ": " + money(v) + " / mo";
  } else {
    byId("chart-crosshair-val").textContent = "Hover curve to inspect percentiles";
  }

  ctx.fillStyle = "#64748b";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText(money(maxVal), 4, padTop + 8);
  ctx.fillText(money(minVal), 4, height - padBottom);
  ctx.fillText("P0", padLeft, height - 8);
  ctx.fillText("P50", padLeft + pWidth / 2 - 10, height - 8);
  ctx.fillText("P100", width - padRight - 28, height - 8);
}

// ---------------------------------------------------------------------------
// Simulation runner & event wiring
// ---------------------------------------------------------------------------
function updatePhaseLabels() {
  const careerStart = Number(byId("inp-careerStartAge").value);
  const retirement = Number(byId("inp-retirementAge").value);
  const pension = Number(byId("inp-pensionStartAge").value);
  if ([careerStart, retirement, pension].every(Number.isFinite)) {
    setText("th-accum", "Accumulation (" + careerStart + "–" + retirement + ")");
    setText("th-bridge", "Early Bridge (" + retirement + "–" + pension + ")");
    setText("th-post", "Post-" + pension);
  }
}

function updateLiveSettingsLabels() {
  const careerStart = parseInt(byId("inp-careerStartAge").value) || 25;
  const tier1 = byId("lbl-tier1");
  if (tier1) tier1.textContent = "Early Career (" + careerStart + " – 32)";
}

function setBusy(active) {
  byId("btn-run-sim").disabled = active;
  byId("slider-sim-count").disabled = active;
  document.querySelectorAll("#settings-window .form-input").forEach(input => input.disabled = active);
  document.querySelectorAll("#settings-window input[type='checkbox']").forEach(input => input.disabled = active);
}

async function runSimulation() {
  if (state.activeRun) {
    logWarn("runSimulation ignored: a run is already active.");
    return;
  }
  // The first click consumes the attention effect permanently.
  byId("btn-run-sim").classList.remove("btn-attract");
  let settings;
  try {
    settings = {simulations: simulationCountFromSlider(), model: readModelInputs()};
  } catch (error) {
    logError("Invalid settings — run aborted:", error.message);
    setStatus(error.message, true);
    setText("run-message", error.message);
    return;
  }
  const run = {cancelled: false};
  state.activeRun = run;
  setBusy(true);
  byId("btn-run-sim").textContent = "Simulating...";
  setStatus("WebGPU run active: " + state.adapterText, false);
  setText("run-message", "Generating paths and solving on GPU...");
  setProgress(0, 100, "Allocating buffers...");
  const totalStart = performance.now();
  // Watchdog: if the GPU hangs (e.g. a driver crash mid-run) the await on
  // onSubmittedWorkDone never resolves; flag it instead of failing silently.
  const watchdog = setTimeout(() => {
    if (state.activeRun === run) {
      logError("Run appears hung: no completion after 120s. Dumping diagnostics — check for GPU device-lost entries above.");
      dumpDiagnostics();
      setStatus("Run appears hung (120s). See the console diagnostics.", true);
      setText("run-message", "The run appears hung. Check the console (F12) for ENGINE diagnostics, or reload the page.");
    }
  }, 120000);
  try {
    const output = await simulate(settings, run);
    state.results = output.results;
    state.dynamic = output.dynamic;
    const totalElapsed = performance.now() - totalStart;
    const samplerNote = output.sampler === "rqmc" ? " · RQMC Sobol" : " · Threefry (?sampler=threefry)";
    setText("completed-meta", settings.simulations.toLocaleString("en-US") + " paths" + samplerNote);
    setText("timing-meta", (totalElapsed / 1000).toFixed(2) + "s total");
    setText("run-message", "Completed " + output.names.length.toLocaleString("en-US") + " allocations x " + settings.simulations.toLocaleString("en-US") + " paths." + (output.sampler === "rqmc" ? " Sampling: RQMC Sobol." : " Sampling: Threefry (?sampler=threefry)."));
    setProgress(100, 100, "Complete. Adjust controls below, then press \u201CUpdate Table & Re-Rank\u201D.");
    logInfo("Simulation completed:", output.names.length, "allocations x", settings.simulations, "paths in", (totalElapsed / 1000).toFixed(2), "s");
    updatePhaseLabels();
    activeStrategy = null;
    renderTable();
  } catch (error) {
    if (error.message === "__CANCELLED__") {
      setText("run-message", "Run cancelled.");
      setProgress(0, 100, "Ready.");
    } else {
      logError("runSimulation failed:", error.message || error, error.stack ? "\n" + error.stack : "");
      setStatus(error.message || "WebGPU simulation failed", true);
      setText("run-message", error.message || "Simulation failed");
    }
  } finally {
    clearTimeout(watchdog);
    state.activeRun = null;
    setBusy(false);
    byId("btn-run-sim").textContent = "Simulate";
  }
}

function updateGammaLabel() {
  const g = Number(byId("slider-gamma").value);
  let label = "Balanced";
  if (g === 0.0) label = "Risk-Neutral";
  else if (g >= 0.5 && g <= 2.5) label = "Aggressive";
  else if (g >= 3.0 && g <= 3.5) label = "Balanced";
  else if (g >= 4.0 && g <= 6.0) label = "Conservative";
  else if (g >= 6.5) label = "Highly-Conservative";
  setText("val-gamma", g.toFixed(1) + " (" + label + ")");
  setText("kpi-ce-sub", "The steady monthly income you would view as equally desirable to this strategy's fluctuating market outcomes, given your risk aversion (\u03B3 = " + g.toFixed(1) + ").");
}

function updateThetaLabel() {
  const t = Number(byId("slider-theta").value) || 0;
  let label = "0.00 (Off)";
  if (t > 0) {
    if (Math.abs(t - 0.5) < 1e-9) label = "0.50 (Balanced)";
    else if (t < 0.5) label = t.toFixed(2) + " (Mild)";
    else label = t.toFixed(2) + " (Strong)";
  }
  setText("val-theta", label);
}

function updateKLabel() {
  const k = Number(byId("slider-k").value) || 0;
  setText("val-k", k === 0 ? "$0" : money(k));
}

function resetDefaults() {
  applyModelToInputs(MODEL.inputs);
  byId("slider-gamma").value = String(DEFAULTS.gamma);
  byId("slider-lambda").value = "0";
  byId("slider-theta").value = String(DEFAULTS.bequestIntensity || 0);
  byId("slider-k").value = String(DEFAULTS.bequestCurvature || 0);
  byId("slider-sim-count").value = String(simulationSliderPosition(DEFAULTS.simulations));
  byId("chk-leverage").checked = false;
  setText("val-leverage-count", effectiveAllocationCount(false).toLocaleString("en-US") + " allocations");
  byId("filter-house").value = "ALL";
  byId("filter-accum").value = "ALL";
  byId("table-search").value = "";
  document.querySelectorAll("#seg-mix .seg-pill").forEach(btn => btn.classList.toggle("active", btn.dataset.mix === "ALL"));
  state.sort = { column: "ce", ascending: false, active: false };
  updateSortIndicators();
  updateGammaLabel();
  updateThetaLabel();
  updateKLabel();
  setText("val-lambda", "0.000 (Neutral)");
  setText("val-sim-count", DEFAULTS.simulations.toLocaleString("en-US") + " Paths");
  updateLiveSettingsLabels();
  updatePhaseLabels();
  renderTable();
}

// --- wiring ---------------------------------------------------------------
byId("btn-run-sim").addEventListener("click", runSimulation);
byId("btn-update-table").addEventListener("click", () => {
  if (!state.results) {
    setText("run-message", "Run a simulation first — there are no results to update.");
    return;
  }
  renderTable(true);
});
byId("btn-reset-defaults").addEventListener("click", resetDefaults);

byId("slider-gamma").addEventListener("input", updateGammaLabel);
byId("slider-lambda").addEventListener("input", (e) => {
  const l = parseFloat(e.target.value);
  setText("val-lambda", l === 0 ? "0.000 (Neutral)" : l.toFixed(3) + " (Active)");
});
byId("slider-theta").addEventListener("input", updateThetaLabel);
byId("slider-k").addEventListener("input", updateKLabel);
byId("slider-sim-count").addEventListener("input", (e) => {
  setText("val-sim-count", parseInt(e.target.value).toLocaleString("en-US") + " Paths");
});
byId("chk-leverage").addEventListener("change", (e) => {
  setText("val-leverage-count", effectiveAllocationCount(e.target.checked).toLocaleString("en-US") + " allocations");
  // With no results cached yet, reflect the new strategy count in the
  // segmented "All" pill; once a run exists the pill mirrors its results.
  if (!state.results) renderTable();
});

["inp-careerStartAge", "inp-retirementAge", "inp-pensionStartAge"].forEach(id => {
  byId(id).addEventListener("input", () => { updateLiveSettingsLabels(); updatePhaseLabels(); });
});

// Filter changes re-rank the visible sub-table and auto-select its top row.
byId("filter-house").addEventListener("change", () => renderTable(true));
byId("filter-accum").addEventListener("change", () => renderTable(true));
byId("table-search").addEventListener("input", () => renderTable(true));
document.querySelectorAll("#seg-mix .seg-pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#seg-mix .seg-pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderTable(true);
  });
});

// Virtualized leaderboard: repaint only the visible window on scroll
// (rAF-throttled; the full renderTable pipeline only runs on data changes).
(function () {
  const wrap = document.querySelector(".table-wrap");
  if (!wrap) return;
  let pending = false;
  wrap.addEventListener("scroll", () => {
    if (pending || !state.results) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (state.results) paintWindow(windowedRows);
    });
  }, {passive: true});
})();

// Column headers: click cycles biggest -> smallest -> smallest -> biggest ->
// default CE / rank order, applied to the currently filtered rows.
document.querySelectorAll("th.sortable").forEach(th => {
  th.addEventListener("click", () => cycleSort(th.dataset.col));
});

// Settings dock window toggle + category navigation (from the template).
const container = byId("app-container");
const btnToggle = byId("btn-toggle-settings");
const btnClose = byId("btn-close-settings");
const btnLabel = byId("settings-btn-label");
function toggleSettings() {
  const isOpen = container.classList.toggle("settings-open");
  btnToggle.classList.toggle("active", isOpen);
  btnLabel.textContent = isOpen ? "Close Settings Window" : "Open Settings Window";
  setTimeout(drawDistributionChart, 290);
}
btnToggle.addEventListener("click", toggleSettings);
btnClose.addEventListener("click", toggleSettings);
document.querySelectorAll(".cat-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".cat-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-section").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    byId(btn.dataset.tab).classList.add("active");
  });
});

// Chart hover crosshair.
const chartCanvas = byId("chart-canvas");
chartCanvas.addEventListener("mousemove", (e) => {
  const rect = chartCanvas.getBoundingClientRect();
  const padLeft = 45, padRight = 20;
  const pWidth = rect.width - padLeft - padRight;
  const x = e.clientX - rect.left - padLeft;
  if (x >= 0 && x <= pWidth) {
    hoverPercentile = x / pWidth;
    drawDistributionChart();
  }
});
chartCanvas.addEventListener("mouseleave", () => {
  hoverPercentile = null;
  drawDistributionChart();
});

window.addEventListener("resize", drawDistributionChart);

// --- initialization --------------------------------------------------------
applyModelToInputs(MODEL.inputs);
byId("slider-gamma").value = String(DEFAULTS.gamma);
byId("slider-lambda").value = "0";
byId("slider-theta").value = String(DEFAULTS.bequestIntensity || 0);
byId("slider-k").value = String(DEFAULTS.bequestCurvature || 0);
byId("slider-sim-count").value = String(simulationSliderPosition(DEFAULTS.simulations));
updateGammaLabel();
updateThetaLabel();
updateKLabel();
setText("val-lambda", "0.000 (Neutral)");
setText("val-sim-count", DEFAULTS.simulations.toLocaleString("en-US") + " Paths");
setText("val-leverage-count", effectiveAllocationCount(leverageEnabled()).toLocaleString("en-US") + " allocations");
updateLiveSettingsLabels();
updatePhaseLabels();
// The settings dock opens by default; the toggle button stays in sync.
container.classList.add("settings-open");
btnToggle.classList.add("active");
btnLabel.textContent = "Close Settings Window";
renderTable();
logInfo("ENGINE initialised: ", TOTAL_ALLOCATIONS, "strategies,", RUN_ALLOCATION_COUNT, "active",
        RUN_ALLOCATION_COUNT !== TOTAL_ALLOCATIONS ? "(?allocations= override)" : "");
createDeviceContext().catch(error => {
  setStatus(error.message || "WebGPU unavailable", true);
  setText("run-message", error.message);
});
"""

# ---------------------------------------------------------------------------
# Template surgery
# ---------------------------------------------------------------------------
STATUS_STRIP = """
<div class="engine-status" style="display:flex; flex-wrap:wrap; gap:6px 18px; align-items:center; padding:6px 12px; background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:2px; font-size:11px; font-family:var(--font-mono); color:var(--text-muted)">
  <span id="gpu-status" class="engine-chip ok">Checking WebGPU…</span>
  <span>Adapter: <b id="adapter-meta" style="color:var(--text-primary)">pending</b></span>
  <span>Paths: <b id="completed-meta" style="color:var(--text-primary)">no run yet</b></span>
  <span>Timing: <b id="timing-meta" style="color:var(--text-primary)">—</b></span>
  <span style="flex:1; text-align:right; color:var(--text-secondary)" id="run-message">Open Settings and press “Simulate” to run the engine.</span>
</div>
<div class="progress" style="height:4px; background:var(--border-subtle); border-radius:2px; overflow:hidden; margin:0">
  <div id="progress-fill" style="height:100%; width:0; background:var(--brand-green); transition:width .12s ease"></div>
</div>
<div style="display:flex; justify-content:space-between; font-size:10px; font-family:var(--font-mono); color:var(--text-dim); margin-top:2px">
  <span id="progress-detail">Ready.</span>
  <span>Calibrated skew-t model · ν = __NU__ · __OBS__ observations · __DATE_START__ → __DATE_END__</span>
</div>
"""


def _inject_status_strip(html: str, payload: dict) -> str:
    strip = STATUS_STRIP
    strip = strip.replace("__NU__", str(payload["returnModel"]["nu"]))
    strip = strip.replace("__OBS__", f"{payload['returnModel']['observations']:,}")
    strip = strip.replace("__DATE_START__", payload["returnModel"]["dateStart"])
    strip = strip.replace("__DATE_END__", payload["returnModel"]["dateEnd"])
    html = html.replace("</header>", "</header>\n" + strip, 1)
    html = html.replace(
        "</head>",
        "<style>.engine-chip.ok{color:var(--brand-green);font-weight:700}"
        ".engine-chip.bad{color:#dc2626;font-weight:700}"
        ".cma-toggle-row{display:flex;align-items:center;padding:2px 0}"
        ".cma-toggle{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:12px;font-weight:600;color:var(--text-primary)}"
        '.cma-toggle input[type="checkbox"]{width:15px;height:15px;accent-color:var(--brand-navy);cursor:pointer}'
        ".cma-toggle-hint{color:var(--text-muted);font-weight:500}"
        ".btn-attract{animation:engine-glow 1.9s ease-in-out infinite;position:relative}"
        ".btn-attract::after{content:\"\";position:absolute;inset:-5px;border-radius:4px;border:2px solid var(--brand-green);opacity:0;animation:engine-ring 1.9s ease-out infinite;pointer-events:none}"
        "@keyframes engine-glow{0%,100%{box-shadow:0 0 0 0 rgba(5,150,105,.40)}50%{box-shadow:0 0 16px 3px rgba(5,150,105,.55)}}"
        "@keyframes engine-ring{0%{opacity:.9;transform:scale(1)}70%{opacity:0;transform:scale(1.07)}100%{opacity:0}}</style></head>",
        1,
    )
    return html


# The settings validation reads the max share from the payload constants
# (C.glidepathMaxShare), so make sure lifecycle_constants carries it — done in
# calibration.lifecycle_constants. The template's section label is renamed
# during the build; the toggle anchor below still matches the ORIGINAL
# template label string (the rename happens in the same replace call).
CMA_TOGGLE_BLOCK = """\
<div class="cma-toggle-row">
  <label class="cma-toggle" for="inp-use-forward-cmas">
    <input type="checkbox" id="inp-use-forward-cmas" checked>
    <span>Use expected returns above <span class="cma-toggle-hint">&mdash; off uses historical</span></span>
  </label>
</div>
"""


def _inject_cma_toggle(html: str) -> str:
    marker = '<div class="section-label">Leverage Borrowing & Cash Buffer</div>'
    if marker not in html:
        raise RuntimeError("Could not locate the 'Leverage Borrowing & Cash Buffer' section label.")
    return html.replace(marker, CMA_TOGGLE_BLOCK + "\n" + marker, 1).replace(
        marker, '<div class="section-label">Leverage, Borrowing, Cash Buffer & Glidepaths</div>', 1
    )


# The house savings stream cap was missing from the UI; inject it right after
# the "House Savings Escalation" field of the "Housing Savings & Registered
# Programs" grid (schema index 8).
HOUSE_CAP_FIELD = """\
          <div class="input-field">
            <span class="field-label">Max House Savings Fraction (Cap)</span>
            <div class="input-affix-wrap"><input type="number" step="1" class="form-input" value="10.0"><span class="affix suffix">% net</span></div>
          </div>
"""


def _inject_house_cap_field(html: str) -> str:
    marker = (
        '<span class="field-label">House Savings Escalation</span>\n'
        '            <div class="input-affix-wrap"><input type="number" step="0.1" class="form-input" value="2.0"><span class="affix suffix">% / yr</span></div>\n'
        '          </div>'
    )
    if marker not in html:
        raise RuntimeError("Could not locate the House Savings Escalation field.")
    return html.replace(marker, marker + "\n" + HOUSE_CAP_FIELD.rstrip("\n"), 1)


def _replace_mock_script(html: str, model_json: str, runtime_js: str) -> str:
    """Replace the template's mock dataset <script> with model data + runtime."""
    script_re = re.compile(r"<script>\s*// Mock Quantitative Dataset.*?</script>", re.DOTALL)
    replacement = (
        '<script type="application/json" id="model-data">' + model_json + "</script>\n"
        "<script>\n" + runtime_js + "\n</script>"
    )
    new_html, count = script_re.subn(lambda _match: replacement, html, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the template's mock dataset script block.")
    return new_html


def build_html(price_path=DEFAULT_PRICE_PATH, output_path=DEFAULT_OUTPUT_PATH, config=None):
    """Assemble the standalone GitHub Pages application."""
    price_path = Path(price_path)
    output_path = Path(output_path)
    if config is None:
        config = cfg.instance_config()

    payload = calibration.build_payload(price_path, config)
    sources = engine.load_shader_sources()
    shader_json = json.dumps(engine._main_shader(sources), ensure_ascii=True)
    quantiles_json = json.dumps(sources[engine.QUANTILES_SHADER], ensure_ascii=True)
    bequest_json = json.dumps(sources[engine.BEQUEST_SHADER], ensure_ascii=True)
    model_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    runtime_js = RUNTIME_JS
    runtime_js = runtime_js.replace(SHADER_MARKER, shader_json)
    runtime_js = runtime_js.replace(QUANTILES_MARKER, quantiles_json)
    runtime_js = runtime_js.replace(BEQUEST_MARKER, bequest_json)
    # 14-bit truncated Joe-Kuo table for ?sampler=rqmc (bit-packed, base64).
    # Falls back to an empty table (RQMC opt-in unavailable) if the direction
    # numbers cannot be loaded or derived at build time.
    try:
        sobol_b64 = calibration.sobol_table_b64(config)
        sobol_dims = calibration.sobol_dimensions(config)
        print(f"Embedded {calibration.SOBOL_BROWSER_BITS}-bit Sobol table: "
              f"{len(sobol_b64) / 1024:.0f} KB base64, {sobol_dims} coordinates "
              f"(opt-in via ?sampler=rqmc)")
    except Exception as exc:
        sobol_b64 = ""
        sobol_dims = 0
        print(f"WARNING: ?sampler=rqmc unavailable ({exc}); the page will run "
              "Threefry only.")
    runtime_js = runtime_js.replace(SOBOL_TABLE_MARKER, sobol_b64)
    runtime_js = runtime_js.replace(SOBOL_DIMS_MARKER, str(sobol_dims))

    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = _inject_status_strip(html, payload)
    html = _inject_cma_toggle(html)
    html = _inject_house_cap_field(html)
    html = _replace_mock_script(html, model_json, runtime_js)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8", newline="\n")
    return output_path, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    started = time.perf_counter()
    output_path, payload = build_html(args.price_path, args.output)
    print(
        f"Built {output_path} with {payload['allocations']['count']:,} allocations "
        f"and {payload['returnModel']['observations']:,} calibrated observations "
        f"({payload['returnModel']['dateStart']} to {payload['returnModel']['dateEnd']})."
    )
    print(f"Generation time: {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()