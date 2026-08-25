"""Build the standalone GitHub Pages HTML for the Horizon engine.

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

RUNTIME_JS = r"""
"use strict";
// ===========================================================================
// HORIZON — WebGPU runtime for the Wealth & Lifetime Allocation Engine.
//
// This script replaces the template's mock dataset with the real engine:
//  1. It reads the calibrated model payload from the #model-data script tag.
//  2. It builds the parameter buffers and runs the five WGSL compute passes
//     (returns -> layoffs -> accumulation -> solver -> drawdowns) plus the
//     GPU quantile reduction.
//  3. Risk Aversion (gamma), Drawdown Aversion (lambda) and the strategy
//     filters re-rank the table INSTANTLY from the cached 201-point quantile
//     ladders (pure JavaScript, no GPU re-simulation):
//         CE_adj = CE(gamma) x exp(-lambda x Composite UI)
// ===========================================================================

const MODEL = JSON.parse(document.getElementById("model-data").textContent);
const SHADER_SOURCE = __SHADER_JSON__;
const QUANTILES_SHADER_SOURCE = __QUANTILES_JSON__;
const C = MODEL.constants;
const DEFAULTS = MODEL.defaults;
const TOTAL_ALLOCATIONS = MODEL.allocations.count;
const RUN_ALLOCATION_COUNT = Number(new URLSearchParams(location.search).get("allocations")) || TOTAL_ALLOCATIONS;
// The five underlying return series (VEQT, VEQT1.5, VEQT2, VGRO, VBAL).
// DECLINING/RISING accumulation glidepaths are monthly switching schedules,
// not return series, so they are sampled on-chip by the accumulate pass.
const RETURN_FUND_COUNT = 5;
const ALLOCATION_NAMES = MODEL.allocations.names;
const ALLOCATION_METADATA = new Uint32Array(MODEL.allocations.metadata);

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
// Console diagnostics: every failure path logs a "HORIZON" line with context,
// and window.dumpDiagnostics() prints a full snapshot for bug reports.
// ---------------------------------------------------------------------------
function horizonLog(level, ...args) {
  try {
    const prefix = "%cHORIZON%c";
    const styles = ["background:#0f172a;color:#fff;border-radius:3px;padding:1px 6px;font-weight:700;", ""];
    console[level](prefix, ...styles, ...args);
  } catch (_) { /* console unavailable */ }
}
const logInfo = (...args) => horizonLog("info", ...args);
const logDebug = (...args) => horizonLog("debug", ...args);
const logWarn = (...args) => horizonLog("warn", ...args);
const logError = (...args) => horizonLog("error", ...args);

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
  logInfo("=== HORIZON DIAGNOSTICS ===");
  logInfo("User agent:", navigator.userAgent);
  logInfo("WebGPU API present:", !!navigator.gpu);
  logInfo("URL:", location.href);
  logInfo("State:", JSON.stringify({
    results: state.results ? state.results.length : null,
    deviceReady: !!state.deviceContext,
    adapterText: state.adapterText,
    activeRun: !!state.activeRun,
    deviceLostReason: state.deviceLostReason || null,
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
  logInfo("=== END HORIZON DIAGNOSTICS ===");
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

// Each entry: [tab id, input index within that tab, model field, kind, apply]
// The input index is the flat position of the field inside the tab's
// .form-grid-2 containers, in DOM order. Kinds: years | money | pct.
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
  ["tab-cma", 9, "cashWedgeYears", "years"],

  ["tab-spend", 0, "smilePhase0", "pct"],
  ["tab-spend", 1, "smilePhase1", "pct"],
  ["tab-spend", 2, "smilePhase2", "pct"],
  ["tab-spend", 3, "smilePhase3", "pct"],
  ["tab-spend", 4, "skewDegreesFreedom", "years"],
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
  return String(value);
}

function parseForKind(kind, text) {
  if (kind === "pct") return parsePercent(text);
  if (kind === "money") return parseMoney(text);
  return Number(text);
}

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
  return input;
}

// ---------------------------------------------------------------------------
// Fiscal & calibration helpers (exact ports of the engine's math).
// ---------------------------------------------------------------------------
function inverse3(matrix) {
  const a = matrix[0], b = matrix[1], c = matrix[2], d = matrix[3], e = matrix[4], f = matrix[5], g = matrix[6], h = matrix[7], i = matrix[8];
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(determinant) < 1e-12) throw new Error("Return calibration covariance is singular.");
  return [
    (e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant,
    (f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant,
    (d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant
  ];
}

function cholesky3(matrix) {
  const output = new Array(9).fill(0);
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column <= row; column++) {
      let sum = matrix[row * 3 + column];
      for (let k = 0; k < column; k++) sum -= output[row * 3 + k] * output[column * 3 + k];
      if (row === column) {
        if (sum <= 1e-12) throw new Error("Return calibration covariance is not positive definite.");
        output[row * 3 + column] = Math.sqrt(sum);
      } else {
        output[row * 3 + column] = sum / output[column * 3 + column];
      }
    }
  }
  return output;
}

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

function calibrateReturnModel(config) {
  const values = MODEL.historicalReturns;
  const count = MODEL.historicalReturnCount;
  const means = [0, 0, 0];
  for (let row = 0; row < count; row++) for (let column = 0; column < 3; column++) means[column] += values[row * 3 + column];
  for (let column = 0; column < 3; column++) means[column] /= count;
  const covariance = new Array(9).fill(0);
  for (let row = 0; row < count; row++) for (let first = 0; first < 3; first++) for (let second = 0; second < 3; second++) covariance[first * 3 + second] += (values[row * 3 + first] - means[first]) * (values[row * 3 + second] - means[second]);
  for (let index = 0; index < 9; index++) covariance[index] /= count - 1;
  const correlation = new Array(9).fill(0);
  for (let first = 0; first < 3; first++) for (let second = 0; second < 3; second++) correlation[first * 3 + second] = covariance[first * 3 + second] / Math.sqrt(covariance[first * 3 + first] * covariance[second * 3 + second]);
  const meanReturns = config.useForwardLookingCmas ? [
    Math.log1p(config.cmas.VEQT) / 12 + 0.5 * covariance[0],
    Math.log1p(config.cmas.VGRO) / 12 + 0.5 * covariance[4],
    Math.log1p(config.cmas.VBAL) / 12 + 0.5 * covariance[8]
  ] : means;
  const nu = config.skewDegreesFreedom;
  const bNu = exactBNu(nu);
  const delta = [0, 1, 2].map(column => {
    let second = 0, third = 0;
    for (let row = 0; row < count; row++) {
      const centered = values[row * 3 + column] - means[column];
      second += centered * centered; third += centered * centered * centered;
    }
    second /= count; third /= count;
    return config.deltaCap * Math.tanh((third / Math.pow(second, 1.5)) / 3);
  });
  const inverseCorrelation = inverse3(correlation);
  let quadratic = 0;
  for (let first = 0; first < 3; first++) for (let second = 0; second < 3; second++) quadratic += delta[first] * inverseCorrelation[first * 3 + second] * delta[second];
  if (quadratic >= config.deltaTolerance) {
    const factor = Math.sqrt(config.deltaTolerance / quadratic);
    for (let index = 0; index < 3; index++) delta[index] *= factor;
  }
  const omega = [0, 1, 2].map(index => Math.sqrt(covariance[index * 3 + index] / (nu / (nu - 2) - delta[index] * delta[index] * bNu * bNu)));
  const calibratedCorrelation = new Array(9).fill(0);
  for (let first = 0; first < 3; first++) for (let second = 0; second < 3; second++) calibratedCorrelation[first * 3 + second] = ((nu - 2) / nu) * (covariance[first * 3 + second] / (omega[first] * omega[second]) + bNu * bNu * delta[first] * delta[second]);
  for (let index = 0; index < 3; index++) calibratedCorrelation[index * 3 + index] = 1;
  const residual = calibratedCorrelation.map((value, index) => value - delta[Math.floor(index / 3)] * delta[index % 3]);
  let cholesky;
  for (const jitter of [0, 1e-10, 1e-8, 1e-6]) {
    try { cholesky = cholesky3(residual.map((value, index) => value + (index % 4 === 0 ? jitter : 0))); break; } catch (_) { /* diagonal jitter */ }
  }
  if (!cholesky) throw new Error("Return calibration could not produce a positive-definite skew-t covariance.");
  return {xi: meanReturns.map((value, index) => value - omega[index] * delta[index] * bNu), omega, correlation: calibratedCorrelation, delta, cholesky, nu};
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
    : 1 - (65 - effectiveQppAge) * 0.072;
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
    hisaMonthly: Math.pow(1 + config.hisaAnnualRealReturn, 1 / 12) - 1, cashWedgeYears: config.cashWedgeYears,
    meltdownMonthly: config.meltdownBracketAnnual / 12, oasThresholdMonthly: config.oasClawbackThreshold / 12, oasClawbackRate: config.oasClawbackRate,
    employerMatchRate: config.employerMatchRate, employerMatchPercent: config.employerMatchPercent,
    realBorrowRateAnnual: config.realBorrowRateAnnual, extraMer15: config.extraMer15, extraMer20: config.extraMer20,
    layoffAnnualProbability: config.layoffAnnualProbability, bisectionSteps: 24,
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
  const returnModel = calibrateReturnModel(config);
  // Append the 18 calibrated skew-t constants (xi, omega, delta, row-major
  // Cholesky) after the tax tail, then the 10 house constants.
  const returnModelArray = [].concat(returnModel.xi, returnModel.omega, returnModel.delta, returnModel.cholesky);
  const houseConstantsArray = [
    constants.targetHouseCapital, constants.mortgagePrincipal, constants.mortgageMonthlyRate,
    constants.monthlyPropertyTaxesCondo, constants.monthlyMarketRent,
    constants.fhsaAnnualLimit, constants.fhsaMaxBalance, constants.hbpMaxWithdrawal,
    constants.hbpRepaymentYears, C.houseCount
  ];
  const staticValues = new Float32Array(career.length + month0.length + month1.length + taxValues.length + returnModelArray.length + houseConstantsArray.length);
  staticValues.set(career, 0); staticValues.set(month0, career.length); staticValues.set(month1, career.length + month0.length); staticValues.set(taxValues, career.length + month0.length + month1.length); staticValues.set(returnModelArray, career.length + month0.length + month1.length + taxValues.length); staticValues.set(houseConstantsArray, career.length + month0.length + month1.length + taxValues.length + returnModelArray.length);
  return {constants, career, month0, month1, taxValues, staticValues, smile, cpmWeights, returnModel, pension};
}

// ---------------------------------------------------------------------------
// WebGPU pipeline
// ---------------------------------------------------------------------------
function makeParams(dynamic, simulations, allocations, batchSims, simOffset) {
  const dimensions = dynamic.constants;
  const buffer = new ArrayBuffer(128);
  new Uint32Array(buffer, 0, 4).set([simulations, allocations, dimensions.totalMonths, dimensions.accumMonths]);
  new Uint32Array(buffer, 16, 4).set([dimensions.retireMonths, RETURN_FUND_COUNT, dimensions.bisectionSteps, dimensions.m75Start]);
  new Uint32Array(buffer, 32, 4).set([dimensions.postWedgeMonth, dimensions.currentAge, dimensions.careerStartAge, dimensions.retirementAge]);
  new Float32Array(buffer, 48, 4).set([dimensions.annualDistributionYield / 12, dimensions.taxOnDistributions, dimensions.hisaMonthly, dimensions.capitalGainsInclusion]);
  new Float32Array(buffer, 64, 4).set([dimensions.capitalGainsTaxRate, dimensions.cashWedgeYears, dimensions.meltdownMonthly, dimensions.oasThresholdMonthly]);
  new Float32Array(buffer, 80, 4).set([dimensions.oasClawbackRate, dimensions.employerMatchRate, dimensions.employerMatchPercent, dimensions.funds.length]);
  new Uint32Array(buffer, 96, 4).set([dimensions.seed, dimensions.skewDegreesFreedom, batchSims, simOffset]);
  new Float32Array(buffer, 112, 4).set([dimensions.realBorrowRateAnnual / 12, dimensions.extraMer15 / 12, dimensions.extraMer20 / 12, dimensions.layoffAnnualProbability]);
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
    const context = {device, layout, quantileLayout, generateReturns, generateLayoffs, accumulate, solve, trackDrawdowns, quantiles, limits: device.limits};
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

function selectedAllocationIndices(count) {
  if (count === TOTAL_ALLOCATIONS) return Array.from({length: TOTAL_ALLOCATIONS}, (_, i) => i);
  return Array.from({length: count}, (_, i) => Math.floor(i * (TOTAL_ALLOCATIONS - 1) / (count - 1)));
}

function glidepathBoundaries(code, months) {
  if (code === 5) {
    const half = Math.floor((months + 1) / 2);
    return [half, half + Math.floor((months + 2) / 4)];
  }
  if (code === 6) {
    const quarter = Math.floor((months + 2) / 4);
    return [quarter, quarter * 2];
  }
  return [0, 0];
}

function selectedAllocationBuffer(count, constants) {
  const indices = selectedAllocationIndices(count);
  const data = new Uint32Array(count * 12);
  for (let i = 0; i < count; i++) {
    const metadata = ALLOCATION_METADATA.subarray(indices[i] * 4, indices[i] * 4 + 4);
    const offset = i * 12;
    data.set(metadata, offset);
    data.set(glidepathBoundaries(metadata[0], constants.accumMonths), offset + 4);
    data.set(glidepathBoundaries(metadata[1], constants.bridgeMonths), offset + 6);
    data.set(glidepathBoundaries(metadata[2], constants.retireMonths - constants.bridgeMonths), offset + 8);
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
  const allocationCount = RUN_ALLOCATION_COUNT;
  const totalSims = settings.simulations;
  logInfo("Simulation start:", {simulations: totalSims, allocations: allocationCount, batchSize,
          totalMonths: dynamic.constants.totalMonths, careerYears: dynamic.constants.careerYears,
          pathCount: dynamic.constants.funds.length, houseCount: C.houseCount,
          retirementAge: settings.model.retirementAge, pensionStartAge: settings.model.pensionStartAge});
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
  const selected = selectedAllocationBuffer(allocationCount, dynamic.constants);
  const allocationBuffer = staticBuffer(device, selected.data);
  const modelBuffer = staticBuffer(device, dynamic.staticValues);
  const scratchSize = (batchSize * totalMonths * RETURN_FUND_COUNT + batchSize * careerYears + houseCount * pathCount * batchSize * 4 + houseCount * batchSize * 2 + batchSize * allocationCount) * 4;
  logDebug("GPU buffers (MB):", {
    scratch: (scratchSize / 1048576).toFixed(1),
    spending: (allocationCount * totalSims * 4 / 1048576).toFixed(1),
    drawdownReadback: (allocationCount * totalSims * 4 / 1048576).toFixed(1),
    quantileOutput: (allocationCount * 201 * 4 / 1048576).toFixed(1),
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
  const batchBindings = [scratchBuffer, allocationBuffer, modelBuffer, dummyA, dummyB, spendingBuffer];
  const totalBatches = Math.ceil(totalSims / batchSize);
  let offset = 0;
  try {
    for (let batchNumber = 0; batchNumber < totalBatches; batchNumber++) {
      if (run.cancelled) throw new Error("__CANCELLED__");
      const count = Math.min(batchSize, totalSims - offset);
      const batchStarted = performance.now();
      logDebug("Batch", (batchNumber + 1) + "/" + totalBatches, "sims", offset, "..", offset + count - 1);
      const params = device.createBuffer({size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(params, 0, makeParams(dynamic, totalSims, allocationCount, count, offset));
      const bindGroup = device.createBindGroup({layout: context.layout, entries: [params, ...batchBindings].map((buffer, binding) => ({binding, resource: {buffer}}))});
      const encoder = device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(context.generateReturns); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count * totalMonths / 64), 1, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.generateLayoffs); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count * careerYears / 64), 1, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.accumulate); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64), pathCount * houseCount, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.solve); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64), allocationCount, 1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(context.trackDrawdowns); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64), allocationCount, 1); pass.end();
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
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (run.cancelled) throw new Error("__CANCELLED__");
    setProgress(95, 100, "Computing quantiles on GPU...");
    logDebug("Quantile reduction: dispatching", allocationCount, "workgroups over", totalSims, "paths each");
    const quantileParams = device.createBuffer({size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(quantileParams, 0, makeParams(dynamic, totalSims, allocationCount, totalSims, 0));
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
      results.push({name: names[index], quantiles, median: quantiles[100], ui: uiMeans[index],
                    buyAge: hs.buyAge, p90BuyAge: hs.p90BuyAge, mortgage: hs.mortgage});
    }
    return {results, names, dynamic};
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

function lowerBound(values, target) {
  let low = 0, high = values.length;
  while (low < high) { const middle = (low + high) >> 1; if (values[middle] < target) low = middle + 1; else high = middle; }
  return low;
}

// Base CE depends only on (gamma, quantiles, dynamic model), so the per-row
// CRRA evaluation is cached per gamma value on the results array. The cache
// is naturally invalidated when a new run replaces state.results.
function ceBasesForGamma(gamma, dynamic) {
  const cache = state.results._ceCache;
  if (cache && cache.gamma === gamma) return cache.values;
  const values = state.results.map(s => ceForQuantiles(s.quantiles, gamma, dynamic));
  state.results._ceCache = {gamma, values};
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
  const floorP = applied.floorP;
  const dynamic = state.dynamic || buildDynamicModel(readModelInputs());
  const floorIndex = Math.round(floorP * 2);
  const ceBases = ceBasesForGamma(gamma, dynamic);

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

function renderTable(selectTop) {
  state.applied = captureAppliedControls();
  updateSortIndicators();
  const rows = displayedRows();
  const body = byId("table-body");
  const pill = document.querySelector('#seg-mix .seg-pill[data-mix="ALL"]');
  if (pill) pill.textContent = "All (" + (state.results ? state.results.length : TOTAL_ALLOCATIONS).toLocaleString("en-US") + ")";

  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="12" style="text-align:center; padding:32px; color:var(--text-muted); font-family:var(--font-mono)">' +
      (state.results ? "No strategies match your criteria." : "No results yet — open Settings and press Simulate.") + "</td></tr>";
    if (activeStrategy) drawDistributionChart();
    return;
  }

  body.innerHTML = rows.map((row, index) => {
    const selected = activeStrategy && row.name === activeStrategy.name ? ' class="selected-row"' : "";
    return "<tr" + selected + " data-name=\"" + escapeHtml(row.name) + "\">" +
      '<td class="mono" style="font-weight:700; color:var(--brand-navy)">' + (index + 1) + "</td>" +
      "<td>" + renderHouseBadge(row.house) + "</td>" +
      '<td><div class="badge-group">' + renderBadge(row.parts[2]) + "</div></td>" +
      '<td><div class="badge-group">' + renderBadge(row.parts[3]) + "</div></td>" +
      '<td><div class="badge-group">' + renderBadge(row.parts[4]) + "</div></td>" +
      '<td class="right mono" style="color:var(--text-muted)">' + (row.buyAge == null ? "—" : row.buyAge.toFixed(1)) + "</td>" +
      '<td class="right mono" style="color:var(--text-muted)">' + (row.mortgage == null ? "—" : money(row.mortgage)) + "</td>" +
      '<td class="right mono" style="font-weight:700; color:var(--brand-green)">' + money(row.ce) + "/mo</td>" +
      '<td class="right mono">' + money(row.median) + "/mo</td>" +
      '<td class="right mono" style="color:' + uiSeverity(row.ui) + '">' + f2(row.ui) + "</td>" +
      '<td class="right mono" style="color:var(--brand-amber); font-weight:600">' + money(row.floor) + "/mo</td>" +
      '<td class="right mono">' + money(row.p90) + "/mo</td></tr>";
  }).join("");

  body.querySelectorAll("tr[data-name]").forEach(tr => {
    tr.addEventListener("click", () => {
      const name = tr.dataset.name;
      const strategy = state.results.find(item => item.name === name);
      if (strategy) selectStrategy(strategy);
    });
  });

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
    ctx.font = "12px Plus Jakarta Sans, sans-serif";
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
    ctx.font = "700 10px Plus Jakarta Sans, sans-serif";
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
      setText("run-message", "The run appears hung. Check the console (F12) for HORIZON diagnostics, or reload the page.");
    }
  }, 120000);
  try {
    const output = await simulate(settings, run);
    state.results = output.results;
    state.dynamic = output.dynamic;
    const totalElapsed = performance.now() - totalStart;
    setText("completed-meta", settings.simulations.toLocaleString("en-US") + " paths");
    setText("timing-meta", (totalElapsed / 1000).toFixed(2) + "s total");
    setText("run-message", "Completed " + output.names.length.toLocaleString("en-US") + " allocations x " + settings.simulations.toLocaleString("en-US") + " paths.");
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

function resetDefaults() {
  applyModelToInputs(MODEL.inputs);
  byId("slider-gamma").value = String(DEFAULTS.gamma);
  byId("slider-lambda").value = "0";
  byId("slider-sim-count").value = String(simulationSliderPosition(DEFAULTS.simulations));
  byId("filter-house").value = "ALL";
  byId("filter-accum").value = "ALL";
  byId("table-search").value = "";
  document.querySelectorAll("#seg-mix .seg-pill").forEach(btn => btn.classList.toggle("active", btn.dataset.mix === "ALL"));
  state.sort = { column: "ce", ascending: false, active: false };
  updateSortIndicators();
  updateGammaLabel();
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
  renderTable();
});
byId("btn-reset-defaults").addEventListener("click", resetDefaults);

byId("slider-gamma").addEventListener("input", updateGammaLabel);
byId("slider-lambda").addEventListener("input", (e) => {
  const l = parseFloat(e.target.value);
  setText("val-lambda", l === 0 ? "0.000 (Neutral)" : l.toFixed(3) + " (Active)");
});
byId("slider-sim-count").addEventListener("input", (e) => {
  setText("val-sim-count", parseInt(e.target.value).toLocaleString("en-US") + " Paths");
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
byId("slider-sim-count").value = String(simulationSliderPosition(DEFAULTS.simulations));
updateGammaLabel();
setText("val-lambda", "0.000 (Neutral)");
setText("val-sim-count", DEFAULTS.simulations.toLocaleString("en-US") + " Paths");
updateLiveSettingsLabels();
updatePhaseLabels();
// The settings dock opens by default; the toggle button stays in sync.
container.classList.add("settings-open");
btnToggle.classList.add("active");
btnLabel.textContent = "Close Settings Window";
renderTable();
logInfo("HORIZON initialised: ", TOTAL_ALLOCATIONS, "strategies,", RUN_ALLOCATION_COUNT, "active",
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
<div class="engine-status" style="display:flex; flex-wrap:wrap; gap:6px 18px; align-items:center; padding:8px 14px; background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:10px; box-shadow:var(--shadow-sm); font-size:11px; font-family:var(--font-mono); color:var(--text-muted)">
  <span id="gpu-status" class="engine-chip ok">Checking WebGPU…</span>
  <span>Adapter: <b id="adapter-meta" style="color:var(--text-primary)">pending</b></span>
  <span>Paths: <b id="completed-meta" style="color:var(--text-primary)">no run yet</b></span>
  <span>Timing: <b id="timing-meta" style="color:var(--text-primary)">—</b></span>
  <span style="flex:1; text-align:right; color:var(--text-secondary)" id="run-message">Open Settings and press “Simulate” to run the engine.</span>
</div>
<div class="progress" style="height:4px; background:var(--border-subtle); border-radius:99px; overflow:hidden; margin:0">
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
        ".btn-attract{animation:horizon-glow 1.9s ease-in-out infinite;position:relative}"
        ".btn-attract::after{content:\"\";position:absolute;inset:-5px;border-radius:11px;border:2px solid var(--brand-green);opacity:0;animation:horizon-ring 1.9s ease-out infinite;pointer-events:none}"
        "@keyframes horizon-glow{0%,100%{box-shadow:0 0 0 0 rgba(5,150,105,.40)}50%{box-shadow:0 0 16px 3px rgba(5,150,105,.55)}}"
        "@keyframes horizon-ring{0%{opacity:.9;transform:scale(1)}70%{opacity:0;transform:scale(1.07)}100%{opacity:0}}</style></head>",
        1,
    )
    return html


# Toggle placed above the "Leverage Borrowing & Cash Buffer" group in the
# CMAs tab: ON = use the expected returns listed above, OFF = historical.
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
    return html.replace(marker, CMA_TOGGLE_BLOCK + "\n" + marker, 1)


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
    model_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    runtime_js = RUNTIME_JS
    runtime_js = runtime_js.replace(SHADER_MARKER, shader_json).replace(QUANTILES_MARKER, quantiles_json)

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