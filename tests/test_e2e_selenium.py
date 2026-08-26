"""Selenium E2E verification of the standalone ``index.html``.

Loads the built page in headless Chrome/Edge with WebGPU enabled, runs a full
simulation, then verifies:

  1. WebGPU device creation succeeds and a 5,040-strategy run completes with
     sane timing and zero browser console errors.
  2. lambda = 0 keeps the base CE exactly (identity gate) and rows are sorted
     by base CE descending.
  3. Setting lambda > 0 and clicking "Update Table & Re-Rank" re-ranks
     INSTANTLY from cached quantiles (no GPU re-simulation; timing metadata
     unchanged) and every displayed CE equals
     CE x exp(-lambda x UI) recomputed from the page's own quantiles.
  4. Table sorting, search filtering and strategy selection stay responsive.
  5. UI ordering sanity: more equity / leverage => higher Composite UI, and
     the +CASH bridge lowers the Composite UI.
  6. Live reactive parameter binding: editing the career start age updates
     the tier labels and table phase headers immediately.
  7. The Leverage checkbox defaults to OFF and gates the strategy pool: a
     default run simulates only the 1,600 non-leveraged allocations (no
     VEQT 1.5 / VEQT 2.0); with it ON the run covers all 5,040.

Requires: pip install selenium (chromedriver resolved by Selenium Manager).

Run with:  python tests/test_e2e_selenium.py [--simulations 1000] [--headed]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "index.html"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    str(Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def make_driver(headed: bool) -> webdriver.Chrome:
    options = Options()
    binary = find_chrome()
    if not binary:
        raise RuntimeError("No Chrome or Edge executable found.")
    options.binary_location = binary
    if not headed:
        options.add_argument("--headless=new")
    for arg in ("--enable-unsafe-webgpu", "--use-angle=d3d11", "--disable-gpu-sandbox", "--no-first-run", "--window-size=1600,1000"):
        options.add_argument(arg)
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    return webdriver.Chrome(options=options)


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"## {title}")
    print("=" * 100)


def run_simulation(driver, simulations: int, timeout: float = 900.0) -> None:
    # The simulation runner lives inside the docked settings window, so open
    # it first (idempotent).
    driver.execute_script(
        "if (!byId('app-container').classList.contains('settings-open')) byId('btn-toggle-settings').click();"
    )
    driver.execute_script(
        "byId('slider-sim-count').value = String(simulationSliderPosition(%d));"
        "byId('slider-sim-count').dispatchEvent(new Event('input'));" % simulations
    )
    driver.find_element(By.ID, "btn-run-sim").click()
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = driver.execute_script(
            "return {active: !!state.activeRun, done: !!state.results, bad: byId('gpu-status').className.includes('bad'), runMsg: byId('run-message').textContent};"
        )
        if info["bad"]:
            raise RuntimeError(f"Run failed: {info['runMsg']}")
        if not info["active"] and info["done"] and "Completed" in info["runMsg"]:
            return
        time.sleep(2)
    raise RuntimeError("Timed out waiting for simulation")


def set_lambda(driver, value: float) -> None:
    driver.execute_script(
        "byId('slider-lambda').value = String(%r);"
        "byId('slider-lambda').dispatchEvent(new Event('input'));" % value
    )


def set_gamma(driver, value: float) -> None:
    driver.execute_script(
        "byId('slider-gamma').value = String(%r);"
        "byId('slider-gamma').dispatchEvent(new Event('input'));" % value
    )


def click_update(driver) -> float:
    started = time.perf_counter()
    driver.find_element(By.ID, "btn-update-table").click()
    return time.perf_counter() - started


def rows_snapshot(driver) -> list[dict]:
    return driver.execute_script(
        "return displayedRows().map(r => ({name: r.name, ce: r.ce, ceBase: r.ceBase, ui: r.ui}));"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    driver = make_driver(args.headed)
    failures = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {label} {detail}")

    try:
        driver.get(HTML.as_uri())
        deadline = time.time() + 60
        while time.time() < deadline:
            if driver.execute_script(
                "return !!state.deviceContext && !byId('gpu-status').className.includes('bad');"
            ):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("WebGPU device not ready")

        settings_open = driver.execute_script(
            "return byId('app-container').classList.contains('settings-open');"
        )
        check("settings window opens by default", settings_open)
        btn_state = driver.execute_script(
            "return {label: byId('btn-run-sim').textContent.trim(), attract: byId('btn-run-sim').classList.contains('btn-attract')};"
        )
        check("Simulate button labelled 'Simulate'", btn_state["label"] == "Simulate", f"'{btn_state['label']}'")
        check("Simulate button carries the attention effect before the first run", btn_state["attract"])

        section("RUN")
        lever = driver.execute_script(
            "return {checked: byId('chk-leverage').checked, label: byId('val-leverage-count').textContent};"
        )
        check("Leverage checkbox defaults to OFF", not lever["checked"] and lever["label"] == "1,600 allocations", f"'{lever['label']}'")
        run_simulation(driver, args.simulations)
        info = driver.execute_script(
            "return {timing: byId('timing-meta').textContent, completed: byId('completed-meta').textContent, n: state.results.length, top: displayedRows()[0].name};"
        )
        print(f"timing: {info['timing']}  completed: {info['completed']}  strategies: {info['n']:,}")
        check("default run (leverage OFF) simulates 1,600 strategies", info["n"] == 1600, f"got {info['n']:,}")
        no_lev_names = driver.execute_script(
            "return state.results.filter(s => s.name.includes('VEQT1.5') || s.name.includes('VEQT2')).length;"
        )
        check("default run excludes every VEQT1.5 / VEQT2 strategy", no_lev_names == 0, f"{no_lev_names:,} leveraged rows")
        check("timing metadata populated", info["timing"] != "—" and "s total" in info["timing"], f"'{info['timing']}'")
        print(f"  top strategy: {info['top']}")
        btn_after = driver.execute_script(
            "return {label: byId('btn-run-sim').textContent.trim(), attract: byId('btn-run-sim').classList.contains('btn-attract')};"
        )
        check("attention effect disappears after the first run", not btn_after["attract"])
        check("button label restored after the run", btn_after["label"] == "Simulate", f"'{btn_after['label']}'")
        # The sections below (leveraged UI ordering, search narrowing) need the
        # full 5,040 space, so re-run with the Leverage checkbox ON.
        driver.execute_script(
            "byId('chk-leverage').checked = true; byId('chk-leverage').dispatchEvent(new Event('change'));"
        )
        run_simulation(driver, args.simulations)
        full = driver.execute_script(
            "return {n: state.results.length, leveraged: state.results.filter(s => s.name.includes('VEQT1.5') || s.name.includes('VEQT2')).length};"
        )
        check("leverage ON runs all 5,040 strategies", full["n"] == 5040, f"got {full['n']:,}")
        check("leverage ON includes leveraged strategies", full["leveraged"] > 0, f"{full['leveraged']:,} leveraged rows")

        section("BADGE COLOR CODING")
        badges = driver.execute_script(
            "return {"
            "v15: renderBadge('VEQT1.5').includes('chip-veqt15'),"
            "v2: renderBadge('VEQT2').includes('chip-veqt2'),"
            "glide: renderBadge('DECLINING').includes('chip-glide') && renderBadge('RISING').includes('chip-glide'),"
            "veqt: renderBadge('VEQT').includes('chip-veqt'),"
            "vgro: renderBadge('VGRO').includes('chip-vgro'),"
            "vbal: renderBadge('VBAL').includes('chip-vbal'),"
            "cash: renderBadge('VEQT+CASH').includes('chip-cash') && renderBadge('VEQT+CASH').includes('HISA')};"
        )
        check("1.5x leverage badge uses the orange chip (chip-veqt15)", badges["v15"])
        check("2.0x leverage badge uses the red chip (chip-veqt2)", badges["v2"])
        check("glidepath badges share one distinct chip (chip-glide)", badges["glide"])
        check("cash wedge renders as HISA chip", badges["cash"])
        house_cash = driver.execute_script("return renderHouseBadge('HOUSE_CASH');")
        check("HOUSE_CASH buyer badge uses the same HISA emerald chip",
              "chip-cash" in house_cash and "HISA" in house_cash and "Buyer (CASH)" in house_cash,
              house_cash[:120])
        colors = driver.execute_script(
            "const chip = cls => { const el = document.createElement('span'); el.className = 'badge-chip ' + cls;"
            " document.body.appendChild(el); const c = getComputedStyle(el).color; const b = getComputedStyle(el).backgroundColor;"
            " el.remove(); return {c, b}; };"
            "const badge = () => { const el = document.createElement('span'); el.className = 'badge';"
            " el.innerHTML = '<span class=\"badge-chip chip-veqt\">100/0</span><span class=\"badge-text\">VEQT</span>';"
            " document.body.appendChild(el); const bg = getComputedStyle(el).backgroundColor;"
            " const tc = getComputedStyle(el.querySelector('.badge-text')).color; el.remove(); return {bg, tc}; };"
            "return {veqt: chip('chip-veqt'), v15: chip('chip-veqt15'), v2: chip('chip-veqt2'),"
            " vgro: chip('chip-vgro'), vbal: chip('chip-vbal'), glide: chip('chip-glide'), cash: chip('chip-cash'), badge: badge()};"
        )
        chips = {
            "veqt": "rgb(30, 58, 138)",      # midnight navy
            "v15": "rgb(234, 88, 12)",       # orange
            "v2": "rgb(220, 38, 38)",        # crimson red
            "vgro": "rgb(37, 99, 235)",      # royal blue
            "vbal": "rgb(13, 148, 136)",     # teal
            "glide": "rgb(2, 132, 199)",     # steel blue
            "cash": "rgb(5, 150, 105)",      # emerald green
        }
        for key, rgb in chips.items():
            check(f"{key} chip renders {rgb} with white text",
                  colors[key]["b"] == rgb and colors[key]["c"] == "rgb(255, 255, 255)",
                  f"bg {colors[key]['b']} text {colors[key]['c']}")
        check("badge is a white box with dark charcoal label text",
              colors["badge"]["bg"] == "rgb(255, 255, 255)" and colors["badge"]["tc"] == "rgb(15, 23, 42)",
              f"bg {colors['badge']['bg']} text {colors['badge']['tc']}")

        section("UI DISTRIBUTION (all strategies)")
        dist = driver.execute_script(
            "const u = state.results.map(s => s.ui).sort((a, b) => a - b);"
            "return {min: u[0], max: u[u.length - 1], p10: u[Math.floor(u.length * 0.1)], p50: u[Math.floor(u.length * 0.5)], p90: u[Math.floor(u.length * 0.9)], mean: u.reduce((a, b) => a + b, 0) / u.length};"
        )
        print(json.dumps(dist, indent=2))
        check("all strategies have positive Composite UI", dist["min"] > 0)
        check("Composite UI spread is material", dist["max"] - dist["min"] > 1.0, f"range {dist['max'] - dist['min']:.2f}")

        section("LAMBDA = 0 (baseline identity)")
        # The shipped defaults carry a bequest motive (theta 0.5 / k $200k),
        # so zero the bequest sliders before capturing the plain-CE baseline
        # the lambda formula gates below compare against.
        driver.execute_script(
            "byId('slider-theta').value = '0'; byId('slider-theta').dispatchEvent(new Event('input'));"
            "byId('slider-k').value = '0'; byId('slider-k').dispatchEvent(new Event('input'));"
        )
        click_update(driver)
        base = rows_snapshot(driver)
        check("lambda=0 CE equals ceBase for every row", all(abs(r["ce"] - r["ceBase"]) < 1e-9 for r in base))
        check("lambda=0 rows sorted by base CE desc", all(base[i]["ce"] >= base[i + 1]["ce"] for i in range(len(base) - 1)))
        print(f"  top1: {base[0]['name']} CE {base[0]['ce']:,.1f} UI {base[0]['ui']:.1f}")

        section("LAMBDA = 0.01 (instant re-rank + CE_adj formula)")
        timing_before = driver.execute_script("return byId('timing-meta').textContent;")
        set_lambda(driver, 0.01)
        in_page_ms = driver.execute_script(
            "const t0 = performance.now(); renderTable(); return performance.now() - t0;"
        )
        elapsed = click_update(driver)
        after = rows_snapshot(driver)
        sel = driver.execute_script("return activeStrategy ? activeStrategy.name : null;")
        check("Update Table auto-selects the new top strategy", sel == after[0]["name"], f"'{sel}'")
        timing_after = driver.execute_script("return byId('timing-meta').textContent;")

        formula_ok = driver.execute_script(
            "const lam = Number(byId('slider-lambda').value);"
            "const gamma = Number(byId('slider-gamma').value);"
            "const dyn = state.dynamic || buildDynamicModel(readModelInputs());"
            "let ok = true; let worst = 0;"
            "for (const r of displayedRows()) {"
            "  const s = state.results.find(x => x.name === r.name);"
            "  const expected = ceForQuantiles(s.quantiles, gamma, dyn) * Math.exp(-lam * s.ui);"
            "  const rel = Math.abs(r.ce - expected) / Math.max(expected, 1);"
            "  if (rel > worst) worst = rel;"
            "  if (rel > 1e-9) ok = false;"
            "}"
            "return {ok, worst};"
        )
        check("every displayed CE == base CE x exp(-lambda x UI)", formula_ok["ok"], f"worst rel {formula_ok['worst']:.2e}")
        check("rows sorted by adjusted CE desc", all(after[i]["ce"] >= after[i + 1]["ce"] for i in range(len(after) - 1)))
        check("lambda>0 CE differs from base CE", any(abs(r["ce"] - r["ceBase"]) > 1e-6 for r in after[:20]))
        check("no GPU re-simulation (timing unchanged)", timing_before == timing_after, f"'{timing_before}'")
        active = driver.execute_script("return !!state.activeRun;")
        check("no active run after Update Table", not active)
        print(f"  Update Table latency: {elapsed * 1000:.1f} ms total (incl. Selenium round-trip); in-page renderTable: {in_page_ms:.1f} ms (pure JS post-processing)")
        print(f"  top1 at lambda=0.01: {after[0]['name']} CE {after[0]['ce']:,.1f} UI {after[0]['ui']:.1f}")
        print(f"  top-10 changed vs lambda=0: {[r['name'] for r in after[:10]] != [r['name'] for r in base[:10]]}")

        section("LAMBDA BACK TO 0 (identity restoration)")
        set_lambda(driver, 0.0)
        click_update(driver)
        restored = rows_snapshot(driver)
        same = all(
            abs(a["ce"] - b["ce"]) < 1e-6 and a["name"] == b["name"]
            for a, b in zip(restored, base)
        )
        check("lambda=0 restores exact baseline order/CE", same)

        section("BEQUEST PREFERENCES (theta / k instant re-rank)")
        title = driver.execute_script(
            "return document.querySelector('.workspace-grid .card .card-title').textContent.trim();"
        )
        check("risk panel renamed to Risk & Estate Preferences", title == "Risk & Estate Preferences", f"'{title}'")
        beq = driver.execute_script(
            "const s = state.results[0];"
            "return {grids: s.estate ? s.estate.length : 0, ladder: s.estate && s.estate[0] ? s.estate[0].length : 0,"
            " finite: s.estate ? s.estate.every(l => l.every(v => Number.isFinite(v))) : false};"
        )
        check("every strategy carries cached estate ladders (grid x 201)",
              beq["grids"] == 6 and beq["ladder"] == 201 and beq["finite"], str(beq))
        driver.execute_script(
            "byId('slider-theta').value = '0.5'; byId('slider-theta').dispatchEvent(new Event('input'));"
        )
        parity_label = driver.execute_script("return byId('val-theta').textContent;")
        check("theta=0.5 labels as Balanced", "Balanced" in parity_label, f"'{parity_label}'")
        driver.execute_script(
            "byId('slider-theta').value = '0'; byId('slider-theta').dispatchEvent(new Event('input'));"
        )

        def set_bequest(theta: float, k: float) -> None:
            driver.execute_script(
                "byId('slider-theta').value = String(%r); byId('slider-theta').dispatchEvent(new Event('input'));"
                "byId('slider-k').value = String(%r); byId('slider-k').dispatchEvent(new Event('input'));" % (theta, k)
            )

        timing_before = driver.execute_script("return byId('timing-meta').textContent;")
        set_bequest(1.0, 0.0)
        click_update(driver)
        beq_on = rows_snapshot(driver)
        timing_after = driver.execute_script("return byId('timing-meta').textContent;")
        check("theta>0 lowers every base CE at the default gamma (bequest costs spending)",
              all(beq_on[i]["ceBase"] < base[i]["ceBase"] - 1.0 for i in range(20)),
              f"top1 {base[0]['ceBase']:,.1f} -> {beq_on[0]['ceBase']:,.1f}")
        check("no GPU re-simulation for the bequest re-rank (timing unchanged)",
              timing_before == timing_after, f"'{timing_before}'")
        check("rows stay sorted by bequest-adjusted CE desc",
              all(beq_on[i]["ce"] >= beq_on[i + 1]["ce"] for i in range(len(beq_on) - 1)))
        set_bequest(1.0, 500000.0)
        click_update(driver)
        beq_k = rows_snapshot(driver)
        check("k>0 softens the bequest motive (CE rises vs k=0)",
              all(beq_k[i]["ceBase"] > beq_on[i]["ceBase"] + 1.0 for i in range(20)),
              f"top1 {beq_on[0]['ceBase']:,.1f} -> {beq_k[0]['ceBase']:,.1f}")
        set_bequest(0.0, 0.0)
        click_update(driver)
        beq_off = rows_snapshot(driver)
        same = all(
            abs(a["ce"] - b["ce"]) < 1e-6 and a["name"] == b["name"]
            for a, b in zip(beq_off, base)
        )
        check("theta=0 restores the exact baseline order/CE", same)

        section("UI ORDERING SANITY")
        order = driver.execute_script(
            "const byName = {};"
            "for (const s of state.results) byName[s.name] = s.ui;"
            "const pick = names => { const rows = names.map(n => ({n, u: byName[n]})).filter(x => x.u !== undefined); return rows; };"
            "const pure = pick(['HOUSE_NONE_VEQT_VEQT_VEQT', 'HOUSE_NONE_VEQT1.5_VEQT1.5_VEQT1.5', 'HOUSE_NONE_VEQT2_VEQT2_VEQT2']);"
            "const cash = pick(['HOUSE_NONE_VEQT_VEQT_VEQT', 'HOUSE_NONE_VEQT_VEQT+CASH_VEQT', 'HOUSE_NONE_VGRO_VGRO_VGRO', 'HOUSE_NONE_VGRO_VGRO+CASH_VGRO', 'HOUSE_NONE_VBAL_VBAL_VBAL', 'HOUSE_NONE_VBAL_VBAL+CASH_VBAL']);"
            "return {pure, cash};"
        )
        for item in order["pure"]:
            print(f"   pure {item['n']:26s} UI {item['u']:.2f}")
        pure_map = {x["n"]: x["u"] for x in order["pure"]}
        check("pure VEQT1.5 UI > pure VEQT UI", pure_map.get("HOUSE_NONE_VEQT1.5_VEQT1.5_VEQT1.5", -1) > pure_map.get("HOUSE_NONE_VEQT_VEQT_VEQT", 1e9))
        check("pure VEQT2 UI > pure VEQT1.5 UI", pure_map.get("HOUSE_NONE_VEQT2_VEQT2_VEQT2", -1) > pure_map.get("HOUSE_NONE_VEQT1.5_VEQT1.5_VEQT1.5", 1e9))
        cash_map = {x["n"]: x["u"] for x in order["cash"]}
        for base_name, cash_name in [("HOUSE_NONE_VEQT_VEQT_VEQT", "HOUSE_NONE_VEQT_VEQT+CASH_VEQT"), ("HOUSE_NONE_VGRO_VGRO_VGRO", "HOUSE_NONE_VGRO_VGRO+CASH_VGRO"), ("HOUSE_NONE_VBAL_VBAL_VBAL", "HOUSE_NONE_VBAL_VBAL+CASH_VBAL")]:
            if base_name in cash_map and cash_name in cash_map:
                check(f"+CASH bridge lowers UI ({cash_name})", cash_map[cash_name] < cash_map[base_name], f"{cash_map[base_name]:.2f} vs {cash_map[cash_name]:.2f}")

        section("GAMMA SLIDER (instant post-processing)")
        set_gamma(driver, 4.0)
        click_update(driver)
        gamma_state = driver.execute_script(
            "const g = Number(byId('slider-gamma').value);"
            "const dyn = state.dynamic || buildDynamicModel(readModelInputs());"
            "const rows = displayedRows();"
            "const expected = ceForQuantiles(state.results.find(x => x.name === rows[0].name).quantiles, g, dyn);"
            "return {sub: byId('kpi-ce-sub').textContent, topCe: rows[0].ce, expected};"
        )
        check("gamma label text updates", "γ = 4.0" in gamma_state["sub"], f"'{gamma_state['sub'][:60]}...'")
        check("CE recomputed at gamma=4", abs(gamma_state["topCe"] - gamma_state["expected"]) < 1e-6, f"{gamma_state['topCe']:,.1f} vs {gamma_state['expected']:,.1f}")
        set_gamma(driver, 3.0)
        click_update(driver)

        section("FILTERS, SEARCH & SELECTION")

        def top_selection():
            return driver.execute_script(
                "const top = displayedRows()[0];"
                "return {active: activeStrategy ? activeStrategy.name : null, top: top ? top.name : null};"
            )

        driver.execute_script("byId('table-search').value = 'VGRO'; byId('table-search').dispatchEvent(new Event('input'));")
        time.sleep(0.3)
        search_count = driver.execute_script("return displayedRows().length;")
        check("search filter narrows the table", 0 < search_count < 5040, f"{search_count:,} rows")
        sel = top_selection()
        check("search auto-selects top strategy", sel["active"] == sel["top"], f"'{sel['top']}'")
        driver.execute_script("byId('table-search').value = ''; byId('table-search').dispatchEvent(new Event('input'));")

        driver.execute_script("byId('filter-accum').value = 'VEQT'; byId('filter-accum').dispatchEvent(new Event('change'));")
        time.sleep(0.3)
        accum_check = driver.execute_script(
            "const rows = displayedRows(); return {n: rows.length, all: rows.every(r => r.parts[2] === 'VEQT')};"
        )
        check("accumulation filter keeps only VEQT", accum_check["all"], f"{accum_check['n']:,} rows")
        sel = top_selection()
        check("accumulation dropdown auto-selects top strategy", sel["active"] == sel["top"], f"'{sel['top']}'")
        driver.execute_script("byId('filter-accum').value = 'ALL'; byId('filter-accum').dispatchEvent(new Event('change'));")

        driver.execute_script("byId('filter-house').value = 'HOUSE_NONE'; byId('filter-house').dispatchEvent(new Event('change'));")
        time.sleep(0.3)
        house_check = driver.execute_script(
            "const rows = displayedRows(); return {n: rows.length, all: rows.every(r => r.house === 'HOUSE_NONE')};"
        )
        check("house dropdown keeps only renters", house_check["all"], f"{house_check['n']:,} rows")
        sel = top_selection()
        check("house dropdown auto-selects top strategy", sel["active"] == sel["top"], f"'{sel['top']}'")
        renter_badge = driver.execute_script(
            "return document.querySelector('#table-body tr[data-name] td:nth-child(2) .badge-text').textContent.trim();"
        )
        check("renter strategy shows a Renter badge", renter_badge == "Renter", f"'{renter_badge}'")
        driver.execute_script("byId('filter-house').value = 'ALL'; byId('filter-house').dispatchEvent(new Event('change'));")
        time.sleep(0.3)
        fund_badge = driver.execute_script(
            "return document.querySelector('#table-body tr[data-name] td:nth-child(2) .badge-text').textContent.trim();"
        )
        check("buyer strategy shows the house fund",
              fund_badge.startswith("Buyer (") and fund_badge.rstrip(")").split("(")[-1] in ("CASH", "VBAL", "VGRO", "VEQT"),
              f"'{fund_badge}'")

        driver.execute_script(
            "document.querySelector('#seg-mix .seg-pill[data-mix=\"NO_CASH\"]').click();"
        )
        time.sleep(0.3)
        no_cash = driver.execute_script("return displayedRows().every(r => !r.hasCash);")
        check("NO_CASH mix filter removes cash strategies", no_cash)
        sel = top_selection()
        check("NO_CASH filter auto-selects top strategy", sel["active"] == sel["top"], f"'{sel['top']}'")
        driver.execute_script(
            "document.querySelector('#seg-mix .seg-pill[data-mix=\"ALL\"]').click();"
        )
        time.sleep(0.3)
        sel = top_selection()
        check("back to ALL auto-selects top strategy", sel["active"] == sel["top"], f"'{sel['top']}'")

        click_row = driver.execute_script(
            "const tr = document.querySelector('#table-body tr[data-name]');"
            "const name = tr.dataset.name; tr.click(); return {name, kpi: byId('kpi-median').textContent};"
        )
        check("row selection updates KPI cards", click_row["kpi"].startswith("$") and "/mo" in click_row["kpi"], f"{click_row['kpi']}")

        section("LIVE PARAMETER REACTIVITY")
        live = driver.execute_script(
            "byId('inp-careerStartAge').value = '24'; byId('inp-careerStartAge').dispatchEvent(new Event('input'));"
            "byId('inp-retirementAge').value = '56'; byId('inp-retirementAge').dispatchEvent(new Event('input'));"
            "return {tier: byId('lbl-tier1').textContent, thAccum: byId('th-accum').textContent, thBridge: byId('th-bridge').textContent};"
        )
        check("career tier label reacts to career start age", live["tier"] == "Early Career (24 – 32)", f"'{live['tier']}'")
        check("phase table headers react to ages", live["thAccum"] == "Accumulation (24–56)" and live["thBridge"] == "Early Bridge (56–65)", f"'{live['thAccum']}' / '{live['thBridge']}'")
        driver.execute_script(
            "byId('inp-careerStartAge').value = '25'; byId('inp-careerStartAge').dispatchEvent(new Event('input'));"
            "byId('inp-retirementAge').value = '55'; byId('inp-retirementAge').dispatchEvent(new Event('input'));"
        )

        section("CMA TOGGLE (expected vs historical returns)")
        toggle = driver.execute_script(
            "return {checked: byId('inp-use-forward-cmas').checked, model: readModelInputs().useForwardLookingCmas};"
        )
        check("CMA toggle defaults to ON", toggle["checked"] is True and toggle["model"] is True)
        xi_on = driver.execute_script("return state.dynamic.returnModel.xi.slice();")
        driver.execute_script("byId('inp-use-forward-cmas').checked = false;")
        off = driver.execute_script("return readModelInputs().useForwardLookingCmas;")
        check("unchecking flips the model input to historical", off is False)
        run_simulation(driver, args.simulations)
        xi_off = driver.execute_script("return state.dynamic.returnModel.xi.slice();")
        shift = max(abs(a - b) for a, b in zip(xi_on, xi_off))
        check("historical mode changes the calibrated means", shift > 1e-5, f"max xi shift {shift:.2e}")
        driver.execute_script("byId('inp-use-forward-cmas').checked = true;")

        section("SCHEMA INPUT BINDING (all settings tabs)")
        # Regression gate: every settings tab must feed readModelInputs.
        binding = driver.execute_script(
            "const s = schemaInputs();"
            "s['tab-re:6'].value = '3,600';"
            "s['tab-re:8'].value = '7.5';"
            "s['tab-tax:0'].value = '62,000';"
            "s['tab-cma:3'].value = '-1.00';"
            "s['tab-spend:0'].value = '1.0';"
            "const m = readModelInputs();"
            "return {houseStart: m.houseSavingsStartAnnual, houseCap: m.houseSavingsMaxFraction,"
            " meltdown: m.meltdownBracketAnnual, hisa: m.hisaAnnualRealReturn, smile0: m.smileSchedule[0].change};"
        )
        check("tab-re house savings fields bind", binding["houseStart"] == 3600 and abs(binding["houseCap"] - 0.075) < 1e-9,
              f"start {binding['houseStart']} cap {binding['houseCap']}")
        check("tab-tax field binds", binding["meltdown"] == 62000, str(binding["meltdown"]))
        check("tab-cma field binds", abs(binding["hisa"] + 0.01) < 1e-9, str(binding["hisa"]))
        check("tab-spend field binds", abs(binding["smile0"] - 0.01) < 1e-9, str(binding["smile0"]))
        # End-to-end: an edited CMA must flow into the calibrated model.
        driver.execute_script(
            "const s = schemaInputs();"
            "s['tab-cma:0'].value = '8.00';"
            "byId('btn-run-sim').click();"
        )
        deadline = time.time() + 900
        while time.time() < deadline:
            if not driver.execute_script("return !!state.activeRun;") and driver.execute_script(
                "return !!state.results && byId('run-message').textContent.includes('Completed');"
            ):
                break
            time.sleep(2)
        xi_cma8 = driver.execute_script("return state.dynamic.returnModel.xi.slice();")
        check("edited CMA input changes the calibrated model", abs(xi_cma8[0] - xi_on[0]) > 1e-4,
              f"xi[0] {xi_on[0]:.5f} -> {xi_cma8[0]:.5f}")
        driver.execute_script("byId('btn-reset-defaults').click();")

        section("BROWSER CONSOLE")
        logs = driver.get_log("browser")
        severe = [entry for entry in logs if entry["level"] == "SEVERE"]
        for entry in severe[:10]:
            print(f"  [SEVERE] {entry['message'][:250]}")
        print(f"  ({len(logs)} total log entries, {len(severe)} severe)")
        check("no SEVERE console errors", len(severe) == 0)

        section("DIAGNOSTICS & ERROR PATH")
        has_diag = driver.execute_script("return typeof window.dumpDiagnostics === 'function';")
        check("dumpDiagnostics() helper is exposed", has_diag)
        driver.execute_async_script("window.dumpDiagnostics().then(() => arguments[0]());")
        time.sleep(0.5)
        # Deliberate validation failure: a zero starting salary must surface
        # in the UI AND be logged to the console with ENGINE context.
        driver.execute_script(
            "const s = schemaInputs(); s['tab-career:5'].value = '0';"
            "byId('btn-run-sim').click();"
        )
        time.sleep(1.5)
        err_state = driver.execute_script(
            "return {bad: byId('gpu-status').className.includes('bad'), msg: byId('run-message').textContent};"
        )
        check("invalid settings surface an error in the UI",
              err_state["bad"] and "Starting salary must be positive" in err_state["msg"],
              f"'{err_state['msg']}'")
        error_logs = driver.get_log("browser")
        engine_lines = [e for e in error_logs if "ENGINE" in e["message"]]
        check("failure is logged to the console with ENGINE context", len(engine_lines) > 0,
              f"{len(engine_lines)} ENGINE log lines")
        driver.execute_script("byId('btn-reset-defaults').click();")

        print("\n" + "=" * 100)
        print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECKS FAILED")
        return 1 if failures else 0
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())