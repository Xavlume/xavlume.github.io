"""Refresh the first-paint mock snapshot in template_readonly.html.

The standalone page shows a STATIC preview (KPIs + the ranked DATASET table)
before the first GPU run; those numbers live in template_readonly.html and
must be re-baked whenever the simulator defaults change (new sampler, new
calibration, new default sim count).  This script runs the REAL page (the
built index.html, default settings = RQMC Sobol, leverage OFF, 1,000 paths,
seed 42) once in a headless browser and rewrites:

  - the DATASET array (top 13 strategies by CE, with every column the
    template renders),
  - the four KPI values (CE / median / P10 floor / buy age + mortgage).

The values are taken VERBATIM from the page's own rendered table
(displayedRows()), so the baked preview is guaranteed to match the first
real run of the same default settings.  Run ``py -3.14 build_html.py``
afterwards to embed into index.html.

Usage:
    py -3.14 refresh_mock_snapshot.py [--headed] [--simulations 1000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

HTML = ROOT / "index.html"
TEMPLATE = ROOT / "template_readonly.html"
TOP_N = 13


def make_driver(headed: bool = False):
    options = Options()
    options.add_argument("--headless=new" if not headed else "")
    options.add_argument("--enable-unsafe-webgpu")
    options.add_argument("--use-angle=d3d11")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def run_default(driver, simulations: int) -> dict:
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
    # Defaults: leverage OFF, RQMC (default sampler), sim count = page default.
    driver.execute_script(
        "byId('chk-leverage').checked = false; byId('chk-leverage').dispatchEvent(new Event('change'));"
    )
    driver.execute_script(
        "byId('slider-sim-count').value = String(simulationSliderPosition(%d));"
        "byId('slider-sim-count').dispatchEvent(new Event('input'));" % simulations
    )
    driver.find_element(By.ID, "btn-run-sim").click()
    deadline = time.time() + 900
    while time.time() < deadline:
        info = driver.execute_script(
            "return {active: !!state.activeRun, done: !!state.results,"
            " bad: byId('gpu-status').className.includes('bad'),"
            " msg: byId('run-message').textContent};"
        )
        if info["bad"]:
            raise RuntimeError(f"Run failed: {info['msg']}")
        if not info["active"] and info["done"] and "Completed" in info["msg"]:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Timed out waiting for simulation")
    # The page's own rendered leaderboard (CE-sorted, default controls).
    return driver.execute_script(
        "return displayedRows().map(r => ({name: r.name, house: r.house,"
        " accum: r.parts[2], bridge: r.parts[3], post: r.parts[4],"
        " hasCash: r.hasCash, hasGlidepath: r.hasGlidepath,"
        " ce: r.ce, median: r.median, floor: r.floor, p90: r.p90, ui: r.ui,"
        " buyAge: r.buyAge, p90BuyAge: r.p90BuyAge, mortgage: r.mortgage}));"
    )


def js_number(v):
    return "null" if v is None else f"{v:g}"


def build_dataset_js(rows: list) -> str:
    lines = []
    for i, r in enumerate(rows, 1):
        parts = [
            f"rank: {i}",
            f'name: "{r["name"]}"',
            f'house: "{r["house"]}"',
            f'accum: "{r["accum"]}"',
            f'bridge: "{r["bridge"]}"',
            f'post: "{r["post"]}"',
            f"buyAge: {js_number(r['buyAge'])}",
            f"mortgage: {js_number(r['mortgage'])}",
            f"ce: {r['ce']:g}",
            f"median: {r['median']:g}",
            f"floor: {r['floor']:g}",
            f"p90: {r['p90']:g}",
            f"ui: {r['ui']:g}",
            f"hasCash: {'true' if r['hasCash'] else 'false'}",
            f"hasGlide: {'true' if r['hasGlidepath'] else 'false'}",
        ]
        if r["house"] != "HOUSE_NONE":
            parts.append("isBuyer: true")
        lines.append("  { " + ", ".join(parts) + " },")
    return "\n".join(lines)


def money(v: float) -> str:
    return "$" + f"{v:,.0f}"


def patch_template(rows: list) -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    top = rows[0]

    # DATASET block
    start = src.index("const DATASET = [")
    end = src.index("];", start) + 2
    new_block = "const DATASET = [\n" + build_dataset_js(rows) + "\n];"
    src = src[:start] + new_block + src[end:]

    def repl_kpi(kpi_id: str, value_html: str):
        nonlocal src
        marker = f'id="{kpi_id}">'
        i = src.index(marker) + len(marker)
        j = src.index("</div>", i)
        src = src[:i] + value_html + src[j:]

    repl_kpi("kpi-ce", f'{money(top["ce"])}<span style="font-size:14px; font-weight:500; color:var(--text-muted)">/mo</span>')
    repl_kpi("kpi-median", f'{money(top["median"])}<span style="font-size:14px; font-weight:500; color:var(--text-muted)">/mo</span>')
    repl_kpi("kpi-floor", f'{money(top["floor"])}<span style="font-size:14px; font-weight:500; color:var(--text-muted)">/mo</span>')
    repl_kpi("kpi-buy-age", "Age " + (f"{top['buyAge']:.1f}" if top["buyAge"] else "—"))
    if top["mortgage"]:
        src = src.replace(
            'id="kpi-mortgage">', f'id="kpi-mortgage">{money(top["mortgage"])}/mo', 1
        )

    TEMPLATE.write_text(src, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--simulations", type=int, default=1000,
                        help="default sim count used for the snapshot (page default = 1000)")
    args = parser.parse_args()

    driver = make_driver(args.headed)
    try:
        raw = run_default(driver, args.simulations)
    finally:
        driver.quit()
    if len(raw) < TOP_N:
        raise SystemExit(f"Only {len(raw)} rows returned — aborting (nothing written)")
    rows = raw[:TOP_N]
    print(f"Top strategy: {rows[0]['name']}  CE ${rows[0]['ce']:,.0f}/mo  "
          f"median ${rows[0]['median']:,.0f}/mo  floor ${rows[0]['floor']:,.0f}/mo")
    patch_template(rows)
    print(f"Patched {TEMPLATE.name}: {len(rows)} DATASET rows + KPIs "
          f"(CE {money(rows[0]['ce'])}, median {money(rows[0]['median'])}, "
          f"floor {money(rows[0]['floor'])}, buy age "
          f"{rows[0]['buyAge'] if rows[0]['buyAge'] is not None else 'renter'}, "
          f"mortgage {money(rows[0]['mortgage']) if rows[0]['mortgage'] else 'none'})")
    print("Next: py -3.14 build_html.py  (embeds into index.html)")


if __name__ == "__main__":
    main()