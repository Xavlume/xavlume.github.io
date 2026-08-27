import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Set start date to 1994-01-01 to capture full 30+ year history
START = '1994-01-01'

# We include both XIU.TO (Total Return, starts Nov 1999) and ^GSPTSE (Price Return, starts pre-1994)
download_tickers = ['VTSMX', 'XIU.TO', '^GSPTSE', 'VEURX', 'VPACX', 'VEIEX', 'VBMFX', 'XBB.TO']
targets = ['VEQT.TO', 'VGRO.TO', 'VBAL.TO']

print("--- Downloading Daily Historical Data via yfinance ---")
raw_data = yf.download(download_tickers + targets, start=START, auto_adjust=True)
close_prices = raw_data['Close']
if isinstance(close_prices.columns, pd.MultiIndex):
    close_prices.columns = close_prices.columns.get_level_values(1)

# Diagnostic: Print earliest available daily data date
print("\n--- Earliest Available Daily Data Dates ---")
for col in close_prices.columns:
    first_valid = close_prices[col].dropna().index[0]
    print(f"Ticker - {col:<10}: {first_valid.date()}")
print("-" * 64 + "\n")

# Resample to month-end prices and calculate monthly percentage returns
monthly_prices = close_prices.resample('ME').last()
monthly_returns = monthly_prices.pct_change()

# -------------------------------------------------------------------------
# Step 1: Calibrate and Backfill Canadian Equity Total Return (XIU.TO)
# -------------------------------------------------------------------------
# Overlap period between XIU.TO (Total Return) and ^GSPTSE (Price Index)
tsx_overlap_mask = monthly_returns['XIU.TO'].notna() & monthly_returns['^GSPTSE'].notna()
tsx_overlap = monthly_returns[tsx_overlap_mask]

slope_tsx, intercept_tsx, r_val_tsx, _, _ = linregress(
    tsx_overlap['^GSPTSE'],
    tsx_overlap['XIU.TO']
)

print("--- Canadian Equity Total Return Calibration ---")
print(f"Calibrated mapping of ^GSPTSE -> XIU.TO ({len(tsx_overlap)} months of overlap):")
print(f"  Formula: XIU_return = {intercept_tsx:.6f} (Dividend Yield) + {slope_tsx:.4f} * GSPTSE_return")
print(f"  R-squared correlation strength: {r_val_tsx**2:.4f}")
print("-" * 64 + "\n")

# Backfill XIU.TO from 1994 to 1999 using calibrated dividend-adjusted returns
cad_equity_series = monthly_returns['XIU.TO'].copy()
missing_xiu = cad_equity_series.isna() & monthly_returns['^GSPTSE'].notna()
cad_equity_series.loc[missing_xiu] = (
    intercept_tsx + slope_tsx * monthly_returns.loc[missing_xiu, '^GSPTSE']
)

# -------------------------------------------------------------------------
# Step 2: Calibrate and Backfill Canadian Bonds (XBB.TO vs Local VBMFX)
# -------------------------------------------------------------------------
# We regress XBB.TO against LOCAL USD VBMFX returns to capture duration/rate cycles
# without injecting unhedged USD/CAD FX volatility into domestic bonds.
bond_overlap_mask = monthly_returns['XBB.TO'].notna() & monthly_returns['VBMFX'].notna()
bond_overlap = monthly_returns[bond_overlap_mask]

slope_bond, intercept_bond, r_val_bond, _, _ = linregress(
    bond_overlap['VBMFX'],
    bond_overlap['XBB.TO']
)

print("--- Canadian Aggregate Bond Calibration ---")
print(f"Calibrated mapping of Local VBMFX -> XBB.TO ({len(bond_overlap)} months of overlap):")
print(f"  Formula: XBB_return = {intercept_bond:.6f} + {slope_bond:.4f} * VBMFX_local_return")
print(f"  R-squared correlation strength: {r_val_bond**2:.4f}")
print("-" * 64 + "\n")

cad_bond_series = monthly_returns['XBB.TO'].copy()
missing_xbb = cad_bond_series.isna() & monthly_returns['VBMFX'].notna()
cad_bond_series.loc[missing_xbb] = (
    intercept_bond + slope_bond * monthly_returns.loc[missing_xbb, 'VBMFX']
)

# -------------------------------------------------------------------------
# Step 3: FX Conversion for Unhedged Foreign Equities
# -------------------------------------------------------------------------
fx = pd.read_csv('dexcaus.csv', parse_dates=['observation_date'], index_col='observation_date')
fx = fx['DEXCAUS'].replace('.', pd.NA).astype(float).dropna()
fx_m = fx.resample('ME').last().ffill()
fx_m = fx_m.reindex(monthly_returns.index).ffill()
fx_ret = fx_m.pct_change().dropna()

# Clean proxy DataFrame
proxies_clean = pd.DataFrame(index=monthly_returns.index)
proxies_clean['TSX'] = cad_equity_series
proxies_clean['BONDS'] = cad_bond_series

# Convert USD unhedged foreign equities to CAD returns
foreign_usd_proxies = ['VTSMX', 'VEURX', 'VPACX', 'VEIEX']
for p in foreign_usd_proxies:
    aligned = pd.concat([monthly_returns[p], fx_ret], axis=1).dropna()
    proxies_clean[p] = (1 + aligned.iloc[:, 0]) * (1 + aligned.iloc[:, 1]) - 1

proxies_clean = proxies_clean.dropna()

# -------------------------------------------------------------------------
# Step 4: Real Return Conversion using Canadian CPI
# -------------------------------------------------------------------------
cpi = pd.read_csv('canadian_cpi.csv', parse_dates=['Date'], index_col='Date')
inflation = cpi['inflation_mom'].dropna()
inflation = inflation[inflation.index >= START]
inflation = inflation[~inflation.index.duplicated(keep='first')]

infl_monthly = inflation.copy()
infl_monthly.index = infl_monthly.index + pd.offsets.MonthEnd(0)
infl_monthly = infl_monthly[~infl_monthly.index.duplicated(keep='first')]

infl_aligned = infl_monthly.reindex(proxies_clean.index).ffill()
# Months past the CPI file's last observation (the committed
# canadian_cpi.csv ends 2023-11) are deflated at the Bank of Canada's 2%
# control-range midpoint - the standard forward-looking inflation anchor
# (realized 2024-2026 Canadian CPI ran ~2-2.5%/yr). The old code
# forward-filled the last observed monthly print (~0.76%/yr), which
# overstated recent real returns; a trailing-average fill would have reused
# the 2022 spike instead. Extend canadian_cpi.csv to use realized inflation.
INFLATION_TARGET_MONTHLY = 0.02 / 12.0
tail_mask = infl_aligned.index > infl_monthly.index[-1]
if tail_mask.any():
    tail_count = int(tail_mask.sum())
    print(f"WARNING: CPI file ends {infl_monthly.index[-1].date()}; deflating "
          f"the last {tail_count} month(s) at the Bank of Canada 2% target "
          f"({INFLATION_TARGET_MONTHLY:.4%}/mo).")
    infl_aligned.loc[tail_mask] = INFLATION_TARGET_MONTHLY

# Calculate real returns for proxies
proxy_ret_real = proxies_clean.copy()
for col in proxy_ret_real.columns:
    proxy_ret_real[col] = (1 + proxy_ret_real[col]) / (1 + infl_aligned) - 1
proxy_ret_real = proxy_ret_real.dropna()

# Calculate real returns for target Vanguard asset allocation ETFs
target_ret = monthly_returns[targets].reindex(proxy_ret_real.index).copy()
for col in target_ret.columns:
    target_ret[col] = (1 + target_ret[col]) / (1 + infl_aligned) - 1

# -------------------------------------------------------------------------
# Step 5: Optimization & Synthetic Asset Stitching (2019 - Present Overlap)
# -------------------------------------------------------------------------
# The proxy funds (VTSMX, VEURX, VPACX, VEIEX, XIU, XBB) are NOT the exact
# holdings of VEQT/VGRO/VBAL, so the SLSQP fit on the 2019+ overlap maps the
# proxies to each target fund as best as possible, and the fitted weights are
# applied to the pre-2019 history. Deliberate approximation - do not replace
# the fitted weights with the published mandate allocations: the proxies'
# tracking differences are exactly what the fit corrects for.
overlap_start = '2019-01-01'
overlap = proxy_ret_real.loc[overlap_start:].join(target_ret.loc[overlap_start:]).dropna()
proxy_cols = ['VTSMX', 'TSX', 'VEURX', 'VPACX', 'VEIEX', 'BONDS']

optimized_weights = {}
for etf in targets:
    clean_name = etf.replace('.TO', '')

    def objective(w):
        synthetic = overlap[proxy_cols] @ w
        return np.sqrt(np.mean((synthetic - overlap[etf]) ** 2))

    bounds = [(0, 1)] * len(proxy_cols)
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    result = minimize(
        objective,
        x0=np.ones(len(proxy_cols)) / len(proxy_cols),
        bounds=bounds,
        constraints=cons,
        method='SLSQP',
    )
    optimized_weights[clean_name] = result.x
    print(f"{clean_name} weights: {dict(zip(proxy_cols, np.round(result.x, 4)))}")

# Generate backfilled synthetic history
synthetic_returns = pd.DataFrame(index=proxy_ret_real.index)
for etf_name, w in optimized_weights.items():
    synthetic_returns[etf_name] = proxy_ret_real[proxy_cols] @ w

# Stitch synthetic history to actual ETF returns post-2019
stitched = synthetic_returns.copy()
for etf in targets:
    ticker = etf.replace('.TO', '')
    mask = stitched.index >= overlap_start
    stitched.loc[mask, ticker] = target_ret.loc[mask, etf].values

# -------------------------------------------------------------------------
# Step 6: Export Clean Base Assets Only (VEQT, VGRO, VBAL)
# -------------------------------------------------------------------------
# Leveraged series (VEQT1.5, VEQT2) are generated dynamically on-the-fly inside 
# the Monte Carlo engine / web UI, allowing dynamic borrowing rate sweeps.
stitched = stitched[['VEQT', 'VGRO', 'VBAL']]

# Compute cumulative real growth
growth = (1 + stitched).cumprod()
growth.index.name = 'Date'
growth.to_csv('downloaded_prices.csv')

print("\n--- Download and Backfill Complete ---")
print("downloaded_prices.csv written successfully.")
print(f"Date range: {growth.index[0].date()} to {growth.index[-1].date()} ({len(growth)} months)")
print(f"Final real cumulative growth index:\n{growth.iloc[-1].round(4).to_string()}")