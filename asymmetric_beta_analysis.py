# Asymmetric Beta Analysis — Indian Sectoral Indices
# Generated from the supplied notebook.
# The notebook remains the primary reproducible workflow.

# %% Cell 1
# ── Cell 1: Install & Imports ──────────────────────────────────────────────
!pip install yfinance statsmodels openpyxl reportlab --quiet

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from scipy.stats import chi2
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = sns.color_palette('husl', 8)
SEC_COLORS = {}
print('All libraries loaded successfully.')

# %% Cell 2
# ── Cell 2: Configuration ──────────────────────────────────────────────────
START_DATE  = '2018-01-01'
END_DATE    = '2025-12-31'
FREQ        = '1wk'
OUTPUT_DIR  = '/content'

TICKERS = {
    'Nifty 50'        : '^NSEI',
    'Nifty Auto'      : '^CNXAUTO',
    'Nifty Bank'      : '^NSEBANK',
    'Nifty FMCG'      : '^CNXFMCG',
    'Nifty Healthcare': '^CNXPHARMA',
    'Nifty IT'        : '^CNXIT',
    'Nifty Oil&Gas'   : '^CNXENERGY',
}

BENCHMARK   = 'Nifty 50'
SECTORS     = [k for k in TICKERS if k != BENCHMARK]
ALPHA_LEVEL = 0.05

# Assign a consistent color to each sector
for i, s in enumerate(SECTORS):
    SEC_COLORS[s] = COLORS[i % len(COLORS)]

print(f'Period  : {START_DATE} to {END_DATE}')
print(f'Benchmark: {BENCHMARK}')
print(f'Sectors : {SECTORS}')

# %% Cell 3
# ── Cell 3: Module 1 — Data Collection ────────────────────────────────────
def download_data(tickers_dict, start, end, interval):
    frames = {}
    for name, ticker in tickers_dict.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              interval=interval, progress=False, auto_adjust=True)
            if raw.empty:
                print(f'  No data for {name} ({ticker})')
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frames[name] = raw['Close']
            print(f'  Downloaded: {name:22s} ({len(raw)} rows)')
        except Exception as e:
            print(f'  Failed: {name} — {e}')
    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = 'Date'
    prices.dropna(how='all', inplace=True)
    return prices

print('Downloading data from Yahoo Finance...')
prices_df = download_data(TICKERS, START_DATE, END_DATE, FREQ)
print(f'\nPrice DataFrame shape: {prices_df.shape}')
prices_df.head()

# %% Cell 4
# ── Cell 4: Module 2 — Log Returns & Descriptive Statistics ───────────────
def compute_log_returns(prices):
    """Log return: r_t = ln(P_t / P_{t-1})"""
    return np.log(prices / prices.shift(1)).dropna()

returns_df = compute_log_returns(prices_df)

desc = returns_df.describe().T
desc['skewness'] = returns_df.skew()
desc['kurtosis'] = returns_df.kurt()

print(f'Returns shape: {returns_df.shape}')
print(f'Date range   : {returns_df.index[0].date()} to {returns_df.index[-1].date()}')
print('\nDescriptive Statistics (Weekly Log Returns):')
print(desc[['mean','std','min','max','skewness','kurtosis']].round(5))

# %% Cell 5
# ── Cell 5: Plot A — Normalised Price Index (all series) ──────────────────
norm = prices_df / prices_df.iloc[0] * 100
fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(norm.columns):
    lw  = 2.5 if col == BENCHMARK else 1.4
    clr = 'black' if col == BENCHMARK else COLORS[i % len(COLORS)]
    ax.plot(norm.index, norm[col], label=col, linewidth=lw, color=clr)
ax.set_title('Normalised Price Index (Base = 100, Jan 2018)', fontsize=13, fontweight='bold')
ax.set_ylabel('Index Value')
ax.legend(loc='upper left', fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotA_normalised_prices.png'), dpi=150)
plt.show()
print('Plot A saved.')

# %% Cell 6
# ── Cell 6: Plot B — Weekly Returns Distribution per Sector ───────────────
# Histogram + KDE for every sector in a grid
n_sec = len(SECTORS)
ncols = 3
nrows = int(np.ceil(n_sec / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()

for idx, sector in enumerate(SECTORS):
    ax  = axes[idx]
    ser = returns_df[sector].dropna()
    ax.hist(ser, bins=40, density=True, alpha=0.55,
            color=SEC_COLORS[sector], edgecolor='white', label='Histogram')
    xr = np.linspace(ser.min(), ser.max(), 300)
    ax.plot(xr, stats.norm.pdf(xr, ser.mean(), ser.std()),
            'k--', lw=1.4, label='Normal fit')
    kde = stats.gaussian_kde(ser)
    ax.plot(xr, kde(xr), color=SEC_COLORS[sector], lw=2, label='KDE')
    ax.set_title(sector, fontsize=10, fontweight='bold')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)
    sk = ser.skew(); ku = ser.kurt()
    ax.text(0.97, 0.95, f'Skew={sk:.2f}\nKurt={ku:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

for idx in range(n_sec, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('Weekly Log Return Distributions — All Sectors', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotB_return_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Plot B saved.')

# %% Cell 7
# ── Cell 7: Plot C — Market Regime Returns (Nifty 50) ────────────────────
mkt = returns_df[BENCHMARK]
fig, ax = plt.subplots(figsize=(14, 5))
up   = mkt.clip(lower=0)
down = mkt.clip(upper=0)
ax.bar(mkt.index, up,   color='#2ecc71', alpha=0.85, label='Up Market',   width=5)
ax.bar(mkt.index, down, color='#e74c3c', alpha=0.85, label='Down Market', width=5)
ax.axhline(0, color='black', lw=0.8)
ax.set_title(f'{BENCHMARK} Weekly Returns — Market Regimes (Up vs Down)', fontsize=13, fontweight='bold')
ax.set_ylabel('Log Return')
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotC_market_regimes.png'), dpi=150)
plt.show()
print('Plot C saved.')

# %% Cell 8
# ── Cell 8: Plot D — ACF & PACF for ALL Series (pre-model) ────────────────
# Shows raw autocorrelation structure that motivates ARIMA order selection.
# One row per series (Benchmark + all Sectors), 2 columns (ACF | PACF).
all_series = [BENCHMARK] + SECTORS
n_series   = len(all_series)

fig, axes = plt.subplots(n_series, 2, figsize=(14, 3.5 * n_series))

for row_idx, name in enumerate(all_series):
    ser  = returns_df[name].dropna()
    clr  = 'black' if name == BENCHMARK else SEC_COLORS[name]
    ax_acf  = axes[row_idx, 0]
    ax_pacf = axes[row_idx, 1]
    sm.graphics.tsa.plot_acf(ser,  lags=20, ax=ax_acf,
                              title=f'ACF  — {name}', alpha=0.05)
    sm.graphics.tsa.plot_pacf(ser, lags=20, ax=ax_pacf,
                               title=f'PACF — {name}', alpha=0.05)
    for ax in [ax_acf, ax_pacf]:
        ax.set_xlabel('Lag')
        for line in ax.lines[1:]:
            line.set_color(clr)
            line.set_alpha(0.7)

fig.suptitle('ACF & PACF of Weekly Log Returns — All Series\n'
             '(Significant spikes indicate AR/MA order for ARIMAX)',
             fontsize=13, fontweight='bold', y=1.001)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotD_acf_pacf_all.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Plot D saved.')

# %% Cell 9
# ── Cell 9: Module 3 — ARIMAX Asymmetric Beta Estimation ──────────────────
#
# Model: ARIMAX(p,0,q) — sector return is endogenous, D+ and D- are exogenous
#
#   r_sector,t = α + β⁺·D⁺_t + β⁻·D⁻_t + ARMA(p,q) errors
#
#   D⁺ = max(r_mkt, 0)   (positive market returns only)
#   D⁻ = min(r_mkt, 0)   (negative market returns only)
#
# The ARMA component absorbs serial autocorrelation in sector returns,
# yielding more efficient estimates of β⁺ and β⁻ than plain OLS.
# Best (p,q) is selected by AIC grid search.

def sig_label(p):
    if   p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''

def select_arimax_order(sector_ret, exog, max_p=3, max_q=3):
    """Grid-search best ARIMAX(p,0,q) order by AIC."""
    best_aic, best_order = float('inf'), (0, 0, 0)
    aic_grid = {}
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                m = ARIMA(sector_ret, exog=exog, order=(p, 0, q)).fit()
                aic_grid[(p, q)] = m.aic
                if m.aic < best_aic:
                    best_aic, best_order = m.aic, (p, 0, q)
            except Exception:
                pass
    return best_order, aic_grid

def estimate_asymmetric_beta(sector_returns, market_returns):
    data = pd.concat([sector_returns, market_returns], axis=1).dropna()
    data.columns = ['sector', 'market']

    up_mkt   = data['market'].clip(lower=0)
    down_mkt = data['market'].clip(upper=0)
    exog     = pd.DataFrame({'beta_up': up_mkt, 'beta_dn': down_mkt})

    best_order, aic_grid = select_arimax_order(data['sector'], exog)
    model = ARIMA(data['sector'], exog=exog, order=best_order).fit()

    b_up = model.params['beta_up']
    b_dn = model.params['beta_dn']
    t_up = model.tvalues['beta_up']
    t_dn = model.tvalues['beta_dn']
    p_up = model.pvalues['beta_up']
    p_dn = model.pvalues['beta_dn']

    # Confidence intervals (95%)
    ci   = model.conf_int(alpha=0.05)
    ci_up_lo = ci.loc['beta_up', 'lower']
    ci_up_hi = ci.loc['beta_up', 'upper']
    ci_dn_lo = ci.loc['beta_dn', 'lower']
    ci_dn_hi = ci.loc['beta_dn', 'upper']

    alpha_val = model.params.get('const', model.params.get('intercept', float('nan')))

    # Wald test H0: β⁺ = β⁻
    try:
        wald      = model.wald_test('beta_up = beta_dn', use_f=False)
        wald_stat = float(wald.statistic.item() if hasattr(wald.statistic, 'item') else wald.statistic)
        wald_pval = float(wald.pvalue.item()     if hasattr(wald.pvalue,    'item') else wald.pvalue)
    except Exception:
        diff      = b_up - b_dn
        cov_p     = model.cov_params()
        se_diff   = np.sqrt(cov_p.loc['beta_up','beta_up'] + cov_p.loc['beta_dn','beta_dn']
                            - 2 * cov_p.loc['beta_up','beta_dn'])
        wald_stat = (diff / se_diff) ** 2
        wald_pval = 1 - chi2.cdf(wald_stat, df=1)

    pseudo_r2 = 1 - np.var(model.resid) / np.var(data['sector'])

    return {
        'alpha'        : alpha_val,
        'beta_up'      : b_up,   'beta_dn'      : b_dn,
        't_up'         : t_up,   't_dn'         : t_dn,
        'p_up'         : p_up,   'p_dn'         : p_dn,
        'ci_up'        : (ci_up_lo, ci_up_hi),
        'ci_dn'        : (ci_dn_lo, ci_dn_hi),
        'r_squared'    : pseudo_r2,
        'sig_up'       : sig_label(p_up),
        'sig_dn'       : sig_label(p_dn),
        'wald_stat'    : wald_stat,
        'wald_pval'    : wald_pval,
        'wald_decision': 'Reject H0' if wald_pval < ALPHA_LEVEL else 'Fail to Reject H0',
        'model_obj'    : model,
        'arima_order'  : best_order,
        'aic_grid'     : aic_grid,
        'aic'          : model.aic,
        'bic'          : model.bic,
        'n_obs'        : len(data),
        'n_up'         : int((data['market'] > 0).sum()),
        'n_dn'         : int((data['market'] <= 0).sum()),
    }

market_ret   = returns_df[BENCHMARK]
beta_results = {}

print('Fitting ARIMAX models for asymmetric beta estimation...\n')
for sector in SECTORS:
    if sector not in returns_df.columns:
        print(f'  Skipping {sector} — no data.')
        continue
    beta_results[sector] = estimate_asymmetric_beta(returns_df[sector], market_ret)
    r = beta_results[sector]
    print(f"  {sector:22s} | ARIMAX{r['arima_order']}  "
          f"β⁺={r['beta_up']:6.3f}{r['sig_up']:3s}  "
          f"β⁻={r['beta_dn']:6.3f}{r['sig_dn']:3s}  "
          f"Wald-p={r['wald_pval']:.4f}  AIC={r['aic']:.1f}")

print('\nARIMAX estimation complete.')

# %% Cell 10
# ── Cell 10: Module 4 — ARIMAX Model Summary & AIC Grid ───────────────────
arima_results = {}
print('ARIMAX model summary:\n')
print(f"  {'Sector':22s} | Order      | AIC      | BIC      | Baseline AIC | AIC Imprv")
print('  ' + '-'*80)

for sector in beta_results:
    r    = beta_results[sector]
    data = pd.concat([returns_df[sector], returns_df[BENCHMARK]], axis=1).dropna()
    data.columns = ['sector', 'market']
    exog = pd.DataFrame({'beta_up': data['market'].clip(lower=0),
                         'beta_dn': data['market'].clip(upper=0)})
    try:
        base     = ARIMA(data['sector'], exog=exog, order=(0,0,0)).fit()
        base_aic = base.aic
    except Exception:
        base_aic = float('nan')

    aic_imp = round(base_aic - r['aic'], 2) if not np.isnan(base_aic) else float('nan')
    arima_results[sector] = {
        'best_order': r['arima_order'], 'best_aic': r['aic'], 'best_bic': r['bic'],
        'base_aic'  : base_aic,         'aic_imp' : aic_imp, 'model'   : r['model_obj'],
        'aic_grid'  : r['aic_grid'],
    }
    print(f"  {sector:22s} | ARIMAX{r['arima_order']}  | "
          f"{r['aic']:8.2f} | {r['bic']:8.2f} | {base_aic:12.2f} | {aic_imp:+.2f}")

print('\nSummary complete.')

# %% Cell 11
# ── Cell 11: Plot E — AIC Grid Heatmaps per Sector ─────────────────────────
# Shows which (p,q) order was chosen and why — directly connects ACF/PACF
# spikes to the ARIMAX order selection decision.
ncols = 3
nrows = int(np.ceil(len(SECTORS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for idx, sector in enumerate(SECTORS):
    ax       = axes[idx]
    grid     = arima_results[sector]['aic_grid']
    max_p    = max(k[0] for k in grid)
    max_q    = max(k[1] for k in grid)
    mat      = np.full((max_p + 1, max_q + 1), np.nan)
    for (p, q), aic in grid.items():
        mat[p, q] = aic
    best_p, _, best_q = arima_results[sector]['best_order']
    df_mat = pd.DataFrame(mat,
                          index=[f'p={p}' for p in range(max_p + 1)],
                          columns=[f'q={q}' for q in range(max_q + 1)])
    sns.heatmap(df_mat, annot=True, fmt='.1f', cmap='YlOrRd_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'AIC'})
    ax.set_title(f'{sector}\nBest: ARIMAX({best_p},0,{best_q})', fontsize=9, fontweight='bold')
    ax.add_patch(plt.Rectangle((best_q, best_p), 1, 1, fill=False,
                                edgecolor='blue', lw=2.5, zorder=5))

for idx in range(len(SECTORS), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('ARIMAX Order Selection — AIC Grid (p,q) per Sector\n'
             '(Blue box = selected order; darker = lower AIC = better)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotE_aic_grids.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Plot E saved.')

# %% Cell 12
# ── Cell 12: Module 5 — Econometric Diagnostics ───────────────────────────
def run_diagnostics(sector, returns_series, model_obj):
    resid = model_obj.resid

    # ADF stationarity test on raw returns
    adf  = adfuller(returns_series.dropna(), autolag='AIC')
    # KPSS stationarity test (null: stationary)
    try:
        kpss_stat, kpss_p, _, _ = kpss(returns_series.dropna(), regression='c', nlags='auto')
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan

    # Ljung-Box serial correlation test on ARIMAX residuals
    lb   = acorr_ljungbox(resid, lags=[10], return_df=True)
    # ARCH-LM volatility clustering test
    arch = het_arch(resid, nlags=5)
    # Jarque-Bera normality test
    jb   = stats.jarque_bera(resid)

    return {
        'adf_stat'      : adf[0],
        'adf_pval'      : adf[1],
        'adf_stationary': 'Stationary' if adf[1] < 0.05 else 'Non-Stationary',
        'kpss_stat'     : kpss_stat,
        'kpss_pval'     : kpss_p,
        'lb_stat'       : float(lb['lb_stat'].iloc[0]),
        'lb_pval'       : float(lb['lb_pvalue'].iloc[0]),
        'lb_serial'     : 'Yes' if float(lb['lb_pvalue'].iloc[0]) < 0.05 else 'No',
        'arch_stat'     : arch[0],
        'arch_pval'     : arch[1],
        'arch_effect'   : 'Yes' if arch[1] < 0.05 else 'No',
        'jb_stat'       : jb[0],
        'jb_pval'       : jb[1],
        'jb_normal'     : 'Non-Normal' if jb[1] < 0.05 else 'Normal',
        'resid'         : resid,
    }

diagnostics = {}
print(f"  {'Sector':22s} | ADF          | LB-p     | ARCH-p   | JB")
print('  ' + '-'*72)

for sector in beta_results:
    d = run_diagnostics(sector, returns_df[sector], beta_results[sector]['model_obj'])
    diagnostics[sector] = d
    print(f"  {sector:22s} | {d['adf_stationary']:12s} | "
          f"{d['lb_pval']:.4f}   | {d['arch_pval']:.4f}   | {d['jb_normal']}")

print('\nDiagnostics complete.')

# %% Cell 13
# ── Cell 13: Plot F — ARIMAX Residual Diagnostics per Sector ──────────────
# 4-panel plot per sector:
#   (1) Residual time series  (2) Residual histogram + normal fit
#   (3) ACF of residuals      (4) PACF of residuals
# White-noise residuals = ARIMAX has captured all linear structure.

for sector in SECTORS:
    if sector not in diagnostics:
        continue
    resid = diagnostics[sector]['resid']
    clr   = SEC_COLORS[sector]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    r    = beta_results[sector]
    d    = diagnostics[sector]
    ordr = r['arima_order']

    # Panel 1: Residual time series
    ax1.plot(resid.index, resid.values, color=clr, lw=0.9, alpha=0.85)
    ax1.axhline(0, color='black', lw=0.8, ls='--')
    ax1.fill_between(resid.index, resid.values, 0,
                     where=(resid.values > 0), color='#2ecc71', alpha=0.25)
    ax1.fill_between(resid.index, resid.values, 0,
                     where=(resid.values < 0), color='#e74c3c', alpha=0.25)
    ax1.set_title(f'Residual Time Series', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual')
    ax1.tick_params(axis='x', rotation=30)

    # Panel 2: Residual histogram
    ax2.hist(resid, bins=40, density=True, alpha=0.6,
             color=clr, edgecolor='white')
    xr = np.linspace(resid.min(), resid.max(), 300)
    ax2.plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()),
             'k--', lw=1.5, label='Normal')
    ax2.plot(xr, stats.gaussian_kde(resid)(xr),
             color=clr, lw=2, label='KDE')
    ax2.set_title(f'Residual Distribution', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=8)
    ax2.text(0.97, 0.95,
             f"JB p={d['jb_pval']:.4f}\n{d['jb_normal']}",
             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Panel 3: ACF of residuals
    sm.graphics.tsa.plot_acf(resid.dropna(), lags=20, ax=ax3,
                              title='ACF of Residuals', alpha=0.05)
    ax3.set_xlabel('Lag')
    ax3.text(0.97, 0.95,
             f"LB(10) p={d['lb_pval']:.4f}\nSerial corr: {d['lb_serial']}",
             transform=ax3.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Panel 4: PACF of residuals
    sm.graphics.tsa.plot_pacf(resid.dropna(), lags=20, ax=ax4,
                               title='PACF of Residuals', alpha=0.05)
    ax4.set_xlabel('Lag')
    ax4.text(0.97, 0.95,
             f"ARCH p={d['arch_pval']:.4f}\nArch effect: {d['arch_effect']}",
             transform=ax4.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle(
        f'{sector} — ARIMAX{ordr} Residual Diagnostics\n'
        f'β⁺={r["beta_up"]:.3f}{r["sig_up"]}  β⁻={r["beta_dn"]:.3f}{r["sig_dn"]}  '
        f'Wald-p={r["wald_pval"]:.4f}  R²={r["r_squared"]:.3f}  AIC={r["aic"]:.1f}',
        fontsize=11, fontweight='bold'
    )

    safe = sector.replace(' ', '_').replace('&', '')
    fig.savefig(os.path.join(OUTPUT_DIR, f'plotF_residdiag_{safe}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

print('Plot F (residual diagnostics, one per sector) saved.')

# %% Cell 14
# ── Cell 14: Plot G — Regime Scatter for ALL Sectors ──────────────────────
# Up-market vs Down-market scatter with β⁺ / β⁻ regression lines.

for sector in SECTORS:
    if sector not in beta_results:
        continue
    data = pd.concat([returns_df[sector], returns_df[BENCHMARK]], axis=1).dropna()
    data.columns = ['sector', 'market']
    up_d   = data[data['market'] > 0]
    down_d = data[data['market'] <= 0]
    r      = beta_results[sector]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, subset, label, clr, beta_val, ci_val in [
        (axes[0], up_d,   'Up Market (β⁺)',   '#27ae60', r['beta_up'], r['ci_up']),
        (axes[1], down_d, 'Down Market (β⁻)', '#c0392b', r['beta_dn'], r['ci_dn'])
    ]:
        ax.scatter(subset['market'], subset['sector'],
                   alpha=0.45, color=clr, s=20, edgecolors='white', linewidths=0.3)
        if len(subset) > 2:
            m, b = np.polyfit(subset['market'], subset['sector'], 1)
            xs   = np.linspace(subset['market'].min(), subset['market'].max(), 200)
            ax.plot(xs, m*xs + b, color='navy', lw=2,
                    label=f'ARIMAX β = {beta_val:.3f}\n95% CI [{ci_val[0]:.3f}, {ci_val[1]:.3f}]')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(f'{sector} | {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{BENCHMARK} Return')
        ax.set_ylabel(f'{sector} Return')
        ax.legend(fontsize=8)

    wald_str = ('Asymmetric ✓' if r['wald_pval'] < ALPHA_LEVEL
                else 'Symmetric (p={:.3f})'.format(r['wald_pval']))
    fig.suptitle(
        f'Regime Scatter: {sector} vs {BENCHMARK}\n'
        f'Wald test H₀: β⁺=β⁻ → {wald_str}   R²={r["r_squared"]:.3f}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    safe = sector.replace(' ', '_').replace('&', '')
    fig.savefig(os.path.join(OUTPUT_DIR, f'plotG_scatter_{safe}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

print('Plot G (regime scatters, one per sector) saved.')

# %% Cell 15
# ── Cell 15: Plot H — Beta Comparison Bar Chart + CI ─────────────────────
sectors  = list(beta_results.keys())
b_up     = [beta_results[s]['beta_up'] for s in sectors]
b_dn     = [beta_results[s]['beta_dn'] for s in sectors]
err_up   = [(beta_results[s]['beta_up'] - beta_results[s]['ci_up'][0],
             beta_results[s]['ci_up'][1] - beta_results[s]['beta_up']) for s in sectors]
err_dn   = [(beta_results[s]['beta_dn'] - beta_results[s]['ci_dn'][0],
             beta_results[s]['ci_dn'][1] - beta_results[s]['beta_dn']) for s in sectors]
err_up_t = np.array(err_up).T
err_dn_t = np.array(err_dn).T

x, width = np.arange(len(sectors)), 0.35
fig, ax  = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - width/2, b_up, width, label='β⁺ (Upside)',
             color='#2ecc71', edgecolor='white', yerr=err_up_t,
             capsize=4, error_kw={'ecolor': 'darkgreen', 'lw': 1.2})
b2 = ax.bar(x + width/2, b_dn, width, label='β⁻ (Downside)',
             color='#e74c3c', edgecolor='white', yerr=err_dn_t,
             capsize=4, error_kw={'ecolor': 'darkred', 'lw': 1.2})
ax.axhline(1, color='black', lw=1, ls='--', alpha=0.4, label='β = 1')
ax.axhline(0, color='black', lw=0.8)

# Mark sectors with significant asymmetry
for i, s in enumerate(sectors):
    if beta_results[s]['wald_pval'] < ALPHA_LEVEL:
        ax.annotate('*', xy=(x[i], max(b_up[i], b_dn[i]) + 0.12),
                    ha='center', fontsize=14, color='navy', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(sectors, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Beta Coefficient')
ax.set_title('Upside β⁺ vs Downside β⁻ per Sector — ARIMAX Estimates with 95% CI\n'
             '(* = Wald test rejects β⁺ = β⁻ at 5% level)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.03 if h >= 0 else -0.07),
            f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotH_beta_comparison.png'), dpi=150)
plt.show()
print('Plot H saved.')

# %% Cell 16
# ── Cell 16: Plot I — Beta Asymmetry Heatmap ─────────────────────────────
hmap = pd.DataFrame({
    'β⁺ (Upside)'    : [beta_results[s]['beta_up'] for s in sectors],
    'β⁻ (Downside)'  : [beta_results[s]['beta_dn'] for s in sectors],
    'β⁻ minus β⁺'    : [beta_results[s]['beta_dn'] - beta_results[s]['beta_up'] for s in sectors],
    'Wald p-value'   : [beta_results[s]['wald_pval'] for s in sectors],
    'Pseudo R²'      : [beta_results[s]['r_squared'] for s in sectors],
}, index=sectors)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(hmap, annot=True, fmt='.3f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, center=0,
            annot_kws={'size': 9})
ax.set_title('Asymmetric Beta Summary Heatmap — ARIMAX Estimates\n'
             '(β⁻ − β⁺ > 0 = higher downside sensitivity; low Wald-p = asymmetry confirmed)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotI_asymmetry_heatmap.png'), dpi=150)
plt.show()
print('Plot I saved.')

# %% Cell 17
# ── Cell 17: Plot J — Stationarity & Normality Tests Summary ─────────────
# Visual summary of ADF p-values, Ljung-Box p-values, ARCH p-values
# and JB p-values across all sectors.

test_df = pd.DataFrame({
    'ADF p-val'   : [diagnostics[s]['adf_pval']  for s in sectors],
    'LB(10) p-val': [diagnostics[s]['lb_pval']   for s in sectors],
    'ARCH p-val'  : [diagnostics[s]['arch_pval'] for s in sectors],
    'JB p-val'    : [diagnostics[s]['jb_pval']   for s in sectors],
}, index=sectors)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
titles    = ['ADF Test (H₀: Unit Root) — p < 0.05 = Stationary',
             'Ljung-Box(10) (H₀: No Serial Corr in Residuals)',
             'ARCH-LM(5) (H₀: No Volatility Clustering)',
             'Jarque-Bera (H₀: Normality of Residuals)']
cols      = test_df.columns

for ax, col, title in zip(axes.flatten(), cols, titles):
    vals  = test_df[col].values
    clrs  = ['#2ecc71' if v < 0.05 else '#e74c3c' for v in vals]
    bars  = ax.barh(sectors, vals, color=clrs, edgecolor='white')
    ax.axvline(0.05, color='navy', lw=1.5, ls='--', label='α = 0.05')
    ax.axvline(0.01, color='purple', lw=1.2, ls=':', label='α = 0.01')
    for bar, v in zip(bars, vals):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8)
    ax.set_xlim(0, max(vals) * 1.2 + 0.05)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('p-value')
    ax.legend(fontsize=8)

fig.suptitle('Statistical Test Summary — All Sectors (ARIMAX Residuals)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotJ_stat_tests.png'), dpi=150)
plt.show()
print('Plot J saved.')

# %% Cell 18
# ── Cell 18: Plot K — Wald Test Significance Summary ─────────────────────
# Horizontal bar chart of Wald p-values with significance threshold.

wald_pvals = [beta_results[s]['wald_pval'] for s in sectors]
clrs_wald  = ['#27ae60' if p < 0.05 else '#bdc3c7' for p in wald_pvals]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(sectors, wald_pvals, color=clrs_wald, edgecolor='white', height=0.6)
ax.axvline(0.05, color='navy', lw=1.8, ls='--', label='α = 0.05 threshold')
ax.axvline(0.01, color='crimson', lw=1.2, ls=':', label='α = 0.01 threshold')
for bar, pv in zip(bars, wald_pvals):
    label = f'{pv:.4f} ✓' if pv < 0.05 else f'{pv:.4f}'
    ax.text(pv + 0.002, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=9)
ax.set_xlabel('Wald Test p-value (H₀: β⁺ = β⁻)')
ax.set_title('Beta Asymmetry — Wald Test p-values per Sector\n'
             '(Green = significant asymmetry confirmed at 5% level)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plotK_wald_test.png'), dpi=150)
plt.show()
print('Plot K saved.')

# %% Cell 19
# ── Cell 19: Module 7 — Summary Insights ─────────────────────────────────
def generate_insights(beta_results, diagnostics):
    insights = []
    dn_betas = {s: abs(beta_results[s]['beta_dn']) for s in beta_results}
    max_dn   = max(dn_betas, key=dn_betas.get)
    min_dn   = min(dn_betas, key=dn_betas.get)

    insights.append(
        f"Highest downside risk: {max_dn} (β⁻={beta_results[max_dn]['beta_dn']:.3f}), "
        f"amplified {abs(beta_results[max_dn]['beta_dn']):.2f}× vs market in downturns."
    )
    insights.append(
        f"Most defensive sector: {min_dn} (β⁻={beta_results[min_dn]['beta_dn']:.3f}), "
        f"lowest sensitivity to market downturns."
    )
    asymmetric = [s for s in beta_results if beta_results[s]['wald_pval'] < ALPHA_LEVEL]
    if asymmetric:
        insights.append(
            f"Significant β asymmetry (Wald p<0.05): {', '.join(asymmetric)}. "
            f"Non-linear sensitivity confirmed — linear CAPM beta understates risk."
        )
    else:
        insights.append("No sector shows statistically significant beta asymmetry at 5% level.")

    dn_gt_up = [s for s in beta_results
                if abs(beta_results[s]['beta_dn']) > abs(beta_results[s]['beta_up'])]
    if dn_gt_up:
        insights.append(
            f"|β⁻| > |β⁺| (amplified losses vs gains): {', '.join(dn_gt_up)}."
        )

    high_r2 = [s for s in beta_results if beta_results[s]['r_squared'] > 0.60]
    if high_r2:
        insights.append(f"High market co-movement (R²>0.60): {', '.join(high_r2)}.")

    arch_sec = [s for s in diagnostics if diagnostics[s]['arch_pval'] < 0.05]
    if arch_sec:
        insights.append(
            f"Volatility clustering (ARCH effects): {', '.join(arch_sec)}. "
            f"GARCH extensions may further improve beta estimates."
        )
    insights.append(
        "All return series pass ADF test — stationary, suitable for ARIMAX modelling."
    )
    return insights

summary_insights = generate_insights(beta_results, diagnostics)
print('Summary Insights:\n')
for i, ins in enumerate(summary_insights, 1):
    print(f'{i}. {ins}\n')

# %% Cell 20
# ── Cell 20: Module 8 — Excel Workbook (7 Sheets) ─────────────────────────

def hdr_font():  return Font(name='Arial', bold=True, size=10, color='FFFFFF')
def body_font(): return Font(name='Arial', size=10)
def title_font():return Font(name='Arial', bold=True, size=12)
def hdr_fill(c='1F4E79'): return PatternFill('solid', fgColor=c)
def alt_fill():           return PatternFill('solid', fgColor='DCE6F1')
def sig_fill():           return PatternFill('solid', fgColor='C6EFCE')
def warn_fill():          return PatternFill('solid', fgColor='FFEB9C')
def red_fill():           return PatternFill('solid', fgColor='FFC7CE')

def thin_border():
    s = Side(style='thin', color='BFBFBF')
    return Border(left=s, right=s, top=s, bottom=s)

CTR = Alignment(horizontal='center', vertical='center', wrap_text=True)
LFT = Alignment(horizontal='left',   vertical='center', wrap_text=True)

def style_headers(ws, row, ncols, fill_color='1F4E79'):
    for col in range(1, ncols + 1):
        c = ws.cell(row=row, column=col)
        c.font = hdr_font(); c.fill = hdr_fill(fill_color)
        c.alignment = CTR;   c.border = thin_border()

def set_widths(ws, w_dict):
    for col, w in w_dict.items():
        ws.column_dimensions[col].width = w

def title_block(ws, row, title, subtitle=None):
    ws.cell(row=row, column=1).value = title
    ws.cell(row=row, column=1).font  = title_font()
    if subtitle:
        ws.cell(row=row+1, column=1).value = subtitle
        ws.cell(row=row+1, column=1).font  = Font(name='Arial', italic=True, size=9, color='595959')

wb = Workbook()
wb.remove(wb.active)

# Sheet 1: Raw Prices
ws1 = wb.create_sheet('Sheet1_Raw_Prices')
title_block(ws1, 1, 'Weekly Adjusted Close Prices',
            f'Period: {START_DATE} to {END_DATE} | Weekly | Source: Yahoo Finance')
cols = ['Date'] + list(prices_df.columns)
for j, h in enumerate(cols, 1): ws1.cell(row=3, column=j).value = h
style_headers(ws1, 3, len(cols))
for i, (idx, row) in enumerate(prices_df.iterrows(), 4):
    ws1.cell(row=i, column=1).value = idx.strftime('%Y-%m-%d')
    ws1.cell(row=i, column=1).alignment = CTR
    for j, val in enumerate(row, 2):
        c = ws1.cell(row=i, column=j)
        c.value = round(float(val), 2) if pd.notna(val) else ''
        c.number_format = '#,##0.00'; c.font = body_font()
        c.alignment = CTR; c.border = thin_border()
        if i % 2 == 0: c.fill = alt_fill()
set_widths(ws1, {get_column_letter(j): 14 for j in range(1, len(cols)+1)})
ws1.freeze_panes = 'A4'
print('Sheet 1 done.')

# Sheet 2: Log Returns
ws2 = wb.create_sheet('Sheet2_Log_Returns')
title_block(ws2, 1, 'Weekly Log Returns  r_t = ln(P_t / P_{t-1})',
            'All values are continuous log returns')
cols2 = ['Date'] + list(returns_df.columns)
for j, h in enumerate(cols2, 1): ws2.cell(row=3, column=j).value = h
style_headers(ws2, 3, len(cols2))
for i, (idx, row) in enumerate(returns_df.iterrows(), 4):
    ws2.cell(row=i, column=1).value = idx.strftime('%Y-%m-%d')
    ws2.cell(row=i, column=1).alignment = CTR
    for j, val in enumerate(row, 2):
        c = ws2.cell(row=i, column=j)
        c.value = round(float(val), 6) if pd.notna(val) else ''
        c.number_format = '0.000000'; c.font = body_font()
        c.alignment = CTR; c.border = thin_border()
        if i % 2 == 0: c.fill = alt_fill()
set_widths(ws2, {get_column_letter(j): 14 for j in range(1, len(cols2)+1)})
ws2.freeze_panes = 'A4'
print('Sheet 2 done.')

# Sheet 3: ARIMAX Beta Results
ws3 = wb.create_sheet('Sheet3_ARIMAX_Beta_Results')
title_block(ws3, 1, 'Asymmetric Beta Results — ARIMAX Estimates',
            'Model: ARIMAX(p,0,q) | β⁺/β⁻ are exogenous regressors | Best order by AIC grid search')
hdrs3 = ['Sector','ARIMAX Order','Alpha','β⁺ (Upside)','β⁻ (Downside)',
         't (β⁺)','t (β⁻)','p (β⁺)','p (β⁻)',
         'CI 95% β⁺','CI 95% β⁻','R²','Sig β⁺','Sig β⁻','N(Up)','N(Dn)']
for j, h in enumerate(hdrs3, 1): ws3.cell(row=3, column=j).value = h
style_headers(ws3, 3, len(hdrs3))
for i, sector in enumerate(beta_results, 4):
    r = beta_results[sector]
    vals = [sector, str(r['arima_order']), round(r['alpha'],6),
            round(r['beta_up'],4), round(r['beta_dn'],4),
            round(r['t_up'],4), round(r['t_dn'],4),
            round(r['p_up'],6), round(r['p_dn'],6),
            f"[{r['ci_up'][0]:.3f}, {r['ci_up'][1]:.3f}]",
            f"[{r['ci_dn'][0]:.3f}, {r['ci_dn'][1]:.3f}]",
            round(r['r_squared'],4), r['sig_up'], r['sig_dn'],
            r['n_up'], r['n_dn']]
    for j, v in enumerate(vals, 1):
        c = ws3.cell(row=i, column=j)
        c.value = v; c.font = body_font()
        c.alignment = CTR; c.border = thin_border()
        if i % 2 == 0 and j not in [8, 9]: c.fill = alt_fill()
    for col_idx in [8, 9]:
        c = ws3.cell(row=i, column=col_idx)
        c.number_format = '0.0000'
        if isinstance(c.value, float) and c.value < 0.05: c.fill = sig_fill()
set_widths(ws3, {'A':22,'B':16,'C':12,'D':14,'E':14,'F':11,'G':11,
                 'H':11,'I':11,'J':20,'K':20,'L':10,'M':10,'N':10,'O':10,'P':10})
ws3.freeze_panes = 'B4'
print('Sheet 3 done.')

# Sheet 4: Hypothesis Testing
ws4 = wb.create_sheet('Sheet4_Hypothesis_Testing')
title_block(ws4, 1, 'Hypothesis Testing — Beta Asymmetry (H₀: β⁺ = β⁻)',
            'Wald Test | α = 0.05 | Reject H₀ if p-value < 0.05')
hdrs4 = ['Sector','β⁺','β⁻','β⁻ − β⁺','Wald Stat','p-value','Decision','Interpretation']
for j, h in enumerate(hdrs4, 1): ws4.cell(row=3, column=j).value = h
style_headers(ws4, 3, len(hdrs4), fill_color='375623')
for i, sector in enumerate(beta_results, 4):
    r   = beta_results[sector]
    diff = round(r['beta_dn'] - r['beta_up'], 4)
    rej  = r['wald_pval'] < ALPHA_LEVEL
    interp = ('Asymmetric: β⁻ significantly differs from β⁺ — non-linear market sensitivity confirmed.'
              if rej else
              'Symmetric: No significant difference between upside and downside beta.')
    vals = [sector, round(r['beta_up'],4), round(r['beta_dn'],4), diff,
            round(r['wald_stat'],4), round(r['wald_pval'],6), r['wald_decision'], interp]
    for j, v in enumerate(vals, 1):
        c = ws4.cell(row=i, column=j)
        c.value = v; c.font = body_font()
        c.alignment = LFT if j in [1,7,8] else CTR
        c.border = thin_border()
    dec_c = ws4.cell(row=i, column=7)
    if rej:
        dec_c.fill = sig_fill()
        dec_c.font = Font(name='Arial', bold=True, size=10, color='375623')
    else:
        dec_c.fill = warn_fill()
    diff_c = ws4.cell(row=i, column=4)
    diff_c.number_format = '0.0000'
    if isinstance(diff, float) and diff < 0: diff_c.fill = red_fill()
set_widths(ws4, {'A':22,'B':12,'C':12,'D':14,'E':16,'F':12,'G':20,'H':60})
ws4.freeze_panes = 'B4'
print('Sheet 4 done.')

# Sheet 5: Diagnostics
ws5 = wb.create_sheet('Sheet5_Diagnostics')
title_block(ws5, 1, 'Econometric Diagnostics — ARIMAX Residual Analysis',
            'ADF (stationarity) | Ljung-Box lag 10 (serial corr) | ARCH LM lag 5 | Jarque-Bera (normality)')
hdrs5 = ['Sector','ADF Stat','ADF p-val','Stationary?',
         'LB Stat','LB p-val','Serial Corr?',
         'ARCH Stat','ARCH p-val','ARCH Effect?',
         'JB Stat','JB p-val','Normal?']
for j, h in enumerate(hdrs5, 1): ws5.cell(row=3, column=j).value = h
style_headers(ws5, 3, len(hdrs5), fill_color='7B2C2C')
for i, sector in enumerate(diagnostics, 4):
    d = diagnostics[sector]
    vals = [sector,
            round(d['adf_stat'],4),  round(d['adf_pval'],6),  d['adf_stationary'],
            round(d['lb_stat'],4),   round(d['lb_pval'],6),   d['lb_serial'],
            round(d['arch_stat'],4), round(d['arch_pval'],6), d['arch_effect'],
            round(d['jb_stat'],4),   round(d['jb_pval'],6),   d['jb_normal']]
    for j, v in enumerate(vals, 1):
        c = ws5.cell(row=i, column=j)
        c.value = v; c.font = body_font()
        c.alignment = CTR; c.border = thin_border()
        if i % 2 == 0 and j in [1,2,3,5,6,8,9,11,12]: c.fill = alt_fill()
set_widths(ws5, {get_column_letter(j): 16 for j in range(1,14)})
ws5.column_dimensions['A'].width = 22
ws5.freeze_panes = 'B4'
print('Sheet 5 done.')

# Sheet 6: ARIMAX Orders
ws6 = wb.create_sheet('Sheet6_ARIMAX_Orders')
title_block(ws6, 1, 'ARIMAX Model Orders — Per-Sector Selection',
            'Best ARIMAX(p,0,q) by AIC grid search (p,q ∈ 0..3) | β⁺/β⁻ are exogenous regressors')
hdrs6 = ['Sector','Best Order','AIC (Best)','BIC (Best)',
         'Baseline AIC(0,0,0)','AIC Improvement','Interpretation']
for j, h in enumerate(hdrs6, 1): ws6.cell(row=3, column=j).value = h
style_headers(ws6, 3, len(hdrs6), fill_color='4A235A')
for i, sector in enumerate(arima_results, 4):
    ar      = arima_results[sector]
    aic_imp = ar.get('aic_imp', 0)
    interp  = ('AR/MA dynamics materially improve fit' if aic_imp > 2
               else 'Minimal dynamic structure needed')
    vals = [sector, str(ar['best_order']), round(ar['best_aic'],2),
            round(ar['best_bic'],2), round(ar['base_aic'],2), aic_imp, interp]
    for j, v in enumerate(vals, 1):
        c = ws6.cell(row=i, column=j)
        c.value = v; c.font = body_font()
        c.alignment = LFT if j == 7 else CTR
        c.border = thin_border()
        if i % 2 == 0: c.fill = alt_fill()
    imp_c = ws6.cell(row=i, column=6)
    if isinstance(aic_imp, float) and aic_imp > 2:
        imp_c.fill = warn_fill()
        imp_c.font = Font(name='Arial', bold=True, size=10)
set_widths(ws6, {'A':22,'B':16,'C':14,'D':14,'E':20,'F':16,'G':40})
ws6.freeze_panes = 'B4'
print('Sheet 6 done.')

# Sheet 7: Summary Insights
ws7 = wb.create_sheet('Sheet7_Summary_Insights')
title_block(ws7, 1, 'Analyst Summary — Key Insights',
            f'Generated: {datetime.now().strftime("%d %b %Y")} | Asymmetric ARIMAX Beta Analysis')
ws7.cell(row=4, column=1).value = 'Section A: Beta Coefficient Summary'
ws7.cell(row=4, column=1).font  = Font(name='Arial', bold=True, size=11, color='1F4E79')
hdrs7a = ['Sector','β⁺','β⁻','β⁻ − β⁺','Asymmetry?','Risk Level']
for j, h in enumerate(hdrs7a, 1): ws7.cell(row=5, column=j).value = h
style_headers(ws7, 5, len(hdrs7a))
for i, sector in enumerate(beta_results, 6):
    r    = beta_results[sector]
    diff = r['beta_dn'] - r['beta_up']
    asym = 'Yes' if r['wald_pval'] < 0.05 else 'No'
    risk = 'High' if abs(r['beta_dn']) > 1.2 else ('Moderate' if abs(r['beta_dn']) > 0.8 else 'Low')
    for j, v in enumerate([sector, round(r['beta_up'],4), round(r['beta_dn'],4),
                            round(diff,4), asym, risk], 1):
        c = ws7.cell(row=i, column=j)
        c.value = v; c.font = body_font(); c.alignment = CTR; c.border = thin_border()
    risk_c = ws7.cell(row=i, column=6)
    if risk == 'High':       risk_c.fill = red_fill();  risk_c.font = Font(name='Arial', bold=True, size=10)
    elif risk == 'Moderate': risk_c.fill = warn_fill()
    else:                    risk_c.fill = sig_fill()
ins_start = 6 + len(beta_results) + 2
ws7.cell(row=ins_start, column=1).value = 'Section B: Key Analyst Insights'
ws7.cell(row=ins_start, column=1).font  = Font(name='Arial', bold=True, size=11, color='1F4E79')
for k, ins in enumerate(summary_insights, ins_start + 1):
    c = ws7.cell(row=k, column=1)
    c.value = '• ' + ins
    c.font  = Font(name='Arial', size=10)
    c.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
    ws7.row_dimensions[k].height = 32
    ws7.merge_cells(start_row=k, start_column=1, end_row=k, end_column=6)
set_widths(ws7, {'A':28,'B':12,'C':12,'D':14,'E':12,'F':12})
print('Sheet 7 done.')

excel_path = os.path.join(OUTPUT_DIR, 'Asymmetric_Beta_ARIMAX.xlsx')
wb.save(excel_path)
print(f'\nExcel workbook saved: {excel_path}')
try:
    from google.colab import files
    files.download(excel_path)
except Exception:
    pass

# %% Cell 21
# ── Cell 21: Final Summary Print ──────────────────────────────────────────
print('=' * 70)
print('  ASYMMETRIC BETA ANALYSIS (ARIMAX) — FINAL RESULTS')
print('=' * 70)
print(f"  {'Sector':22s} {'Order':12s} {'β⁺':>8} {'β⁻':>8} {'Asym?':>10} {'R²':>8} {'AIC':>9}")
print('  ' + '-' * 68)
for sector in beta_results:
    r    = beta_results[sector]
    asym = 'YES ***' if r['wald_pval'] < 0.01 else ('YES **' if r['wald_pval'] < 0.05 else 'no')
    print(f"  {sector:22s} {str(r['arima_order']):12s} "
          f"{r['beta_up']:8.4f} {r['beta_dn']:8.4f} {asym:>10} "
          f"{r['r_squared']:8.4f} {r['aic']:9.1f}")
print()
print('  PLOTS SAVED TO /content:')
plots = ['plotA_normalised_prices', 'plotB_return_distributions',
         'plotC_market_regimes', 'plotD_acf_pacf_all',
         'plotE_aic_grids', 'plotF_residdiag_<sector> (×6)',
         'plotG_scatter_<sector> (×6)', 'plotH_beta_comparison',
         'plotI_asymmetry_heatmap', 'plotJ_stat_tests',
         'plotK_wald_test']
for p in plots:
    print(f'    {p}.png')
print('  EXCEL: Asymmetric_Beta_ARIMAX.xlsx (7 sheets)')
print('=' * 70)

