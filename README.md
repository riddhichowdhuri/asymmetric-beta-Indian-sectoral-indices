# Asymmetric Beta Analysis — Indian Sectoral Indices

Empirical finance project examining **non-linear/asymmetric market sensitivity** of six Indian sectoral indices relative to the **Nifty 50** using weekly data from **January 2018 to December 2025**.

## Project question

Do Indian sectoral indices respond differently to positive versus negative market movements?

The analysis estimates:

`R_sector,t = α + β⁺ D⁺_t + β⁻ D⁻_t + error_t`

where:

- `D⁺_t = max(R_market,t, 0)` captures the up-market regime.
- `D⁻_t = min(R_market,t, 0)` captures the down-market regime.
- `β⁺` measures sector sensitivity during positive market weeks.
- `β⁻` measures sector sensitivity during negative market weeks.

The supplied notebook extends this framework using an **ARIMAX(p,0,q)** error structure with AIC-based order selection.

## Sectors

- Nifty Auto
- Nifty Bank
- Nifty FMCG
- Nifty Healthcare
- Nifty IT
- Nifty Oil & Gas

## Key results

The supplied OLS workbook finds **no statistically significant beta asymmetry at the 5% level** for any sector.

Highlights:

- Highest downside beta: **Nifty Bank (β⁻ = 1.1563)**
- Lowest downside beta: **Nifty FMCG (β⁻ = 0.5528)**
- Largest positive `(β⁻ − β⁺)` spread: **Nifty IT (+0.3455)**
- Highest OLS R²: **Nifty Bank (0.7829)**
- Nifty IT has the closest evidence to asymmetry at the 5% threshold, but its Wald p-value is **0.0803**, so it is not significant at 5%.

See [`FINDINGS.md`](FINDINGS.md) for the full interpretation.

## Repository structure

```text
.
├── FINDINGS.md
├── README.md
├── requirements.txt
├── notebooks/
│   └── asymmetric_beta_analysis.ipynb
├── src/
│   └── asymmetric_beta_analysis.py
├── data/
│   └── README.md
├── results/
│   ├── asymmetric_beta_dashboard.html
│   ├── Asymmetric_Beta_Analysis_OLS.xlsx
│   ├── regression_results.csv
│   ├── hypothesis_testing.csv
│   └── diagnostics.csv
└── plots/
    ├── time_series_returns/
    ├── diagnostics/
    ├── regression/
    └── stationarity/
```

## Methodology

### 1. Data collection
Weekly adjusted closing prices are collected using `yfinance`.

### 2. Return construction
Continuous log returns are calculated as:

`r_t = ln(P_t / P_{t-1})`

### 3. Market regimes
The Nifty 50 return is split into positive and negative components:

```python
D_plus  = market_return.clip(lower=0)
D_minus = market_return.clip(upper=0)
```

### 4. Asymmetric beta estimation
The supplied workflow estimates separate upside and downside market sensitivities.

The workbook contains the HC3-robust OLS results and Wald symmetry tests.

The notebook also implements:

- ARIMAX(p,0,q)
- AIC grid search over p,q ∈ {0,1,2,3}
- ACF/PACF inspection
- residual diagnostics
- stationarity tests
- ARCH testing
- Jarque–Bera normality testing

### 5. Hypothesis test

The formal symmetry hypothesis is:

- **H₀:** β⁺ = β⁻
- **H₁:** β⁺ ≠ β⁻

A Wald chi-squared test is used with α = 0.05.

## Reproducing the analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Python workflow:

```bash
python src/asymmetric_beta_analysis.py
```

Or open:

```text
notebooks/asymmetric_beta_analysis.ipynb
```

The original notebook was supplied in Jupyter/Colab JSON format and has been retained as the primary research workflow.

## Dashboard

Open https://riddhichowdhuri.github.io/asymmetric-beta-Indian-sectoral-indices/results/asymmetric_beta_dashboard.html locally in a browser.

The dashboard includes:

- β⁺ vs β⁻ comparison
- ARIMAX model information
- AIC grid visualisation
- diagnostic summaries
- asymmetry heatmap
- regime scatter plots
- auto-generated insights

**Note:** the supplied dashboard explicitly labels its displayed ARIMAX numbers as representative/simulated outputs. They are retained for presentation purposes and should not be conflated with the OLS workbook results.

## Visual outputs

The `plots/` directory contains the supplied research figures covering:

- normalised price indices
- weekly return distributions
- market regimes
- ACF/PACF
- residual diagnostics
- regression scatter plots
- beta comparison
- asymmetry heatmap
- stationarity tests and related diagnostics

## Data source

The supplied notebook specifies **Yahoo Finance via `yfinance`** as the data source.

## Author

**Riddhi**

MSc Economics (Data Analytics) — Symbiosis School of Economics, Pune

