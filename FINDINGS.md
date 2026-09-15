# Findings

## Executive summary

This project studies asymmetric market sensitivity of six Indian sectoral indices relative to the Nifty 50 using weekly data for January 2018–December 2025.

The supplied workbook contains the **HC3-robust OLS asymmetric-beta results**. The supplied notebook additionally implements the intended **ARIMAX asymmetric-beta workflow**, including AIC-based ARMA order selection and residual diagnostics.

### Main empirical findings from the supplied OLS workbook

- **No sector shows statistically significant beta asymmetry at the 5% level.** The Wald-test p-values are above 0.05 for all six sectors.
- **Nifty Bank has the highest estimated downside beta** among the six sectors: β⁻ = 1.1563.
- **Nifty FMCG has the lowest downside beta:** β⁻ = 0.5528, making it the most defensive on this measure.
- **Nifty IT has the largest positive downside-minus-upside beta spread:** β⁻ − β⁺ = 0.3455. Its estimated downside sensitivity is higher than its upside sensitivity, although the Wald test does not reject symmetry at 5%.
- **Nifty Oil & Gas also has β⁻ > β⁺**, with a spread of 0.1120, again without statistically significant asymmetry.
- Auto, Bank, FMCG and Healthcare have estimated β⁺ > β⁻, suggesting stronger participation in market upturns than downturns in the point estimates.
- All six return series are reported as stationary by the ADF test.
- Ljung–Box diagnostics report no serial autocorrelation for Auto, Bank, FMCG, IT and Oil & Gas; Healthcare shows evidence of residual/return serial correlation at the reported test specification.
- ARCH effects are reported for Auto, Bank and Healthcare, while FMCG, IT and Oil & Gas do not show significant ARCH effects at the stated threshold.
- Jarque–Bera results indicate non-normality for Auto, Bank, FMCG, Healthcare and Oil & Gas; IT is reported as normal at the 5% level.

## OLS beta results

| Sector | β⁺ | β⁻ | β⁻ − β⁺ | Wald p-value | R² |
|---|---:|---:|---:|---:|---:|
| Nifty Auto | 1.2403 | 1.0891 | -0.1512 | 0.4310 | 0.6499 |
| Nifty Bank | 1.3638 | 1.1563 | -0.2075 | 0.3496 | 0.7829 |
| Nifty FMCG | 0.6058 | 0.5528 | -0.0530 | 0.8142 | 0.3942 |
| Nifty Healthcare | 0.6654 | 0.5903 | -0.0751 | 0.8292 | 0.2619 |
| Nifty IT | 0.6829 | 1.0284 | 0.3455 | 0.0803 | 0.3998 |
| Nifty Oil & Gas | 0.8874 | 0.9994 | 0.1120 | 0.6405 | 0.5081 |

## Interpretation

The point estimates suggest economically different sector responses across market regimes, but the formal Wald tests do not provide sufficient statistical evidence to conclude that the upside and downside betas are different at the 5% significance level.

The strongest evidence against symmetry is observed for **Nifty IT** (p = 0.0803), but this remains above 0.05. This should therefore be described as suggestive rather than statistically significant at the conventional 5% level.

## Important reproducibility note

The HTML dashboard supplied with the project contains a separate set of **representative ARIMAX outputs** explicitly labelled as such in its source. Those values should not be presented as identical to the OLS workbook results. The repository therefore keeps the notebook, workbook and dashboard as separate artifacts and identifies the OLS workbook as the source for the findings above.

## Data and methodology

- Benchmark: Nifty 50 (`^NSEI`)
- Sectors: Nifty Auto, Nifty Bank, Nifty FMCG, Nifty Healthcare, Nifty IT, Nifty Oil & Gas
- Frequency: weekly
- Period: January 2018–December 2025
- Returns: log returns, `ln(P_t / P_{t-1})`
- Asymmetric regressors: positive and negative market-return regimes
- OLS workbook: HC3 robust standard errors and Wald chi-squared symmetry test
- Notebook ARIMAX workflow: `statsmodels.ARIMA`, exogenous positive/negative market-return splits, AIC grid search over p,q = 0–3, and residual diagnostics

## Caveats

This repository preserves the supplied analysis rather than silently changing its estimates or methodology. The ARIMAX notebook and the OLS workbook should be treated as distinct model outputs. The supplied dashboard should be read with its own in-file note that its displayed ARIMAX numbers are representative/simulated outputs.
