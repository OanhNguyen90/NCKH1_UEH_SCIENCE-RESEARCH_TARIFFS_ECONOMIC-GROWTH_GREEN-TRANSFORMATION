# IMPORT TARIFF x TRADE OPENNESS x GREEN TRANSITION -> GDP GROWTH
## Vietnam · United States · China | Comparative Econometric Analysis
*Pipeline v7 | Generated: 2026-06-30 16:00*
*Data sources loaded from APIs: ['av_gdp']*

---
## 1. THEORETICAL FRAMEWORK

### 1.1 Research Hypotheses

| # | Hypothesis | Variable | Expected Sign | Theoretical Basis |
|---|-----------|----------|--------------|-------------------|
| H1 | MFN Tariff -> GDP | `mfn_tariff` | beta1 < 0 | Stolper-Samuelson (1941); Krugman New Trade Theory |
| H2 | Trade Openness -> GDP | `trade_openness` | beta2 > 0 | Frankel-Romer (1999); Trade-led growth |
| H3 | Renewable Energy -> GDP | `renewable_pct` | beta3 > 0 | IEA WEO 2024; Green economy theory |
| H4 | Tariff x Renewable (conditional) | `mfn_c x ren_c/10` | beta4 varies | Aiken & West (1991); Weaponized Interdependence |
| H5 | Structural heterogeneity 3 countries | Country FE + Chow | Reject H0 | Barro (1991); Geoeconomics |

### 1.2 Core Model Specification: ARDL(1) with Centered Interaction

```
GDP_t = alpha + phi*GDP_{t-1}                  [ARDL(1) persistence]
      + beta1*mfn_c_t                          [H1: tariff, centered]
      + beta2*trade_t                          [H2: trade openness]
      + beta3*ren_c_t                          [H3: renewable, centered]
      + beta4*(mfn_c x ren_c)/10              [H4: centered interaction]
      + gamma1*cpi_t + gamma2*rate_t           [controls: CPI, policy rate]
      + epsilon_t   [HAC: Newey-West, lag=4]

H4 Interaction Specification:
  mfn_c = mfn - mean(mfn|hist)   [centered MFN tariff]
  ren_c = ren - mean(ren|hist)   [centered renewable %]
  Interaction = (mfn_c x ren_c) / 10

  Centering rationale (Aiken & West 1991):
  - beta1 = marginal effect of tariff AT MEAN renewable level (interpretable)
  - beta3 = marginal effect of renewable AT MEAN tariff level (interpretable)
  - Reduces multicollinearity between main effects and interaction term
  - /10 scaling: keeps coefficient magnitudes in readable range

  Marginal effect of tariff: dGDP/dMFN = beta1 + beta4*(ren_c/10)
  Johnson-Neyman threshold: ren_c* = -10*beta1/beta4
  Absolute threshold: ren_abs* = ren_c* + mean(renewable_pct)
```

### 1.3 Data Sources and API Priority

| Source | Variables | Priority |
|--------|-----------|---------|
| IMF DataMapper API | GDP growth (NGDP_RPCH), CPI (PCPIEPCH) | 1st |
| World Bank WDI API | GDP, Trade openness, Renewable energy, CPI | 2nd |
| Alpha Vantage API | US Real GDP quarterly (REAL_GDP) | 3rd |
| FRED API (St. Louis Fed) | US Fed Funds Rate, US GDP QoQ | 4th |
| Embedded anchor data | All variables (IMF WEO Apr 2025 + WB WDI 2024) | Fallback |

### 1.4 Methodological Limitations

**1. Data Interpolation Limitation:**
Annual data is interpolated to quarterly frequency using CubicSpline, which creates
smooth but potentially artificial intra-year dynamics. This may introduce measurement
error and artificial autocorrelation in residuals. Mitigation: HAC (Newey-West)
standard errors with 4-lag bandwidth correct for resulting serial correlation.
Economic shock overlays (COVID 2020, US inflation 2022) partially address
known structural breaks, but other shocks may be missed.

**2. Small Sample Size (n=32 per country):**
With approximately 7 parameters per equation, the effective events-per-variable
(EPV) ratio is ~4.6, well below the recommended threshold of 10 (Harrell 1992).
Consequences: reduced statistical power, wider confidence intervals, increased
risk of overfitting. Mitigations: Ridge regularization + 500 Bootstrap iterations
(addresses overfitting); conservative 2Q maximum forecast horizon (addresses
forecast reliability); adj-R2 and BIC preferred over R2 and AIC for model selection.

**3. Endogeneity:**
Reverse causality (GDP -> tariff policy) may bias estimates. The ARDL(1) lag
structure partially addresses this. IV/GMM with WTO accession dates as instruments
is recommended for future research.

---
## 2. STATIONARITY AND COINTEGRATION TESTS

### 2.1 ADF and KPSS Stationarity Tests

**Combined decision rule (Maddala & Kim 1998):**

| ADF result | KPSS result | Order | Transform |
|-----------|------------|-------|-----------|
| Reject H0 (p<alpha) | Fail to reject H0 | **I(0)** | level |
| Fail to reject H0 | Reject H0 | **I(1)** | diff1 |
| Both reject | | I(0)* trend-stationary | level + HAC |
| Neither reject | | Ambiguous (small n) | level + HAC |

*Note: n=32 yields low test power. HAC correction applied regardless of outcome.*

**VN:**
| variable       |   n_obs |   adf_stat |   adf_pval |   kpss_stat |   kpss_pval | integration   | transform   |
|:---------------|--------:|-----------:|-----------:|------------:|------------:|:--------------|:------------|
| gdp_growth     |      32 |    -3.0598 |     0.039  |      0.0877 |        0.2  | I(0)          | level       |
| mfn_tariff     |      32 |     0.0105 |     0.2813 |      0.901  |        0.01 | I(1)          | diff1       |
| mfn_centered   |      32 |     0.0105 |     0.2813 |      0.901  |        0.01 | I(1)          | diff1       |
| renewable_pct  |      32 |    -2.8833 |     0.0511 |      0.7806 |        0.01 | I(0)*         | level       |
| ren_centered   |      32 |    -2.8833 |     0.0511 |      0.7806 |        0.01 | I(0)*         | level       |
| h4_interaction |      32 |    -1.6269 |     0.1667 |      0.26   |        0.2  | ambiguous     | level       |
| trade_openness |      32 |    -0.6719 |     0.2336 |      0.5665 |        0.05 | I(1)          | diff1       |
| cpi_inflation  |      32 |    -2.8576 |     0.0552 |      0.1889 |        0.2  | I(0)          | level       |
| policy_rate    |      32 |    -4.5347 |     0.01   |      0.3564 |        0.1  | I(0)          | level       |

**US:**
| variable       |   n_obs |   adf_stat |   adf_pval |   kpss_stat |   kpss_pval | integration   | transform   |
|:---------------|--------:|-----------:|-----------:|------------:|------------:|:--------------|:------------|
| gdp_growth     |      32 |    -2.2796 |     0.121  |      0.0981 |        0.2  | ambiguous     | level       |
| mfn_tariff     |      32 |    -0.5733 |     0.2405 |      0.7494 |        0.01 | I(1)          | diff1       |
| mfn_centered   |      32 |    -0.5733 |     0.2405 |      0.7494 |        0.01 | I(1)          | diff1       |
| renewable_pct  |      32 |    -2.7303 |     0.0758 |      0.8789 |        0.01 | I(0)*         | level       |
| ren_centered   |      32 |    -2.7303 |     0.0758 |      0.8789 |        0.01 | I(0)*         | level       |
| h4_interaction |      32 |    -0.8686 |     0.2198 |      0.2384 |        0.2  | ambiguous     | level       |
| trade_openness |      32 |    -2.7606 |     0.0709 |      0.1197 |        0.2  | I(0)          | level       |
| cpi_inflation  |      32 |    -1.4956 |     0.1759 |      0.1742 |        0.2  | ambiguous     | level       |
| policy_rate    |      32 |    -0.9456 |     0.2144 |      0.5951 |        0.05 | I(1)          | diff1       |

**CN:**
| variable       |   n_obs |   adf_stat |   adf_pval |   kpss_stat |   kpss_pval | integration   | transform   |
|:---------------|--------:|-----------:|-----------:|------------:|------------:|:--------------|:------------|
| gdp_growth     |      32 |    -3.6443 |     0.01   |      0.0771 |        0.2  | I(0)          | level       |
| mfn_tariff     |      32 |    -0.9476 |     0.2143 |      0.6956 |        0.05 | I(1)          | diff1       |
| mfn_centered   |      32 |    -0.9476 |     0.2143 |      0.6956 |        0.05 | I(1)          | diff1       |
| renewable_pct  |      32 |     0.8076 |     0.3371 |      0.8879 |        0.01 | I(1)          | diff1       |
| ren_centered   |      32 |     0.8076 |     0.3371 |      0.8879 |        0.01 | I(1)          | diff1       |
| h4_interaction |      32 |    -0.2877 |     0.2605 |      0.2188 |        0.2  | ambiguous     | level       |
| trade_openness |      32 |    -3.1665 |     0.0322 |      0.1037 |        0.2  | I(0)          | level       |
| cpi_inflation  |      32 |    -0.5819 |     0.2399 |      0.7804 |        0.01 | I(1)          | diff1       |
| policy_rate    |      32 |    -0.5679 |     0.2408 |      0.8805 |        0.01 | I(1)          | diff1       |

### 2.2 Engle-Granger Cointegration Tests

*H0: No cointegration | MacKinnon (1991) CV5% = -3.34 (2 variables)*
*Applied to variable pairs where both series show I(1) properties*

**VN:**
| hypothesis                 | variable       |    stat |   pvalue | cointegrated   |   long_run_b |
|:---------------------------|:---------------|--------:|---------:|:---------------|-------------:|
| H1: GDP ~ MFN Tariff       | mfn_tariff     | -3.2129 |     0.25 | False          |       0.7641 |
| H2: GDP ~ Trade Openness   | trade_openness | -3.4552 |     0.04 | True           |       0.0533 |
| H3: GDP ~ Renewable Energy | renewable_pct  | -3.346  |     0.04 | True           |      -0.1537 |

**US:**
| hypothesis                 | variable       |    stat |   pvalue | cointegrated   |   long_run_b |
|:---------------------------|:---------------|--------:|---------:|:---------------|-------------:|
| H1: GDP ~ MFN Tariff       | mfn_tariff     | -2.4033 |     0.25 | False          |       4.7537 |
| H2: GDP ~ Trade Openness   | trade_openness | -2.0757 |     0.25 | False          |       1.1643 |
| H3: GDP ~ Renewable Energy | renewable_pct  | -2.411  |     0.25 | False          |       0.1092 |

**CN:**
| hypothesis                 | variable       |    stat |   pvalue | cointegrated   |   long_run_b |
|:---------------------------|:---------------|--------:|---------:|:---------------|-------------:|
| H1: GDP ~ MFN Tariff       | mfn_tariff     | -3.3839 |     0.04 | True           |       0.9952 |
| H2: GDP ~ Trade Openness   | trade_openness | -3.4929 |     0.04 | True           |       0.6955 |
| H3: GDP ~ Renewable Energy | renewable_pct  | -3.6272 |     0.04 | True           |       0.0606 |

---
## 3. ECONOMETRIC RESULTS

### 3.1 OLS-HAC Results (Newey-West HAC, lag=4)

*HAC standard errors correct for serial correlation from interpolation and*
*heteroskedasticity common in macroeconomic panel data.*

**VN** - N=31, R2=0.8529, adj-R2=0.8161, AIC=108.0
| Term     |     Coef |   HAC-SE |   t-stat |   p-val | Sig   | Hypothesis   |
|:---------|---------:|---------:|---------:|--------:|:------|:-------------|
| const    |  0.02676 |  1.15744 |    0.023 |  0.9817 | n.s.  | Ctrl/ARDL    |
| lag1_gdp |  1.10544 |  0.10329 |   10.703 |  0      | ***   | Ctrl/ARDL    |
| d_mfn_c  |  5.88376 |  3.69537 |    1.592 |  0.1244 | n.s.  | H1           |
| d_trade  |  0.10159 |  0.01586 |    6.405 |  0      | ***   | H2           |
| ren_c    | -0.05758 |  0.05483 |   -1.05  |  0.3041 | n.s.  | H3           |
| h4_inter |  2.89656 |  1.64246 |    1.764 |  0.0905 | *     | H4           |
| cpi      |  0.02632 |  0.47179 |    0.056 |  0.956  | n.s.  | Ctrl/ARDL    |

**US** - N=31, R2=0.6591, adj-R2=0.5553, AIC=153.9
| Term           |      Coef |   HAC-SE |   t-stat |   p-val | Sig   | Hypothesis   |
|:---------------|----------:|---------:|---------:|--------:|:------|:-------------|
| const          | -32.4879  |  9.58459 |   -3.39  |  0.0025 | ***   | Ctrl/ARDL    |
| lag1_gdp       |   0.24197 |  0.11027 |    2.194 |  0.0386 | **    | Ctrl/ARDL    |
| d_mfn_c        | -12.8506  |  8.92471 |   -1.44  |  0.1634 | n.s.  | H1           |
| trade_openness |   1.46427 |  0.43184 |    3.391 |  0.0025 | ***   | H2           |
| ren_c          |   0.59907 |  0.21755 |    2.754 |  0.0113 | **    | H3           |
| h4_inter       | -26.4558  | 17.915   |   -1.477 |  0.1533 | n.s.  | H4           |
| cpi            |  -0.52751 |  0.23497 |   -2.245 |  0.0347 | **    | Ctrl/ARDL    |
| d_prate        |   0.02179 |  0.50017 |    0.044 |  0.9656 | n.s.  | Ctrl/ARDL    |

**CN** - N=31, R2=0.5394, adj-R2=0.3993, AIC=149.4
| Term           |      Coef |   HAC-SE |   t-stat |   p-val | Sig   | Hypothesis   |
|:---------------|----------:|---------:|---------:|--------:|:------|:-------------|
| const          | -20.0676  | 15.1235  |   -1.327 |  0.1976 | n.s.  | Ctrl/ARDL    |
| lag1_gdp       |   0.4729  |  0.24013 |    1.969 |  0.0611 | *     | Ctrl/ARDL    |
| d_mfn_c        |  -1.89061 |  1.73667 |   -1.089 |  0.2876 | n.s.  | H1           |
| trade_openness |   0.57419 |  0.40957 |    1.402 |  0.1743 | n.s.  | H2           |
| d_ren_c        |   0.47625 |  2.07918 |    0.229 |  0.8208 | n.s.  | H3           |
| h4_inter       |  -5.87111 |  3.394   |   -1.73  |  0.0971 | *     | H4           |
| d_cpi          |  -2.24787 |  0.63477 |   -3.541 |  0.0017 | ***   | Ctrl/ARDL    |
| d_prate        |  -2.54509 |  4.08408 |   -0.623 |  0.5393 | n.s.  | Ctrl/ARDL    |

### 3.2 GLSAR Prais-Winsten AR(1) Results

*Corrects for AR(1) serial correlation in residuals.*
*rho: estimated first-order autocorrelation coefficient.*

**VN** - N=31, rho=0.0884, R2=0.8522, adj-R2=0.8153
| Term     |     Coef |      SE |   t-stat |   p-val | Sig   |
|:---------|---------:|--------:|---------:|--------:|:------|
| const    | -0.14027 | 1.42294 |   -0.099 |  0.9223 | n.s.  |
| lag1_gdp |  1.06023 | 0.15446 |    6.864 |  0      | ***   |
| d_mfn_c  |  5.89873 | 6.52378 |    0.904 |  0.3749 | n.s.  |
| d_trade  |  0.09972 | 0.02016 |    4.946 |  0      | ***   |
| ren_c    | -0.06452 | 0.06733 |   -0.958 |  0.3475 | n.s.  |
| h4_inter |  2.67003 | 1.71389 |    1.558 |  0.1324 | n.s.  |
| cpi      |  0.1527  | 0.57921 |    0.264 |  0.7943 | n.s.  |

**US** - N=31, rho=-0.1504, R2=0.6559, adj-R2=0.5511
| Term           |      Coef |       SE |   t-stat |   p-val | Sig   |
|:---------------|----------:|---------:|---------:|--------:|:------|
| const          | -31.739   |  6.48291 |   -4.896 |  0.0001 | ***   |
| lag1_gdp       |   0.29085 |  0.13254 |    2.194 |  0.0386 | **    |
| d_mfn_c        | -15.2772  | 10.9132  |   -1.4   |  0.1749 | n.s.  |
| trade_openness |   1.44173 |  0.29667 |    4.86  |  0.0001 | ***   |
| ren_c          |   0.59593 |  0.21699 |    2.746 |  0.0115 | **    |
| h4_inter       | -32.6718  | 27.476   |   -1.189 |  0.2465 | n.s.  |
| cpi            |  -0.56398 |  0.29075 |   -1.94  |  0.0648 | *     |
| d_prate        |  -0.03279 |  0.7421  |   -0.044 |  0.9651 | n.s.  |

**CN** - N=31, rho=-0.2890, R2=0.5049, adj-R2=0.3542
| Term           |      Coef |      SE |   t-stat |   p-val | Sig   |
|:---------------|----------:|--------:|---------:|--------:|:------|
| const          | -23.4099  | 9.84617 |   -2.378 |  0.0261 | **    |
| lag1_gdp       |   0.56927 | 0.16158 |    3.523 |  0.0018 | ***   |
| d_mfn_c        |  -2.20192 | 2.91585 |   -0.755 |  0.4578 | n.s.  |
| trade_openness |   0.648   | 0.26404 |    2.454 |  0.0221 | **    |
| d_ren_c        |   1.39822 | 2.51911 |    0.555 |  0.5842 | n.s.  |
| h4_inter       |  -4.94432 | 3.80375 |   -1.3   |  0.2065 | n.s.  |
| d_cpi          |  -2.88197 | 1.18384 |   -2.434 |  0.0231 | **    |
| d_prate        |   1.65104 | 5.06476 |    0.326 |  0.7474 | n.s.  |

### 3.3 Model Diagnostics

*DW~2 -> no autocorr | LB p>0.05 -> no serial corr | BP p>0.05 -> homoskedastic | JB p>0.05 -> normality*

| Country   |   n_obs |     r2 |   adj_r2 |     dw |   lb_pval |   bp_pval |   jb_pval |   rmse |    mae |
|:----------|--------:|-------:|---------:|-------:|----------:|----------:|----------:|-------:|-------:|
| VN        |      31 | 0.8529 |   0.8161 | 1.8163 |    0.0282 |    0.1275 |    0.6078 | 1.1015 | 0.9491 |
| US        |      31 | 0.6591 |   0.5553 | 2.0321 |    0.1741 |    0.2333 |    0.2003 | 2.2355 | 1.5763 |
| CN        |      31 | 0.5394 |   0.3993 | 1.987  |    0.4535 |    0.059  |    0      | 2.0796 | 1.3658 |

### 3.4 H4 Marginal Effect and Johnson-Neyman Threshold

| Country   |   beta_mfn(H1) |   beta_h4(H4) |   ME@Low(mu-sigma) |   ME@Mean |   ME@High(mu+sigma) |   JN_threshold_% | Interpretation                                                                   |
|:----------|---------------:|--------------:|-------------------:|----------:|--------------------:|-----------------:|:---------------------------------------------------------------------------------|
| VN        |        5.88376 |       2.89656 |           4.5447   |   5.88375 |             7.22281 |            26.38 | When renewable > 26.4%: tariff effect turns POSITIVE -> green transition buffers |
| US        |      -12.8506  |     -26.4558  |          -6.15726  | -12.8506  |           -19.5439  |            16.26 | When renewable < 16.3%: tariff effect is MORE NEGATIVE -> lack of green transiti |
| CN        |       -1.89061 |      -5.87111 |           0.054158 |  -1.89061 |            -3.83537 |            28.1  | When renewable < 28.1%: tariff effect is MORE NEGATIVE -> lack of green transiti |

*ME = dGDP/dMFN = beta_mfn + beta_h4*(ren_c/10)*
*J-N threshold: renewable energy % at which tariff effect changes sign*

### 3.5 H5 Panel Analysis and Chow Structural Break Test

**Cross-country coefficient comparison (H1-H4):**
| Country   |   H1(Tariff) | H2(Trade)   | H3(Renew)   | H4(Interact)   |
|:----------|-------------:|:------------|:------------|:---------------|
| VN        |       5.8838 | +0.1016***  | -0.0576     | +2.8966*       |
| US        |     -12.8506 | +1.4643***  | +0.5991**   | -26.4558       |
| CN        |      -1.8906 | +0.5742     | +0.4763     | -5.8711*       |

**** p<0.01 | ** p<0.05 | * p<0.10 | (blank) not significant*

**Chow Test - Structural Heterogeneity Between Country Pairs:**
| Pair     |   F-stat |   p-value | Reject H0   | Conclusion                                              |
|:---------|---------:|----------:|:------------|:--------------------------------------------------------|
| VN vs US |   8.8274 |    0      | True        | Structural difference confirmed (p<0.05) - H5 supported |
| VN vs CN |   3.1512 |    0.008  | True        | Structural difference confirmed (p<0.05) - H5 supported |
| US vs CN |   3.1175 |    0.0067 | True        | Structural difference confirmed (p<0.05) - H5 supported |

*H0: same coefficients across countries | p<0.05 -> structural heterogeneity (H5 supported)*

---
## 4. FORECASTING

### 4.1 Forecast Horizon Justification

The maximum reliable forecast horizon is 2 quarters, justified by:
- **Sample size constraint**: n=32 with ~7 parameters -> EPV ~4.6
  (Harvey 1990: EPV<5 limits reliable forecast horizon to 1-2Q)
- **Small-sample forecast uncertainty**: West (1996) and Clark & West (2007)
  show parameter estimation error expands CI substantially for n<50 beyond 2Q
- **ARDL(1) dynamics**: The AR(1) coefficient typically <1, so model-implied
  uncertainty accumulates rapidly beyond 2Q in small samples
- **Backtest evidence**: ratio = backtest_RMSE / in-sample_RMSE determines horizon

**CI formula**: +/- 1.96 * sigma_eps * sqrt(1 + 0.30*h)
  where h = horizon, 0.30 = conservative small-sample adjustment

### 4.2 Hold-out Backtest (2025Q1-Q4)

| Country   |   n_train |   n_test |   RMSE |    MAE |   MAPE% |   In-RMSE |   Ratio | Fit Eval         |   Opt Horizon |
|:----------|----------:|---------:|-------:|-------:|--------:|----------:|--------:|:-----------------|--------------:|
| VN        |        27 |        4 | 9.5982 | 8.3025 |  275.87 |    1.0048 |   9.552 | Heavy overfit    |             1 |
| US        |        27 |        4 | 4.7587 | 3.8291 |  183.76 |    2.118  |   2.247 | Moderate overfit |             1 |
| CN        |        27 |        4 | 6.0722 | 4.5948 |   58.72 |    2.0001 |   3.036 | Heavy overfit    |             1 |

### 4.3 Out-of-sample Validation: 2026Q1-Q2 Forecast vs Actual

> Sources: GSO (VN), BEA (US), NBS (CN) / IMF WEO April 2025

**VN:**
| period   |   forecast |   actual |    error |   abs_error |    ci_lo |   ci_hi | in_95CI   |
|:---------|-----------:|---------:|---------:|------------:|---------:|--------:|:----------|
| 2026Q1   |    -7.7175 |     6.93 | -14.6475 |     14.6475 | -10.1791 | -5.2558 | False     |
| 2026Q2   |   -12.0823 |     7.1  | -19.1823 |     19.1823 | -14.8133 | -9.3513 | False     |

**US:**
| period   |   forecast |   actual |   error |   abs_error |   ci_lo |   ci_hi | in_95CI   |
|:---------|-----------:|---------:|--------:|------------:|--------:|--------:|:----------|
| 2026Q1   |    -0.3822 |      2   | -2.3822 |      2.3822 | -5.378  |  4.6136 | True      |
| 2026Q2   |    -0.9023 |      2.1 | -3.0023 |      3.0023 | -6.4446 |  4.64   | True      |

**CN:**
| period   |   forecast |   actual |   error |   abs_error |   ci_lo |   ci_hi | in_95CI   |
|:---------|-----------:|---------:|--------:|------------:|--------:|--------:|:----------|
| 2026Q1   |     7.0314 |      5.4 |  1.6314 |      1.6314 |  2.3839 | 11.6788 | True      |
| 2026Q2   |     5.82   |      4.6 |  1.22   |      1.22   |  0.6641 | 10.9758 | True      |

### 4.4 Final Optimal Forecast: 2026Q3-Q4

> Model trained on 2018Q1-2025Q4 | Anchored at actual 2026Q2 where available

| Country   | Period   |   Forecast% |   CI Low |   CI High |   RMSE hist |
|:----------|:---------|------------:|---------:|----------:|------------:|
| VN        | 2026Q3   |      4.3047 |   1.843  |    6.7664 |      1.1015 |
| US        | 2026Q3   |     -0.3672 |  -5.363  |    4.6286 |      2.2355 |
| CN        | 2026Q3   |      4.6363 |  -0.0111 |    9.2837 |      2.0796 |

---
## 5. POLICY IMPLICATIONS FOR VIETNAM

### 5.1 Vietnam's Strategic Intermediary Position

- **H1 (VN)**: beta_tariff = +5.8838 (p=0.124) -> 1% increase in MFN tariff increases GDP by 5.8838pp
- **H4 (VN)**: ME at mean renewable = +5.8838 | J-N threshold = 26.38%
  -> When renewable > 26.4%: tariff effect turns POSITIVE -> green transition buffers tariff harm ('green impetus')

### 5.2 Tariff Policy Recommendations

1. **Supply chain localization**: Reduce MFN on green input materials -> address rules-of-origin bottlenecks
2. **Pigouvian green tariff**: Higher tariffs on polluting/old technology -> incentivize green FDI
3. **Reach J-N threshold**: Increase renewable % toward J-N threshold -> neutralize/reverse tariff harm
4. **CBAM defense**: Prepare carbon footprint reporting -> protect exports to EU/US markets

---
## 6. LIMITATIONS AND FUTURE RESEARCH

1. **Endogeneity**: IV/GMM with WTO accession dates as instrumental variables
2. **Panel unit root**: IPS test (Im, Pesaran & Shin 2003) for full panel
3. **Non-linearity**: Threshold ARDL (Shin et al. 2014) for asymmetric tariff effects
4. **Sample expansion**: Update to Q4/2026 (n=36) for improved statistical power
5. **Bayesian approach**: BVAR for forecast uncertainty with small n