# Strategic Asset Allocation — Black-Litterman Model

**Course:** Quantitative Asset Allocation — Collegio Carlo Alberto, Turin  
**Author:** Prola Andrea  
**Date:** May 2024

---

## Overview

This project implements a full **Strategic Asset Allocation (SAA)** pipeline based on the
**Black-Litterman (BL) model**, applied to a 14-asset universe spanning Bonds, Equities,
and Opportunities. The analysis integrates market equilibrium returns with subjective investor
views, and constructs the efficient frontier under three progressive constraint regimes.

The key distinction from standard mean-variance optimisation is the **treatment of uncertainty**:
rather than using raw historical means (which are notoriously noisy inputs), the BL model
shrinks expected returns toward a market equilibrium baseline — producing more stable,
better-diversified portfolios.

---

## Asset Universe

| Asset Class | Count | Indices |
|-------------|-------|---------|
| **Bonds** | 6 | Bloomberg Global Agg Treasuries, EM Sovereign, Euro Corp., Global HY, EuroAgg 1-3Y, EuroAgg Govt |
| **Equities** | 4 | MSCI USA, MSCI Europe, MSCI Pacific, MSCI Emerging Markets |
| **Opportunities** | 4 | Bloomberg Commodity, MSCI Europe Energy, MSCI World Biotech, MSCI India |

Monthly total return data — source: Bloomberg / MSCI.

---

## Methodology

### 1. Risk Metrics
A comprehensive set of risk statistics is computed for each asset:
annualised return, volatility, Sharpe ratio, Sortino ratio, upside/downside semi-volatility,
maximum drawdown, and historical VaR.

Key finding: the three asset classes show clearly distinct risk-return profiles.
Bonds offer low, stable returns; Equities provide higher returns at higher volatility;
Opportunities carry the highest risk (vol up to 24%) but also the highest potential upside —
notably MSCI India (+14.6% annualised) and MSCI World Biotech (+14.6%).

### 2. Black-Litterman Model

**Market Neutral baseline:**  
Implied equilibrium returns are derived from the market-capitalisation-weighted portfolio
using the reverse-optimisation formula:
$$\pi = r_f + \lambda \cdot \Sigma \cdot w_{MN}$$
with risk aversion $\lambda = 4.5$.

**Investor views (P, Q, Ω):**

| View | Direction | Magnitude | Confidence |
|------|-----------|-----------|------------|
| Short-term bonds vs Euro Corporate bonds | Short-term outperforms | +1% | Low (15%) |
| MSCI USA vs MSCI Europe | USA outperforms | +5% | Moderate (30%) |
| MSCI India vs Bloomberg Commodity | India outperforms | +2% | High (22%) |

The uncertainty matrix Ω is calibrated as:
$$\Omega_{ii} = \left(\frac{1}{c_i} - 1\right) \cdot p_i^\top (\tau \Sigma) p_i$$

**Posterior returns** are computed via the standard BL formula:
$$\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
             \left[(\tau\Sigma)^{-1} \pi + P^\top \Omega^{-1} Q\right]$$

### 3. Efficient Frontier — Three Constraint Regimes

| Regime | Description | Key observation |
|--------|-------------|-----------------|
| **No constraints** | Long-only, weights ∈ [0,1] | Highly concentrated; a few high-return assets dominate |
| **Absolute constraints** | Per-asset min/max weight bounds | More diversified; frontier shifts right (higher min volatility) |
| **Infra-group constraints** | Relative bounds *within* each asset class + macro exposure caps | Best diversification across and within classes |

The progression illustrates a core insight: **constraints are not just compliance tools —
they are the mechanism that makes BL portfolios practically investable**.

### 4. Benchmarking vs MSCI World

| Metric | Black-Litterman | MSCI World |
|--------|----------------|------------|
| Annualised Return | 5.82% | 8.99% |
| Volatility | 9.17% | 14.80% |
| **Sharpe Ratio** | **0.91** | **0.54** |
| Max Drawdown | -12.6% | -21.1% |
| VaR (5%) | -3.37% | -7.05% |

While MSCI World delivers higher raw returns, the BL portfolio achieves superior
**risk-adjusted performance** — nearly double the Sharpe ratio — with significantly
lower drawdowns and VaR. This is consistent with the model's design goal: not to
maximise return, but to optimise the return-to-risk trade-off.

---

## Repository Structure

```
black-litterman-saa/
├── black_litterman_saa.py   # Full analysis pipeline
├── database.xlsx            # Monthly index prices (not tracked — see Data section)
├── requirements.txt
└── README.md
```

---

## Data

- `database.xlsx`: monthly index prices, sheet `selection`. Not included in this repo
  due to data licensing. Source: Bloomberg Terminal / MSCI.
- `Equity_Data.h5`: daily ETF returns used for the MSCI World benchmark (Section 6).
  Same licensing restriction applies.

Place both files in the repo root directory (or update `DATA_DIR` in the script).

---

## Setup

```bash
pip install -r requirements.txt
python black_litterman_saa.py
```

The script is also structured to run cell-by-cell in Spyder or VS Code
(sections delimited by `# %%`).

---

## Tools

| Library | Purpose |
|---------|---------|
| Python 3.10+ | — |
| NumPy / Pandas | Numerical computing, data handling |
| SciPy (`optimize`) | Mean-variance optimisation (SLSQP) |
| Matplotlib / Seaborn | Visualisation, heatmaps |
