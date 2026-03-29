"""
QUANTITATIVE ASSET ALLOCATION
Strategic Asset Allocation via the Black-Litterman Model

Description
-----------
This script implements a full Strategic Asset Allocation (SAA) pipeline based on
the Black-Litterman (BL) model applied to a 14-asset universe spanning Bonds,
Equities, and Opportunities.

The analysis proceeds in five stages:
  1. Data loading and return computation
  2. Risk metrics and exploratory analysis
  3. Portfolio optimisation functions (mean-variance)
  4. Black-Litterman model: Market Neutral baseline + investor views
  5. Efficient frontiers under three constraint regimes:
       (a) No constraints
       (b) Absolute weight constraints
       (c) Infra-group (relative) constraints
  6. Benchmarking against MSCI World

Data
----
  - database.xlsx      : monthly index prices (sheet: 'selection')
  - Equity_Data.h5     : daily ETF returns used for MSCI World benchmarking
    Place both files in the same directory as this script and update DATA_DIR.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as spopt
from scipy.stats import norm
from IPython.display import display

DATA_DIR   = "./"                  # update to data directory
ANNUAL     = 12                    # monthly data → annualisation factor
RISK_AVERSION = 4.5                # lambda for implied equilibrium returns
SAVE_PLOTS = False                 # set True to save figures to disk

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(DATA_DIR + "database.xlsx", sheet_name="selection")
df.set_index("Index", inplace=True)
df = df.iloc[:, 1:]
df.replace("n.e.", np.nan, inplace=True)
df.dropna(inplace=True)
df = df.pct_change().dropna()       # price levels → monthly returns

# Asset class partitions
AC_bonds        = df.columns[:6].tolist()
AC_equities     = df.columns[6:10].tolist()
AC_opportunities = df.columns[10:].tolist()

print("Asset universe loaded.")
print(f"  Bonds         ({len(AC_bonds)}): {AC_bonds}")
print(f"  Equities      ({len(AC_equities)}): {AC_equities}")
print(f"  Opportunities ({len(AC_opportunities)}): {AC_opportunities}")
print(f"\nDate range : {df.index[0].date()} to {df.index[-1].date()}")
print(f"Observations: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def annualize_rets(s, periods_per_year):
    """Annualised return for a Series or DataFrame."""
    def _series(s):
        s = s.dropna()
        return (1 + s).prod() ** (periods_per_year / len(s)) - 1
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def annualize_vol(s, periods_per_year):
    """Annualised volatility for a Series or DataFrame."""
    def _series(s):
        return s.dropna().std() * periods_per_year ** 0.5
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def sharpe_ratio(s, risk_free_rate, periods_per_year):
    """Annualised Sharpe ratio."""
    def _series(s):
        s = s.dropna()
        if isinstance(risk_free_rate, pd.Series):
            excess = s.sub(risk_free_rate).dropna()
        else:
            rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
            excess = s - rf_period
        return annualize_rets(excess, periods_per_year) / annualize_vol(s, periods_per_year)
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def annualize_vol_up(s, periods_per_year):
    """Upside (positive-return) annualised volatility."""
    def _series(s):
        s = s.dropna()
        return (s[s > 0].std() * periods_per_year) ** 0.5
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def annualize_vol_dn(s, periods_per_year):
    """Downside (negative-return) annualised volatility."""
    def _series(s):
        s = s.dropna()
        return (s[s < 0].std() * periods_per_year) ** 0.5
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def sortino_ratio(s, risk_free_rate, periods_per_year):
    """Annualised Sortino ratio (uses downside volatility as denominator)."""
    def _series(s):
        s = s.dropna()
        if isinstance(risk_free_rate, pd.Series):
            excess = s.sub(risk_free_rate).dropna()
        else:
            rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
            excess = s - rf_period
        return annualize_rets(excess, periods_per_year) / annualize_vol_dn(s, periods_per_year)
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def max_drawdown(s, window):
    """Maximum drawdown over a rolling window."""
    def _series(s):
        s = s.dropna()
        roll_max = (1 + s).cumprod().rolling(window, min_periods=1).max()
        daily_dd = (1 + s).cumprod() / roll_max - 1.0
        return daily_dd.rolling(window, min_periods=1).min().min()
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def var_historic(s, level=0.05):
    """Historical VaR at the given confidence level."""
    def _series(s):
        return np.percentile(s.dropna(), level * 100)
    if isinstance(s, pd.DataFrame):
        return s.aggregate(_series)
    return _series(s)


def risk_metrics_table(s, risk_free_rate, periods_per_year, window):
    """
    Compute a full risk metrics table for a Series or DataFrame.

    Returns a DataFrame with rows:
        rets, vol, Sharpe, Skew, vol_up, vol_dn, Sortino, MDD, VaR
    """
    metrics = {
        "rets":    annualize_rets(s, periods_per_year),
        "vol":     annualize_vol(s, periods_per_year),
        "Sharpe":  sharpe_ratio(s, risk_free_rate, periods_per_year),
        "Skew":    s.skew() if isinstance(s, pd.DataFrame) else pd.Series(s.skew(), index=[s.name]),
        "vol_up":  annualize_vol_up(s, periods_per_year),
        "vol_dn":  annualize_vol_dn(s, periods_per_year),
        "Sortino": sortino_ratio(s, risk_free_rate, periods_per_year),
        "MDD":     max_drawdown(s, window),
        "VaR":     var_historic(s.dropna()),
    }
    if isinstance(s, pd.DataFrame):
        return pd.DataFrame(metrics, index=s.columns).T
    return pd.DataFrame(metrics, index=[s.name]).T


# ── Compute and display risk metrics ─────────────────────────────────────────
risk_free = df["Bloomberg EuroAgg 1-3 Year Total Return Index Value Unhedged EUR"]
rm = risk_metrics_table(df, risk_free_rate=risk_free, periods_per_year=ANNUAL, window=6)
print("\nRisk Metrics (all assets):")
display(rm)

# ── Annualised covariance matrix ──────────────────────────────────────────────
Sigma = df.cov() * ANNUAL

# ── Correlation heatmap ───────────────────────────────────────────────────────
short_names = {name: " ".join(name.split()[:4]) for name in df.columns}
corr_short  = df.rename(columns=short_names).corr()

plt.figure(figsize=(14, 14))
sns.heatmap(corr_short, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# ── Returns and volatility bar charts ─────────────────────────────────────────
for metric, label in [("rets", "Annualised Return"), ("vol", "Annualised Volatility")]:
    data = rm.loc[metric]
    plt.figure(figsize=(10, 8))
    plt.barh(data.index, data.values,
             color=["tomato" if v < 0 else "steelblue" for v in data.values])
    plt.xlabel(label)
    plt.title(f"{label} by Asset")
    plt.tight_layout()
    plt.show()

# ── Cumulative returns ────────────────────────────────────────────────────────
(1 + df[AC_bonds + AC_equities]).cumprod().plot(figsize=(12, 5),
    title="Cumulative Returns — Bonds & Equities")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.show()

(1 + df[AC_opportunities]).cumprod().plot(figsize=(12, 5),
    title="Cumulative Returns — Opportunities")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3. PORTFOLIO OPTIMISATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

mu    = df.mean()
x0    = pd.Series(1 / df.shape[1], index=df.columns)   # equal-weight starting point


def port_ret(w, mu, annual):
    """Expected annualised portfolio return."""
    return np.dot(w, mu) * annual


def port_variance(w, r, annual):
    """Expected annualised portfolio variance."""
    return np.dot(w, r.cov() @ w) * annual


def port_vola(w, r, annual):
    """Expected annualised portfolio volatility."""
    return np.sqrt(port_variance(w, r, annual))


def port_ret_eq(w, mu, annual, mu_0):
    """Return constraint for optimiser: port_ret(w) - mu_0 = 0."""
    return np.dot(w, mu) * annual - mu_0


# ─────────────────────────────────────────────────────────────────────────────
# 4. BLACK-LITTERMAN MODEL
# ─────────────────────────────────────────────────────────────────────────────

# ── 4.1 Market Neutral (equilibrium) portfolio ────────────────────────────────
MN_weights = [
    0.106, 0.028, 0.017, 0.020, 0.087, 0.206,   # Bonds
    0.228, 0.096, 0.041, 0.082,                   # Equities
    0.0225, 0.0225, 0.0225, 0.0225,               # Opportunities
]
MN_series = pd.Series(MN_weights, index=df.columns)

rf_annual  = mu["Bloomberg EuroAgg 1-3 Year Total Return Index Value Unhedged EUR"] * ANNUAL
MN_rets    = rf_annual + RISK_AVERSION * (MN_series.T @ Sigma).T

plt.figure(figsize=(10, 8))
plt.barh(MN_series.index, MN_series.values, color="steelblue")
plt.xlabel("Weight"); plt.title("Market Neutral Portfolio Weights")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 8))
plt.barh(MN_rets.index, MN_rets.values, color="steelblue")
plt.xlabel("Expected Return"); plt.title("Market Neutral Implied Returns")
plt.tight_layout(); plt.show()

# ── 4.2 Investor views (P matrix & Q vector) ─────────────────────────────────
"""
Three relative views:
  1. Short-term bonds (EuroAgg 1-3Y) outperform Euro Corporate bonds by  1% — low confidence
  2. MSCI USA outperforms MSCI Europe                              by  5% — moderate confidence
  3. MSCI India outperforms Bloomberg Commodity Index              by  2% — high confidence
"""

asset_classes = df.columns.tolist()

P_matrix = pd.DataFrame([
    [-1,  0,  1,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0],  # view 1
    [ 0,  0,  0,  0,  0,  0,   1, -1,  0,  0,   0,  0,  0,  0],  # view 2
    [ 0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  -1,  0,  0,  1],  # view 3
], columns=asset_classes)

Q_vector = np.array([0.01, 0.05, 0.02])    # view magnitudes (annual)
C_vector = np.array([0.15, 0.30, 0.22])    # confidence levels

tau = 1 / ((len(df) + 1) / ANNUAL)         # uncertainty scaling factor

views_summary = pd.DataFrame({
    "Market 1":   ["Short-term bonds",  "USA equities",      "India equities"],
    "View":       ["Low outperformance","High outperformance","Moderate outperformance"],
    "Market 2":   ["Euro Corp. bonds",  "Europe equities",   "Commodities"],
    "Confidence": ["Low",               "Moderate",          "High"],
})
print("\nInvestor Views:")
display(views_summary)

# ── 4.3 BL posterior returns ──────────────────────────────────────────────────

def omega_matrix(P, confidence, tau, Sigma):
    """
    Diagonal uncertainty matrix Omega.
    Omega_ii = (1/c_i - 1) * p_i^T (tau * Sigma) p_i
    """
    n = len(confidence)
    Omega = np.zeros((n, n))
    for i in range(n):
        p = P.iloc[i].values
        Omega[i, i] = (1 / confidence[i] - 1) * p @ (tau * Sigma) @ p
    return Omega


def BL_posterior_returns(tau, Sigma, MN_rets, P, Omega, Q):
    """
    Black-Litterman posterior expected returns.
    mu_BL = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1
            [(tau*Sigma)^-1 pi + P^T Omega^-1 q]
    """
    inv_tau_sigma = np.linalg.inv(tau * Sigma)
    inv_omega     = np.linalg.inv(Omega)
    posterior_cov = np.linalg.inv(inv_tau_sigma + P.values.T @ inv_omega @ P.values)
    posterior_ret = posterior_cov @ (inv_tau_sigma @ MN_rets + P.values.T @ inv_omega @ Q)
    return posterior_ret, posterior_cov


Omega    = omega_matrix(P_matrix, C_vector, tau, Sigma)
BL_rets, BL_cov = BL_posterior_returns(tau, Sigma, MN_rets, P_matrix, Omega, Q_vector)
BL_series = pd.Series(BL_rets, index=df.columns)
BL_mu     = BL_rets / ANNUAL   # monthly equivalent

plt.figure(figsize=(10, 8))
plt.barh(BL_series.index, BL_series.values, color="steelblue")
plt.xlabel("Expected Return"); plt.title("Black-Litterman Posterior Returns")
plt.tight_layout(); plt.show()

rets_diff = BL_rets - MN_rets
plt.figure(figsize=(10, 8))
plt.barh(rets_diff.index, rets_diff.values,
         color=["tomato" if v < 0 else "steelblue" for v in rets_diff])
plt.xlabel("Return Difference"); plt.title("BL minus Market Neutral Returns")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5. EFFICIENT FRONTIERS
# ─────────────────────────────────────────────────────────────────────────────

def run_efficient_frontier(BL_mu, df, mu_range, bnds, extra_constraints=None, label=""):
    """
    Compute the mean-variance efficient frontier for a given set of constraints.

    Args:
        BL_mu            : monthly BL expected returns (array)
        df               : return DataFrame (used for covariance)
        mu_range         : target annualised return grid
        bnds             : list of (min, max) weight bounds per asset
        extra_constraints: list of additional scipy constraint dicts
        label            : plot title suffix

    Returns:
        eff_front : DataFrame with columns ['Mu', 'Std']
        weights   : DataFrame of optimal weights per target return
    """
    eff_front = pd.DataFrame(index=mu_range, columns=["Mu", "Std"], dtype=float)
    weights   = pd.DataFrame(index=mu_range, columns=df.columns, dtype=float)

    for mu_0 in mu_range:
        constraints = [
            {"type": "eq", "fun": lambda x: x.sum() - 1},
            {"type": "eq", "fun": port_ret_eq, "args": (BL_mu, ANNUAL, mu_0)},
        ]
        if extra_constraints:
            constraints += extra_constraints

        res = spopt.minimize(
            port_variance, x0, method="SLSQP",
            args=(df, ANNUAL),
            bounds=bnds, constraints=constraints,
            options={"disp": False},
        )
        eff_front.loc[mu_0, "Mu"]  = np.dot(res.x, BL_mu) * ANNUAL
        eff_front.loc[mu_0, "Std"] = port_vola(res.x, df, ANNUAL)
        weights.loc[mu_0]          = res.x

    # Efficient frontier scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(eff_front["Std"], eff_front["Mu"], c="tomato", s=15)
    plt.xlabel("Volatility"); plt.ylabel("Expected Return")
    plt.title(f"Efficient Frontier — {label}")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # Weights evolution stack plot
    plt.figure(figsize=(14, 6))
    plt.stackplot(eff_front["Mu"].tolist(), weights.values.T.tolist(), labels=weights.columns)
    plt.title(f"Weights Evolution — {label}")
    plt.xlabel("Expected Return"); plt.ylabel("Portfolio Weight")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7)
    plt.tight_layout(); plt.show()

    return eff_front, weights


# ── 5a. No Constraints ────────────────────────────────────────────────────────
mu_range_nc = np.arange(0.02, 0.06, 0.0025)
bnds_nc     = [(0, 1)] * len(df.columns)

eff_nc, w_nc = run_efficient_frontier(BL_mu, df, mu_range_nc, bnds_nc, label="No Constraints")

# ── 5b. Absolute Constraints ──────────────────────────────────────────────────
abs_min = [0.00, 0.00, 0.00, 0.08, 0.08, 0.00,
           0.08, 0.12, 0.04, 0.00,
           0.00, 0.00, 0.00, 0.00]
abs_max = [0.15, 0.35, 0.60, 0.40, 0.30, 1.00,
           0.50, 1.00, 0.30, 0.20,
           0.05, 0.10, 0.10, 0.10]
bnds_abs = list(zip(abs_min, abs_max))

eff_abs, w_abs = run_efficient_frontier(BL_mu, df, mu_range_nc, bnds_abs, label="Absolute Constraints")

# ── 5c. Infra-Group (Relative) Constraints ────────────────────────────────────

def make_relative_constraint_matrix(assets, asset_class, etf, weight_threshold, name):
    """
    Build a relative constraint row vector A such that:
        A * w >= 0  →  w_etf / sum(w_asset_class) >= weight_threshold
    """
    others = [a for a in asset_class if a not in etf]
    A = pd.DataFrame(0, index=assets, columns=[name])
    A.loc[others, name]  = -weight_threshold
    A.loc[etf,    name]  =  1 - weight_threshold
    return A.T


def Axb_lower(x, A, b):
    """Constraint: A*x <= b  →  -A*x + b >= 0."""
    return -A.dot(x) + b


def Axb_bigger(x, A, b):
    """Constraint: A*x >= b  →  A*x - b >= 0."""
    return A.dot(x) - b


# Per-asset relative bounds within each asset class
infra_bounds = {
    "bonds":         {"lower": [0.00, 0.10, 0.05, 0.00, 0.20, 0.35],
                      "upper": [1.00, 0.25, 0.20, 0.15, 0.40, 0.55]},
    "equities":      {"lower": [0.37, 0.22, 0.07, 0.02],
                      "upper": [0.60, 0.40, 0.15, 0.12]},
    "opportunities": {"lower": [0.05, 0.00, 0.10, 0.15],
                      "upper": [0.45, 0.35, 0.40, 0.60]},
}

def build_constraint_matrices(assets, asset_class, bounds):
    LB = pd.DataFrame()
    UB = pd.DataFrame()
    for i, asset in enumerate(asset_class):
        LB = pd.concat([LB, make_relative_constraint_matrix(assets, asset_class, [asset], bounds["lower"][i], asset)])
        UB = pd.concat([UB, make_relative_constraint_matrix(assets, asset_class, [asset], bounds["upper"][i], asset)])
    return LB, UB, pd.Series(0, index=LB.index), pd.Series(0, index=UB.index)

assets = df.columns

bonds_LB, bonds_UB, zero_b_LB, zero_b_UB = build_constraint_matrices(assets, AC_bonds, infra_bounds["bonds"])
eq_LB,    eq_UB,    zero_e_LB, zero_e_UB = build_constraint_matrices(assets, AC_equities, infra_bounds["equities"])
opp_LB,   opp_UB,  zero_o_LB, zero_o_UB = build_constraint_matrices(assets, AC_opportunities, infra_bounds["opportunities"])

# Asset class exposure caps (Bonds ≤ 40%, Equities ≤ 60%, Opportunities ≤ 5%)
a_class = pd.DataFrame(0, index=df.columns, columns=["Bonds", "Equities", "Opportunities"])
a_class.loc[AC_bonds,         "Bonds"]         = 1
a_class.loc[AC_equities,      "Equities"]       = 1
a_class.loc[AC_opportunities, "Opportunities"]  = 1
a_class = a_class.T

b_class = pd.Series({"Bonds": 0.40, "Equities": 0.60, "Opportunities": 0.05})

infra_constraints = [
    {"type": "ineq", "fun": Axb_lower,  "args": (eq_UB,    zero_e_UB)},
    {"type": "ineq", "fun": Axb_bigger, "args": (eq_LB,    zero_e_LB)},
    {"type": "ineq", "fun": Axb_lower,  "args": (bonds_UB,  zero_b_UB)},
    {"type": "ineq", "fun": Axb_bigger, "args": (bonds_LB,  zero_b_LB)},
    {"type": "ineq", "fun": Axb_lower,  "args": (opp_UB,   zero_o_UB)},
    {"type": "ineq", "fun": Axb_bigger, "args": (opp_LB,   zero_o_LB)},
    {"type": "ineq", "fun": Axb_lower,  "args": (a_class,   b_class)},
]

mu_range_infra = np.arange(0.02, 0.08, 0.002)
bnds_infra     = [(0, 1)] * len(df.columns)

eff_infra, w_infra = run_efficient_frontier(
    BL_mu, df, mu_range_infra, bnds_infra,
    extra_constraints=infra_constraints,
    label="Infra-Group Constraints",
)

# Macro asset class weights evolution
w_classes = pd.DataFrame({
    "Bonds":         w_infra[AC_bonds].sum(axis=1),
    "Equities":      w_infra[AC_equities].sum(axis=1),
    "Opportunities": w_infra[AC_opportunities].sum(axis=1),
})
plt.figure(figsize=(14, 6))
plt.stackplot(eff_infra["Mu"].tolist(), w_classes.values.T.tolist(), labels=w_classes.columns,
              colors=["steelblue", "tomato", "seagreen"])
plt.title("Macro Asset Class Weights Evolution — Infra-Group Constraints")
plt.xlabel("Expected Return"); plt.ylabel("Portfolio Weight")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6. BENCHMARKING vs MSCI WORLD
# ─────────────────────────────────────────────────────────────────────────────

def monthly_returns(s):
    """Resample daily returns to monthly compound returns."""
    return s.resample("ME").apply(lambda x: (1 + x).prod() - 1)


ETFs      = pd.read_hdf(DATA_DIR + "Equity_Data.h5", "returns/ETFs/").dropna()
MSCI_World = monthly_returns(ETFs["MSCI_World"])

# BL portfolio: last optimal weights from the infra-group frontier
bl_portfolio = (df * w_infra.iloc[-1].values).reindex(MSCI_World.index).sum(axis=1)

# Cumulative performance
bl_cum   = (1 + bl_portfolio).cumprod()
msci_cum = (1 + MSCI_World).cumprod()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(bl_cum.index,   bl_cum.values,   label="Black-Litterman (BL)", color="steelblue", linewidth=2)
ax.plot(msci_cum.index, msci_cum.values, label="MSCI World",           color="tomato",    linewidth=2)
ax.set_title("Black-Litterman vs MSCI World — Cumulative Returns")
ax.set_ylabel("Growth of $1")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Risk metrics comparison
rm_bl   = risk_metrics_table(bl_portfolio, risk_free_rate=risk_free, periods_per_year=ANNUAL, window=6)
rm_msci = risk_metrics_table(MSCI_World,   risk_free_rate=risk_free, periods_per_year=ANNUAL, window=6)

comparison = pd.concat([rm_bl, rm_msci], axis=1)
comparison.columns = ["Black-Litterman", "MSCI World"]
print("\nBenchmarking — Risk Metrics Comparison:")
display(comparison)
