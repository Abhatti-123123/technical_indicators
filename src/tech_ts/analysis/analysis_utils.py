# indicator_diagnostics.py
# ─────────────────────────────────────────────────────────────────
import warnings
from itertools import combinations
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ────────────────────────────────────────────────────────────────
# 1. Line-plot helpers
# ────────────────────────────────────────────────────────────────
def plot_series(
    df: pd.DataFrame,
    title: str | None = None,
    figsize: Tuple[int, int] = (12, 6),
    grid: bool = True,
) -> plt.Figure:
    """Overlay any number of series on one axis."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    fig, ax = plt.subplots(figsize=figsize)
    df.plot(ax=ax)
    ax.set_title(title or "Indicator time-series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(grid, alpha=0.3)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────
# 2. ACF / PACF
# ────────────────────────────────────────────────────────────────
def acf_pacf(
    series: pd.Series,
    lags: int = 40,
    title: str | None = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Side-by-side ACF & PACF."""
    x = series.dropna().astype(float)
    if x.empty:
        raise ValueError("Series has no numeric data after dropna().")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_acf(x, lags=lags, ax=axes[0])
    plot_pacf(x, lags=lags, ax=axes[1], method="ywmle")
    fig.suptitle(title or f"ACF / PACF – {series.name}")
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────
# 3. Stationarity tests
# ────────────────────────────────────────────────────────────────
def stationarity(series: pd.Series) -> pd.DataFrame:
    """
    ADF (H0: unit root)  &  KPSS (H0: level-stationary).
    Returns tidy DataFrame with stats & p-values.
    """
    x = series.dropna().astype(float)
    adf_stat, adf_p, adf_lags, adf_n, adf_crit = adfuller(x, autolag="AIC")[:5]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(x, nlags="auto")

    return pd.DataFrame(
        index=["adf", "kpss"],
        data=dict(
            statistic=[adf_stat, kpss_stat],
            p_value=[adf_p, kpss_p],
            lags=[adf_lags, kpss_lags],
            n_obs=[adf_n, len(x)],
            crit_1pct=[adf_crit["1%"], kpss_crit["1%"]],
            crit_5pct=[adf_crit["5%"], kpss_crit["5%"]],
            crit_10pct=[adf_crit["10%"], kpss_crit["10%"]],
        ),
    )


# ────────────────────────────────────────────────────────────────
# 4. Cointegration checks
# ────────────────────────────────────────────────────────────────
def eg_pair(series1: pd.Series, series2: pd.Series) -> Dict[str, float]:
    """
    Engle-Granger two-step.  Returns statistic & p-value.
    """
    stat, p, _ = coint(series1.dropna(), series2.dropna())
    return {"eg_stat": stat, "p_value": p}


def pairwise_coint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Engle-Granger for **every pair** of columns in df.
    Returns DataFrame of p-values (lower = stronger cointegration).
    """
    cols = df.columns
    mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a, b in combinations(cols, 2):
        p = coint(df[a].dropna(), df[b].dropna())[1]
        mat.loc[a, b] = mat.loc[b, a] = p
    return mat


def johansen_test(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int | None = None,
) -> pd.DataFrame:
    """
    Johansen trace test for rank-r cointegration (multi-variate).
    det_order : 0 ⇒ no deterministic term; -1 ⇒ none & no intercept.
    """
    res = coint_johansen(df.dropna(), det_order, k_ar_diff or 1)
    out = pd.DataFrame(
        {
            "trace_stat": res.lr1,
            "crit_90": res.cvt[:, 0],
            "crit_95": res.cvt[:, 1],
            "crit_99": res.cvt[:, 2],
        },
        index=[f"rank ≤ {i}" for i in range(len(res.lr1))],
    )
    return out
