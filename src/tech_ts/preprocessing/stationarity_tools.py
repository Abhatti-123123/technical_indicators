"""
Stationarity diagnostics & transformations
------------------------------------------
Utility functions to (i) test ADF + KPSS, (ii) difference / log-diff until
stationary, and (iii) write a tidy report for logging.

Usage
-----
    from tech_ts.preprocessing.stationarity_tools import (
        stationarity_report, force_stationary
    )

    df_s = force_stationary(df_raw)          # auto-transformed copy
    report = stationarity_report(df_raw)     # DataFrame of p-values/flags
"""

from __future__ import annotations
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


# ────────────────────────────────────────────────────────────────
# Core helpers
# ────────────────────────────────────────────────────────────────
def _adf(series: pd.Series) -> Tuple[float, float]:
    stat, p, *_ = adfuller(series.dropna(), autolag="AIC")
    return stat, p


def _kpss(series: pd.Series) -> Tuple[float, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")        # KPSS warns on short series
        stat, p, *_ = kpss(series.dropna(), nlags="auto")
    return stat, p


def stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with ADF & KPSS p-values + a boolean 'stationary' flag
    (true iff ADF p<0.05 **and** KPSS p>0.10).
    """
    rows: List[Dict[str, float]] = []
    for col in df:
        adf_stat, adf_p = _adf(df[col])
        kpss_stat, kpss_p = _kpss(df[col])
        rows.append({
            "series":      col,
            "adf_p":       adf_p,
            "kpss_p":      kpss_p,
            "is_stationary": (adf_p < 0.05) and (kpss_p > 0.09),
        })
    return pd.DataFrame(rows).set_index("series")


def make_diff(series: pd.Series, log: bool = False) -> pd.Series:
    """
    First-difference or log-difference a series, preserving name & index.
    """
    if log:
        series = np.log(series)
    return series.diff().rename(series.name)


def force_stationary(df: pd.DataFrame,
                     max_diff: int = 2,
                     force_log: List[str] | None = None
                     ) -> pd.DataFrame:
    """
    Iterate (up to `max_diff` times) diff’ing non-stationary cols until they
    pass ADF<0.05 & KPSS>0.10.  Optionally log-diff specific columns.
    Returns a *new* DataFrame; does **not** mutate original.
    """
    force_log = force_log or []
    out = df.copy()

    for col in df.columns:
        x = out[col]
        log_flag = col in force_log
        for d in range(max_diff + 1):
            rep = stationarity_report(x.to_frame())
            if rep.loc[col, "is_stationary"]:
                break
            x = make_diff(x, log=log_flag and d == 0)   # log on first pass
        out[col] = x
    return out
