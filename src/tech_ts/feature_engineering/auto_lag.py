# tech_ts/feature_engineering/auto_lag.py
from statsmodels.graphics.tsaplots import pacf
import pandas as pd
import numpy as np

def pacf_lags(series: pd.Series,
              max_lag: int = 30,
              alpha: float = 0.05) -> list[int]:
    """
    Return lags where PACF is significantly different from 0
    at (1-alpha) confidence.  Uses Yule-Walker 'yw' method.
    """
    vals, conf = pacf(series.dropna(), nlags=max_lag,
                      alpha=alpha, method="ywmle")
    sig = np.where((conf[:, 0] > 0) | (conf[:, 1] < 0))[0]   # indexes = lags
    sig = sig[sig > 0]      # skip lag 0
    return sig.tolist()


def build_lag_map(df: pd.DataFrame,
                  max_lag: int = 30,
                  alpha: float = 0.05) -> dict[str, list[int]]:
    return {col: pacf_lags(df[col], max_lag=max_lag, alpha=alpha)
            for col in df.columns}


def add_lags(df: pd.DataFrame,
             lag_map: dict[str, list[int]],
             drop_na: bool = True) -> pd.DataFrame:
    """
    Return a new DataFrame with extra columns of specified lags.

    Parameters
    ----------
    df        : original (already stationary) DataFrame
    lag_map   : {"col_name": [lag1, lag2, ...], ...}
    drop_na   : drop rows with NaNs introduced by shifting (default True)

    Example
    -------
    lag_map = {"RSI": [1],
               "MACD_1": [1,2,3],
               "Volatility": [5, 20]}

    df_lagged = add_lags(indicators, lag_map)
    """
    out = df.copy()

    for col, lags in lag_map.items():
        if col not in df.columns:
            raise KeyError(f"{col} not in DataFrame")
        for k in lags:
            out[f"{col}_lag{k}"] = df[col].shift(k)

    return out.dropna() if drop_na else out