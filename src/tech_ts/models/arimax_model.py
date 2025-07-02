"""
Minimal ARIMAX wrapper
======================

Functions
---------
fit_arimax(y, X, p=1, q=1)
    Fit SARIMAX with given (p,0,q) orders and exogenous regressors.
predict_arimax(model, X_test)
    Return in-sample or out-of-sample predictions aligned to X_test index.

Notes
-----
* Assumes `y` is already **stationary** (returns, diff, etc.).
* No seasonal component; set `order=(p,0,q)`.
* `enforce_stationarity` & `enforce_invertibility` are True so the fit fails
  fast if parameters imply an unstable process.
"""

from __future__ import annotations
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ────────────────────────────────────────────────────────────────
# Fit
# ────────────────────────────────────────────────────────────────
def fit_arimax(y: pd.Series,
               X: pd.DataFrame,
               p: int = 1,
               q: int = 1) -> SARIMAX:
    """
    Parameters
    ----------
    y : pd.Series
        Target series (already stationary).
    X : pd.DataFrame
        Exogenous regressors aligned to `y`.
    p, q : int
        AR and MA orders.

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResults
    """
    # Ensure alignment
    if isinstance(X, pd.Series):         # up-cast if the cleaner left 1 column
        X = X.to_frame()
    X = X.reindex(y.index)
    model = SARIMAX(
        y,
        exog=X,
        order=(p, 0, q),
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False)
    return model


# ────────────────────────────────────────────────────────────────
# Predict
# ────────────────────────────────────────────────────────────────
def predict_arimax(model, X_new: pd.DataFrame) -> pd.Series:
    """
    Forecast len(X_new) steps ahead using the fitted SARIMAX model.

    Parameters
    ----------
    model  : fitted SARIMAXResults
    X_new  : exogenous vars for the forecast horizon (must match training columns)

    Returns
    -------
    pd.Series of point forecasts, indexed like X_new
    """
    # ensure 2-D
    if isinstance(X_new, pd.Series):
        X_new = X_new.to_frame()

    steps = len(X_new)
    # statsmodels ≥0.13 has .forecast(); it’s safer than .predict w/ indices
    y_hat = model.forecast(steps=steps, exog=X_new)

    # restore the DatetimeIndex for downstream merging / plotting
    y_hat.index = X_new.index
    return y_hat.rename("arimax_pred")
