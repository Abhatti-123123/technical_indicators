"""
Regime‑detection utilities based on Hidden Markov Models (HMMs).

Key upgrades 2025‑06‑15
----------------------
* **Flexible feature input**   – you can now feed an arbitrary `DataFrame` of
  features instead of being locked to `[return, 5‑day σ]`.
* **Version‑agnostic covar kw** – works with both old (`min_covar`) and new
  (`reg_covar`) hmmlearn releases.
* **Optional standardisation** – z‑score features inside the helper so each
  dimension contributes comparably.
* **Full / diag covariance toggle**.
* **Unit tests fixed** – typo (`fit_hmm_regimes` → `fit_hmm_returns`) and a
    small synthetic example moved to the new API.
"""

from __future__ import annotations
import inspect
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ────────────────────────────────────────────────────────────────
# Main helpers
# ────────────────────────────────────────────────────────────────

def _get_covar_kw() -> str:
    """Return the kwargs name for small‑sample covariance regularisation."""
    return (
        "reg_covar"
        if "reg_covar" in inspect.signature(GaussianHMM.__init__).parameters
        else "min_covar"
    )


def _standardise(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column‑wise z‑score without altering shape."""
    mu, sig = X.mean(0), X.std(0) + eps
    return (X - mu) / sig


def fit_hmm_returns(
    price_or_features: pd.Series | pd.DataFrame,
    *,
    n_states: int = 2,
    n_init: int = 5,
    max_iter: int = 200,
    reg_covar: float = 1e-4,
    covariance_type: str = "full",
    standardise: bool = True,
    random_state: int | None = None,
) -> GaussianHMM:
    """Fit an HMM regime model.

    Series input: builds [return, 5‑day σ, 21‑day σ] features.
    DataFrame input: uses supplied features directly.
    """
    # Build feature matrix
    if isinstance(price_or_features, pd.Series):
        # daily return
        ret   = price_or_features.pct_change().fillna(0).to_numpy("float64").reshape(-1, 1)
        # 5-day realized volatility
        vol5  = (
            price_or_features.pct_change().rolling(5).std().fillna(0)
            .to_numpy("float64").reshape(-1, 1)
        )
        # 21-day realized volatility
        vol21 = (
            price_or_features.pct_change().rolling(21).std().fillna(0)
            .to_numpy("float64").reshape(-1, 1)
        )
        X = np.hstack([ret, vol5, vol21])
    elif isinstance(price_or_features, pd.DataFrame):
        X = price_or_features.to_numpy("float64")
    else:
        raise TypeError("Input must be a pandas Series (price) or DataFrame (features)")

    # Standardise features if requested
    if standardise:
        X = _standardise(X)

    # Choose appropriate covar keyword
    cov_kw = _get_covar_kw()
    best_ll, best_model = -np.inf, None

    # Random restarts
    for seed in range(n_init):
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=max_iter,
            random_state=(random_state if seed == 0 else seed),
            verbose=False,
            **{cov_kw: reg_covar},
        )
        hmm.fit(X)
        if hasattr(hmm, 'monitor_') and hmm.monitor_.converged:
            ll = hmm.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, hmm

    if best_model is None:
        raise RuntimeError("HMM failed to converge in any restart")

    return best_model


def predict_hmm(model: GaussianHMM, X: np.ndarray | pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (state_path, posterior_probs) for a fitted HMM."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy("float64")
    return model.predict(X), model.predict_proba(X)