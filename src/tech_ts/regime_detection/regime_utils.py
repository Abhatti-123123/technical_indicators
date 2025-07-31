import numpy as np, pandas as pd, inspect, scipy.stats as ss
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -----------------------------------------------------------------------------
#  Forward return & IC-based regime weights
# -----------------------------------------------------------------------------
def forward_return(series: pd.Series, horizon: int) -> pd.Series:
    return series.pct_change(periods=horizon).shift(-horizon)

def calc_regime_weights(signal_df: pd.DataFrame,
                        fwd_ret: pd.Series,
                        regime: pd.Series,
                        min_obs: int = 15,
                        ridge_alpha: float = 3.0) -> dict:
    """
    Returns {regime: weight_vector} in the same column order as signal_df.
    Weight = Ridge-β * sign(IC)   (IC = Spearman ρ)
    Falls back to equal weights if too few observations.
    """
    w = {}
    p = signal_df.shape[1]
    X = signal_df.values
    y = fwd_ret.values

    def safe_ic(a, b):
        if np.all(a == a[0]) or np.all(b == b[0]):  # constant
            return 0.0
        return ss.spearmanr(a, b)[0] or 0.0

    for r in np.unique(regime):
        m = (regime == r).values & np.isfinite(y)
        if m.sum() < min_obs:
            w[r] = np.ones(p) / p
            continue
        ic = np.array([safe_ic(X[m, j], y[m]) for j in range(p)])
        scaler = StandardScaler(with_mean=False)  # preserve sparsity sign
        Xr = scaler.fit_transform(X[m])
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
        ridge.fit(Xr, y[m])
        wvec = ridge.coef_ * np.sign(ic)
        if np.allclose(wvec, 0):
            wvec = np.ones(p)
        w[r] = wvec / np.sum(np.abs(wvec))
    return w

def apply_weights(
        signal_df: pd.DataFrame,
        regime: pd.Series,
        weight_dict: dict,
        thresh: float = 0.3
) -> pd.Series:
    """
    • Dot-product signals with their regime-specific weights.
    • Divide by the L1-norm of that weight vector → output is in [-1, +1].
    • Fire a ±1 only if |score| ≥ thresh (e.g. 0.7).
    """
    n_sig   = signal_df.shape[1]
    default = np.ones(n_sig) / n_sig

    # Build weight matrix row-wise
    W = np.vstack([weight_dict.get(r, default) for r in regime.values])

    # L1-norm per regime row to normalise dot product
    l1 = np.sum(np.abs(W), axis=1, keepdims=True)
    l1[l1 == 0] = 1                       # guard against div/0
    norm_W = W / l1

    score = np.einsum('ij,ij->i', signal_df.values, norm_W)

    pos = pd.Series(0, index=signal_df.index)
    pos[ score >=  thresh] =  1
    pos[ score <= -thresh] = 0
    return pos
