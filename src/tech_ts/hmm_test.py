import numpy as np, pandas as pd, warnings
from hmmlearn.hmm import GaussianHMM
from tech_ts.data.fetch_data       import download_prices
from tech_ts.data.date_config      import START_DATE, END_DATE
from tech_ts.regime_detection.regime_detection import fit_hmm_returns
from scipy.stats import ConstantInputWarning
from sklearn.exceptions import ConvergenceWarning

# ──────────────────────────── warning hygiene
warnings.filterwarnings("ignore", message=r".*Model is not converging.*")
warnings.filterwarnings("ignore", message=r".*rows of transmat_.*")
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import inspect
def make_hmm(n_components, **kwargs):
    """Return GaussianHMM with the right *covar* keyword for this hmmlearn version."""
    cov_kw = "reg_covar" if "reg_covar" in inspect.signature(GaussianHMM.__init__).parameters else "min_covar"
    return GaussianHMM(n_components=n_components,
                       covariance_type="full",
                       **{cov_kw: 1e-4},
                       **kwargs)

# ──────────────────────────── parameters you can tweak
ticker      = "SPY"
start, end  = START_DATE, END_DATE     # project-level dates
train_len   = 252 * 3                  # 3-year train
test_len    = 63                       # 3-month test
step_len    = 63                       # slide 3 months
horizon     = 3
n_states    = 3

dll_thresh  = 4        # scaled for 63-day test
dur_thresh  = 10       # ≥ 2 trading weeks
mu_gap_mult = 0.3      # |μi-μj| ≥ 0.3·σ

# ──────────────────────────── load prices
prices = download_prices([ticker], start=start, end=end)
close  = prices.iloc[:, 0].dropna()

# ──────────────────────────── build feature matrix once

ret  = close.pct_change().fillna(0)
vol5 = close.pct_change().rolling(5 ).std().fillna(0)
vol21= close.pct_change().rolling(21).std().fillna(0)

import yfinance as yf

# ───────── pull VIX & VXV ─────────
# ───────── pull VIX & VXV ─────────
# ^VXV is the 3-month VIX future.  If your data vendor uses ^VIX3M, change the ticker.
vix_ser = (
    download_prices(["^VIX"], start=start, end=end)
    .squeeze("columns")            # -> Series
    .rename("VIX")
)

vxv_ser = (
    download_prices(["^VXV"], start=start, end=end)
    .squeeze("columns")
    .rename("VXV")
)

# align to equity dates, forward-fill short gaps
vix_df = (
    pd.concat([vix_ser, vxv_ser], axis=1)
      .reindex(close.index)        # same DatetimeIndex as SPY
      .ffill()
)

vix_df["VIX_ratio"] = vix_df["VIX"] / vix_df["VXV"]

# ───────── expand feature matrix ─────────
X_full = np.column_stack([
    ret,                        # daily return
    vol5,                       # 5-day realised vol
    vol21,                      # 21-day realised vol
    vix_df["VIX"].pct_change().fillna(0).values,  # ΔVIX %
    vix_df["VIX_ratio"].fillna(0).values          # term-structure slope
])

# standardise
X_full = (X_full - X_full.mean(0)) / (X_full.std(0) + 1e-12)


# standardise (mean 0, std 1) – crucial for small windows
X_full = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-12)

fwd_ret = close.pct_change(horizon).shift(-horizon)

# ──────────────────────────── walk-forward diagnostics
records, ptr = [], 0
while ptr + train_len + test_len <= len(close):
    tr = slice(ptr, ptr + train_len)
    te = slice(ptr + train_len, ptr + train_len + test_len)

    X_tr, X_te = X_full[tr], X_full[te]
    price_tr   = close.iloc[tr]

    hmm1 = make_hmm(1).fit(X_tr)          # 1-state baseline
    hmm3 = make_hmm(n_states).fit(X_tr)   # 3-state candidate

    dll    = hmm3.score(X_te) - hmm1.score(X_te)

    st_tr  = hmm3.predict(X_tr)
    mean_dur = pd.Series(st_tr).groupby(
        pd.Series(st_tr).diff().ne(0).cumsum()).size().mean()

    st_te  = hmm3.predict(X_te)
    cnt    = np.bincount(st_te, minlength=n_states)
    fwd_te = fwd_ret.iloc[te].values
    mu     = [np.nanmean(fwd_te[st_te == s]) for s in range(n_states)]
    gaps   = [abs(mu[i] - mu[j]) for i in range(n_states) for j in range(i+1, n_states)]
    diff_ok = any(g >= mu_gap_mult * np.nanstd(fwd_te) for g in gaps)

    passed = (dll >= dll_thresh) and (cnt > 0).all() and diff_ok and (mean_dur >= dur_thresh)

    records.append({
        "train_start": close.index[tr][0],
        "dll": dll,
        "mean_dur": mean_dur,
        **{f"cnt_s{s}": cnt[s] for s in range(n_states)},
        "best_gap": max(gaps),
        "pass": passed
    })

    ptr += step_len

# ──────────────────────────── results
df = pd.DataFrame(records).set_index("train_start")
pd.set_option("display.float_format", "{:,.2f}".format)
print(df)
print("\nSummary:\n", df["pass"].value_counts()
      .rename({True: "slices pass", False: "slices fail"}))
