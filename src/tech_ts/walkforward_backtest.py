import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid GUI thread errors
import matplotlib.pyplot as plt
import logging
import scipy.stats as ss
from sklearn.linear_model import Ridge
from pathlib import Path

from tech_ts.data.fetch_data import download_prices
from tech_ts.data.constants    import TICKERS, INDICATOR_LIST
from tech_ts.data.date_config  import START_DATE, END_DATE
from tech_ts.indicators.indicators import add_indicator
from tech_ts.signal_generation.signal_generation import (
    generate_rsi_signal,
    generate_macd_signal,
    generate_bollinger_signal,
    generate_volatility_signal,
    combine_signals,
    train_nonlinear_weighter,
    score_to_position
)
from tech_ts.regime_detection.regime_detection import fit_hmm_returns, predict_hmm
from tech_ts.regime_detection.regime_utils import forward_return, calc_regime_weights, apply_weights
from tech_ts.regime_detection.ml_classifier import train_classifier, predict_classifier, evaluate_classifier
from tech_ts.parameter_tuning.rolling_tuning import rolling_parameter_tuning
from tech_ts.back_testing.back_testing import backtest_strategy
from tech_ts.preprocessing.stationarity_tools import (
    stationarity_report, force_stationary
)

from tech_ts.vol_sizing.volatility_scaled_signal_sizer import  VolatilityScaledSignalSizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
import warnings
import logging
import os
from scipy.stats import ConstantInputWarning            # for SciPy IC warnings
from sklearn.exceptions import ConvergenceWarning       # if sklearn is in the stack

# ────────────────────────────────────────────────────────────────
# 1.  Silence hmmlearn-specific warnings
# ────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message=r".*Model is not converging.*"
)
warnings.filterwarnings(
    "ignore",
    message=r".*rows of transmat_ have zero sum.*"
)

# ────────────────────────────────────────────────────────────────
# 2.  Silence SciPy’s “ConstantInputWarning” from Spearman IC
# ────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    category=ConstantInputWarning
)

# ────────────────────────────────────────────────────────────────
# 3.  (Optional) silence generic convergence warnings
# ────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# ✅ Suppress sklearn + joblib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# ────────────────────────────────────────────────────────────────
# 4.  Turn down hmmlearn’s own logger to ERROR
# ────────────────────────────────────────────────────────────────
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# def _safe_spearman(a, b):
#     if np.all(a == a[0]) or np.all(b == b[0]):   # constant check
#         return 0.0
#     return ss.spearmanr(a, b)[0] or 0.0


# # ---------- forward-return ----------
# def calc_forward_return(prices, horizon):
#     """Simple pct-change over 'horizon' days."""
#     return prices.pct_change(periods=horizon).shift(-horizon)

# ---------- median holding period ----------
def estimate_median_holding(signals_dict):
    """
    Crude estimate: average length (in bars) between signal direction changes.
    Returns int >=1.
    """
    lengths = []
    for sig in signals_dict.values():
        flips = (np.sign(sig).diff() != 0).astype(int)
        seg_lens = np.diff(np.flatnonzero(np.append(flips.values, 1)))
        if len(seg_lens):
            lengths.append(np.median(seg_lens))
    return int(max(1, np.median(lengths))) if lengths else 1


def apply_volatility_sizing(
    price_series: pd.Series,
    base_pos: pd.Series,
    *,
    fast_lambda: float = 0.60,
    slow_lambda: float = 0.97,
    fast_weight: float = 0.7,
    slow_weight: float = 0.3,
    max_leverage: float = 3.0,
    vol_floor: float = 1e-3,
    leverage_smoothing_alpha: float | None = 0.2,
) -> pd.Series:
    """
    Walk the series bar by bar, applying VolatilityScaledSignalSizer to the base position.
    Returns a scaled position series aligned with base_pos.
    """

    sizer = VolatilityScaledSignalSizer(
        fast_lambda=fast_lambda,
        slow_lambda=slow_lambda,
        fast_weight=fast_weight,
        slow_weight=slow_weight,
        max_leverage=max_leverage,
        vol_floor=vol_floor,
        leverage_smoothing_alpha=None,
    )

    scaled_pos = []
    returns = price_series.pct_change()  # daily returns

    for idx in base_pos.index:
        base = base_pos.loc[idx]
        ret = returns.loc[idx]
        if pd.isna(ret):
            ret = 0.0  # warmup; could also skip first few if preferred
        scaled = sizer.update_and_scale(base, ret)
        scaled_pos.append(scaled)

    return pd.Series(scaled_pos, index=base_pos.index, name="vol_scaled_position")


import time

def safe_download_prices(*args, max_retries=5, backoff_base=2, **kwargs):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            df = download_prices(*args, **kwargs)
            if df is None or df.empty:
                raise ValueError(f"Downloaded empty data for {args} {kwargs}")
            return df
        except Exception as e:
            last_exc = e
            wait = backoff_base ** (attempt - 1)
            logger.warning(f"Download attempt {attempt} failed for {args}: {e!r}; retrying in {wait}s")
            time.sleep(wait)
    # final failure: propagate with context
    logger.warning(f"Failed to download after {max_retries} attempts: {last_exc!r}")
    return None

# -----------------------------------------
# Walk-Forward Backtesting Orchestration
# -----------------------------------------


def run_walkforward_pipeline(
    vix,
    vxv,
    tickers=TICKERS,
    start=START_DATE,
    end=END_DATE,
    window_size=252,
    step_size=7,
    test_size=7,
    transaction_cost=0.001
):
    """
    Orchestrates the full walk-forward backtest pipeline:
    1. Fetch data
    2. For each rolling window:
       a) Compute indicators
       b) Tune indicator parameters
       c) Generate signals based on tuned parameters
       d) Detect regimes via HMM and ML classifier
       e) Combine signals
       f) Backtest strategy for the window
    3. Aggregate results across windows
    """
    # 1. Fetch data
    prices = safe_download_prices(tickers, start=start, end=end)
    if prices is None:
        return None, None
    prices = prices.dropna()
    close = prices.iloc[:, 0]  # e.g., SPY by default
    print(close.index[0])
    
    vix_df = pd.concat([vix, vxv], axis=1).reindex(close.index).ffill()
    vix_df["VIX_ratio"] = vix_df["VIX"] / vix_df["VXV"]

    # Store aggregated metrics
    all_metrics = []
    results_list = []

    n = len(close)
    start_idx = 0

    close_new = close.to_frame(name='Close')
    for ind in INDICATOR_LIST:
        close_new = add_indicator(close_new, ind)\
    
    indicators_raw = close_new.drop(columns="Close")

    OUTDIR = Path("stationarity")
    OUTDIR.mkdir(exist_ok=True)
    
    # 1.  Write a one-off CSV report (good for PR / tracking)
    stationarity_report(indicators_raw).to_csv(f"{OUTDIR}/stationarity_pre_{tickers[0]}.csv")

    # 2.  Transform to stationarity – example: log-diff Volatility only
    indicators = force_stationary(indicators_raw, force_log=['Volatility'])

    # 3.  Save post-check report
    stationarity_report(indicators).to_csv(f"{OUTDIR}/stationarity_post_{tickers[0]}.csv")

    while start_idx + window_size + test_size <= n:
        # Define indices for train and test
        train_idx = range(start_idx, start_idx + window_size)
        test_idx = range(start_idx + window_size, start_idx + window_size + test_size)
        full_idx = range(0, start_idx + window_size)

        train_data = close.iloc[train_idx].to_frame(name='Close')
        test_data = close.iloc[test_idx].to_frame(name='Close')
        full_data = close.iloc[full_idx].to_frame(name='Close')

        # 2a. Tune parameters for each indicator on train set
        param_grids = {
            'RSI': {'window': [10, 14, 20]},
            'MACD': {'fast': [10,12], 'slow': [26,30], 'signal': [9]},
            # 'BollingerBands': {'window': [20, 30]},
            'Volatility': {'window': [10, 20]}
        }
        best_params = {}
        tune_train_size = window_size - test_size
        tune_test_size  = test_size
        tune_step       = test_size  # one iteration; could be smaller if you want more folds

        for ind, grid in param_grids.items():
            # ensures tune_train_size + tune_test_size == len(train_data)
            tune_results = rolling_parameter_tuning(
                data=train_data,
                indicator_name=ind,
                param_grid=grid,
                train_size=tune_train_size,
                test_size=tune_test_size,
                step=tune_step
            )
            # now tune_results will have exactly one row, so best_params exists
            best_params[ind] = tune_results['best_params'].iloc[-1]

        # 2b. Compute indicators on test set with best params
        df = test_data.copy()
        df_train = train_data.copy()
        df_full = full_data.copy()
        for ind, params in best_params.items():
            df = add_indicator(df, ind, **params)
            df_train = add_indicator(df_train, ind, **params)
            df_full = add_indicator(df_full, ind, **params)

        # indicators_raw = df_train.drop(columns="Close")

        # 2c. Generate signals
        signals_test = {}
        signals_test['rsi'] = generate_rsi_signal(df, column='RSI', lower=30, upper=70)
        signals_test['macd'] = generate_macd_signal(df)
        # signals_test['bb'] = generate_bollinger_signal(df, column='BollingerBands', lower=-1, upper=1)
        signals_test['vol'] = generate_volatility_signal(df, column='Volatility')

        signals_train = {}
        signals_train['rsi'] = generate_rsi_signal(df_train, column='RSI', lower=30, upper=70)
        signals_train['macd'] = generate_macd_signal(df_train)
        # signals_train['bb'] = generate_bollinger_signal(df_train, column='BollingerBands', lower=-1, upper=1)
        signals_train['vol'] = generate_volatility_signal(df_train, column='Volatility')

        signals_full = {}
        signals_full['rsi'] = generate_rsi_signal(df_full, column='RSI', lower=30, upper=70)
        signals_full['macd'] = generate_macd_signal(df_full)
        # signals_train['bb'] = generate_bollinger_signal(df_train, column='BollingerBands', lower=-1, upper=1)
        signals_full['vol'] = generate_volatility_signal(df_full, column='Volatility')

         # --- Regime Detection using HMM on multi-features ---
        # Build feature DataFrames for HMM input
        def build_feats(series: pd.Series) -> pd.DataFrame:
            ret = series.pct_change().fillna(0)
            vol5 = series.pct_change().rolling(5).std().fillna(0)
            vol21= series.pct_change().rolling(21).std().fillna(0)
            vix_pct = vix_df['VIX'].pct_change().reindex(series.index).fillna(0)
            ratio  = vix_df['VIX_ratio'].reindex(series.index).fillna(1)
            return pd.concat([ret.rename('ret'), vol5.rename('vol5'), vol21.rename('vol21'),
                              vix_pct.rename('dVIX'), ratio.rename('VIX_ratio')], axis=1)

        feats_train = (
            build_feats(train_data['Close'])          # create features
            .reindex(train_data.index)              # align to TRAIN dates
        )

        feats_test  = (
            build_feats(test_data['Close'])
            .reindex(test_data.index)               # align to TEST dates
        )

        feats_full  = (
            build_feats(full_data['Close'])           # ← note: variable is data_full
            .reindex(full_data.index)               # align to FULL dates
        )
        # # Fit HMM and predict regimes
        hmm_model = fit_hmm_returns(feats_full, n_states=3)
        reg_train, _ = predict_hmm(hmm_model, feats_full)
        reg_test,  _ = predict_hmm(hmm_model, feats_test)

        reg_train = pd.Series(reg_train, index=feats_full.index, name="regime")
        reg_train = reg_train.reindex(train_data.index)

        reg_train = pd.Series(reg_train, index=train_data.index)
        reg_test  = pd.Series(reg_test,  index=test_data.index)

        # Build z-scored signal matrices for weight calculation

        sig_cols = ['rsi','macd', 'vol']
        Xtrain = pd.DataFrame({c: signals_train[c] for c in sig_cols}, index=train_data.index)
        XFull  = pd.DataFrame({c: signals_full[c]  for c in sig_cols}, index=full_data.index)
        Xtest  = pd.DataFrame({c: signals_test[c]  for c in sig_cols}, index=test_data.index)

        Xtrain = pd.concat([feats_train, Xtrain], axis=1, join="outer")
        Xtest  = pd.concat([feats_test,  Xtest],  axis=1, join="outer")
        XFull  = pd.concat([feats_full,  XFull],  axis=1, join="outer")

        mu, sigma = Xtrain.mean(), Xtrain.std().replace(0,1e-12)
        Xtrain = (Xtrain - mu)/sigma
        XFull = (XFull - mu)/sigma
        Xtest  = (Xtest  - mu)/sigma

        # >>> 3.  Forward target & regime weights <<<
        horizon   = estimate_median_holding(signals_full)                     # pick what matches your avg holding
        fwd_ret   = forward_return(full_data['Close'], horizon)
        fwd_ret_train = forward_return(train_data['Close'], horizon)
        w_dict    = calc_regime_weights(Xtrain, fwd_ret_train, reg_train)

        model     = train_nonlinear_weighter(XFull, fwd_ret,
                                            degree=2,
                                            C=0.1)                      # or 0.05

        # ------------------------------------------------------------
        # 5.  Score TEST slice  → composite position
        # ------------------------------------------------------------
        proba     = model.predict_proba(Xtest.values)[:, 1]
        # composite = pd.Series(
        #     score_to_position(proba, thresh=0.3),
        #     index=Xtest.index,
        #     name="nl_weight_position"
        # )
        raw_signal = pd.Series(
            score_to_position(proba, thresh=0.3),
            index=Xtest.index,
            name="base_position"
        )

        reg_composite = apply_weights(Xtest, reg_test, w_dict)

        base_pos = 0.5 * reg_composite + 0.5 * raw_signal

        # apply volatility-based sizing overlay
        composite = apply_volatility_sizing(
            price_series=close.iloc[test_idx],
            base_pos=base_pos,
            fast_lambda=0.60,
            slow_lambda=0.97,
            fast_weight=0.7,
            slow_weight=0.3,
            max_leverage=3.0,
            vol_floor=1e-3,
            leverage_smoothing_alpha=0.2,
        )

        # >>> 4.  Composite position series for TEST <<<
        

        # 2e. Combine signals with equal weights
        # composite = combine_signals(signals_test)

        # 2f. Backtest composite strategy on test period
        results_df, metrics = backtest_strategy(close.iloc[test_idx], composite, transaction_cost)
        # metrics.update(
        #     {'hmm_accuracy': np.mean(probs.argmax(axis=1) == states),
        #      'ml_accuracy': ml_metrics['accuracy']}
        # )

        win_start = close.index[test_idx][0]
        metrics['window_start'] = win_start
        # Store
        all_metrics.append(metrics)
        results_list.append(results_df.assign(window_start=close.index[test_idx][0]))

        start_idx += step_size

    # Aggregate metrics into DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.set_index('window_start', inplace=True)
    all_results = pd.concat(results_list)
    all_results = all_results[~all_results.index.duplicated(keep="first")]  # kill exact dupes
    return all_results, metrics_df


def main():

    # Fetch VIX & VXV once for full period
    # vix = download_prices(["^VIX"], start=START_DATE, end=END_DATE).squeeze("columns").rename("VIX")
    # vxv = download_prices(["^VXV"], start=START_DATE, end=END_DATE).squeeze("columns").rename("VXV")
    try:
        vix = safe_download_prices(["^VIX"], start=START_DATE, end=END_DATE).squeeze("columns").rename("VIX")
        vxv = safe_download_prices(["^VXV"], start=START_DATE, end=END_DATE).squeeze("columns").rename("VXV")
    except Exception as e:
        logger.error("Critical: VIX/VXV download failed, aborting: %s", e)
        return  # or sys.exit(1)

    for ticker in TICKERS:
        logger.info(f"\n===== Running walkforward for {ticker} =====")
        results, metrics = run_walkforward_pipeline(vix=vix, vxv=vxv, tickers=[ticker])  # note: pass as list
        if results is None:
            logger.info("Data fetch failed")
            continue
        dupes = results.index[results.index.duplicated()]
        print("Duplicate timestamps:", len(dupes))
        assert results.index.to_series().diff().dt.days.mode().iloc[0] == 1, \
       "daily_returns is NOT daily – drop the √252 if this fails"
        # === Cumulative PnL ===
        results["cum_pnl"] = (1 + results["strategy_return"]).cumprod()

        # === Drawdown from simulated equity ===
        results["rolling_max"] = results["cum_pnl"].cummax()
        results["drawdown"] = results["cum_pnl"] / results["rolling_max"] - 1
        results["max_drawdown"] = results["drawdown"].expanding().min()

        # === Reconstruct full equity curve and return stream ===
        # Assumes per-window PnL starts where previous left off
        daily_returns = results["strategy_return"]  # assume uniform return per day

        agg_volatility = daily_returns.std() * np.sqrt(252)
        agg_sharpe = daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252)

        # CAGR from overall cumulative return
        final_cum_return = results["cum_pnl"].iloc[-1]
        total_days = len(daily_returns)
        agg_cagr = (1 + final_cum_return) ** (252 / total_days) - 1

        # Final max drawdown
        agg_drawdown = results["drawdown"].min()

        print(f"{ticker}  | 99-pctile abs return:", results["strategy_return"].abs().quantile(0.99))

        # Aggregate others
        agg_trades = metrics["total_trades"].sum()
        # ── Filter out useless windows (zero trades or NaNs) ──────────────────────────
        mask = (
            (metrics["total_trades"] > 0) &                  # traded at least once
            metrics["hit_rate"].notna() &
            metrics["avg_holding_period"].notna()
        )

        if not mask.any():          # nothing to aggregate – bail early
            agg_hit, agg_holding = np.nan, np.nan
        else:
            w = metrics.loc[mask, "total_trades"].to_numpy()
            assert w.sum() > 0, "All weights are zero – cannot compute weighted average."

            agg_hit     = np.average(metrics.loc[mask, "hit_rate"].to_numpy(),          weights=w)
            agg_holding = np.average(metrics.loc[mask, "avg_holding_period"].to_numpy(),weights=w)

        logger.info("==== AGGREGATED PERFORMANCE METRICS ====")
        logger.info(f"Final Cumulative PnL     : {final_cum_return:.4f}")
        logger.info(f"Aggregated CAGR          : {agg_cagr:.4f}")
        logger.info(f"Aggregated Sharpe        : {agg_sharpe:.4f}")
        logger.info(f"Aggregated Volatility    : {agg_volatility:.4f}")
        logger.info(f"Aggregated Max Drawdown  : {agg_drawdown:.4f}")
        logger.info(f"Total Trades             : {agg_trades}")
        logger.info(f"Weighted Avg Hit Rate    : {agg_hit:.3f}")
        logger.info(f"Weighted Holding Period  : {agg_holding:.2f}")

        # Plot Cumulative Metrics
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results["cum_pnl"], label="cum_pnl", color='blue')
        plt.plot(results.index, results["drawdown"], label="drawdown", color='orange')
        # metrics["cum_sharpe"] = metrics["total_return"].expanding().mean() / (metrics["total_return"].expanding().std() + 1e-10) * np.sqrt(252)
        # plt.plot(metrics.index, metrics["cum_sharpe"], label="cum_sharpe", color='green')

        plt.title(f"{ticker}: Cumulative Metrics")
        plt.xlabel("window_start")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        # Where this script lives
        here = Path(__file__).resolve().parent

        # Go two levels up
        root = here.parent.parent          # same as here.parents[1] if you prefer
        plot_path = root / f"metrics_{ticker}_plot.png"
        csv_path = root / f"walkforward_metrics_{ticker}.csv"
        results_path = root / f"results_{ticker}.csv"
        plt.savefig(plot_path)
        logger.info(f"Saved plot to {plot_path}")
        results.to_csv(results_path)
        metrics.to_csv(csv_path)


if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# """
# Continuous (one-bar-ahead) training + trading loop.
# Refits the model every `retrain_every` days using ALL data seen so far,
# then takes a position for the next day.  Results are back-tested once
# at the end to give a true walk-forward equity curve.

# Brutally simple by design; optimise later if it’s too slow.
# """
# import warnings, logging, os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy.stats import ConstantInputWarning
# from sklearn.exceptions import ConvergenceWarning

# # ─── Internal modules ──────────────────────────────────────────
# from tech_ts.data.fetch_data            import download_prices
# from tech_ts.data.constants             import TICKERS, INDICATOR_LIST
# from tech_ts.data.date_config           import START_DATE, END_DATE
# from tech_ts.indicators.indicators      import add_indicator
# from tech_ts.signal_generation.signal_generation import (
#     generate_rsi_signal, generate_macd_signal,
#     generate_volatility_signal, score_to_position,
#     train_nonlinear_weighter,
# )
# from tech_ts.preprocessing.stationarity_tools import (
#     stationarity_report, force_stationary
# )
# from tech_ts.back_testing.back_testing  import backtest_strategy
# from tech_ts.regime_detection.regime_utils import forward_return

# # ─── Logger & warning hygiene ─────────────────────────────────
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s %(levelname)s %(message)s")

# warnings.filterwarnings("ignore",
#     message=r".*Model is not converging.*")               # hmmlearn
# warnings.filterwarnings("ignore",
#     category=ConstantInputWarning)                        # SciPy
# warnings.filterwarnings("ignore",
#     category=ConvergenceWarning)                          # sklearn
# logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# # ──────────────────────────────────────────────────────────────
# # Helper: feature builder (returns, vol, VIX term structure…)
# # ──────────────────────────────────────────────────────────────
# def build_feats(price_series: pd.Series, vix_block: pd.DataFrame) -> pd.DataFrame:
#     ret    = price_series.pct_change().fillna(0)
#     vol5   = price_series.pct_change().rolling(5).std().fillna(0)
#     vol21  = price_series.pct_change().rolling(21).std().fillna(0)
#     d_vix  = vix_block["VIX"].pct_change().fillna(0)
#     ratio  = vix_block["VIX_ratio"]
#     return pd.concat([ret.rename("ret"),
#                       vol5.rename("vol5"),
#                       vol21.rename("vol21"),
#                       d_vix.rename("dVIX"),
#                       ratio.rename("VIX_ratio")], axis=1)

# # ──────────────────────────────────────────────────────────────
# def estimate_median_holding(signals_dict) -> int:
#     """Approx. median segment length across signal directions."""
#     lengths = []
#     for sig in signals_dict.values():
#         flips = (np.sign(sig).diff() != 0).astype(int)
#         seg   = np.diff(np.flatnonzero(np.append(flips.values, 1)))
#         if len(seg):
#             lengths.append(np.median(seg))
#     return int(max(1, np.median(lengths))) if lengths else 1

# # ──────────────────────────────────────────────────────────────
# def run_online_pipeline(
#     vix, vxv, tickers=TICKERS,
#     start=START_DATE, end=END_DATE,
#     warmup=252,                # start trading after one year of data
#     retrain_every=1,           # full refit cadence (set 5 or 21 for speed)
#     transaction_cost=0.001
# ):
#     # 1. Download prices & pre-compute all indicator columns once
#     prices = download_prices(tickers, start, end).dropna()
#     close  = prices.iloc[:, 0]                       # assume 1 ticker only
#     df     = close.to_frame("Close")
#     for ind in INDICATOR_LIST:
#         df = add_indicator(df, ind)

#     # Optional: stationarity check / transform
#     OUTDIR = Path("stationarity_online"); OUTDIR.mkdir(exist_ok=True)
#     stationarity_report(df.drop(columns="Close")).to_csv(
#         OUTDIR / f"stationarity_raw_{tickers[0]}.csv")
#     df.iloc[:, 1:] = force_stationary(df.iloc[:, 1:])   # only indicators
#     stationarity_report(df.drop(columns="Close")).to_csv(
#         OUTDIR / f"stationarity_post_{tickers[0]}.csv")

#     # VIX features
#     vix_df = pd.concat([vix, vxv], axis=1).reindex(df.index).ffill()
#     vix_df["VIX_ratio"] = vix_df["VIX"] / vix_df["VXV"]

#     # Containers for positions and per-day debug metrics
#     position = pd.Series(index=df.index, dtype=float)

#     # Cached feature μ/σ for standardisation
#     mu, sigma = None, None

#     # 2. MAIN WALK-FORWARD LOOP (one bar ahead)
#     for t in range(warmup, len(df) - 1):            # last bar untradable
#         if (t - warmup) % retrain_every == 0:
#             train_slice = df.iloc[:t]               # up-to-yesterday
#             feats  = build_feats(train_slice["Close"],
#                                  vix_df.iloc[:t])
#             # z-score
#             mu, sigma = feats.mean(), feats.std().replace(0, 1e-12)
#             Z = (feats - mu) / sigma

#             # --- forward target & signals (for weighter training) ---
#             horizon   = estimate_median_holding({   # crude: use price dir flips
#                 "dummy": train_slice["Close"].pct_change().fillna(0)
#             })
#             fwd_ret   = forward_return(train_slice["Close"], horizon)

#             # Non-linear ridge weighter (replace w/ partial_fit if big data)
#             model = train_nonlinear_weighter(Z, fwd_ret,
#                                              degree=2, C=0.1)

#         # --- ONE-DAY-AHEAD PREDICTION -----------------------------
#         today_feat  = build_feats(df["Close"].iloc[[t]],
#                                   vix_df.iloc[[t]])
#         today_Z     = (today_feat - mu) / sigma
#         proba       = model.predict_proba(today_Z.values)[:, 1][0]
#         position.iloc[t] = score_to_position(proba, thresh=0.3)

#     # 3. Back-test the whole generated position vector
#     results_df, metrics = backtest_strategy(close, position, transaction_cost)
#     return results_df, metrics

# # ──────────────────────────────────────────────────────────────
# def main():
#     vix = download_prices(["^VIX"], start=START_DATE, end=END_DATE)\
#             .squeeze("columns").rename("VIX")
#     vxv = download_prices(["^VXV"], start=START_DATE, end=END_DATE)\
#             .squeeze("columns").rename("VXV")

#     here  = Path(__file__).resolve().parent
#     root  = here.parent.parent

#     for ticker in TICKERS:
#         logger.info(f"===== ONLINE walk-forward for {ticker} =====")
#         res, met = run_online_pipeline(vix, vxv, tickers=[ticker])

#         # Equity curve, drawdown, summary
#         res["cum_pnl"] = (1 + res["strategy_return"]).cumprod()
#         res["rolling_max"] = res["cum_pnl"].cummax()
#         res["drawdown"]    = res["cum_pnl"] / res["rolling_max"] - 1
#         res["max_drawdown"]= res["drawdown"].expanding().min()

#         # Aggregate top-level performance
#         daily = res["strategy_return"]
#         agg_vol    = daily.std() * np.sqrt(252)
#         agg_sharpe = daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
#         final_eq   = res["cum_pnl"].iloc[-1]
#         cagr       = (1 + final_eq) ** (252 / len(daily)) - 1
#         max_dd     = res["drawdown"].min()

#         logger.info(f"CAGR       : {cagr:.4f}")
#         logger.info(f"Sharpe     : {agg_sharpe:.4f}")
#         logger.info(f"Volatility : {agg_vol:.4f}")
#         logger.info(f"Max DD     : {max_dd:.4f}")

#         # Plot & save
#         plt.figure(figsize=(12,6))
#         plt.plot(res.index, res["cum_pnl"], label="cum_pnl")
#         plt.plot(res.index, res["drawdown"], label="drawdown")
#         plt.title(f"{ticker} – Online Walk-Forward Equity & DD")
#         plt.legend(); plt.grid(True); plt.tight_layout()
#         plot_path    = root / f"online_metrics_{ticker}.png"
#         res_path     = root / f"online_results_{ticker}.csv"
#         plt.savefig(plot_path); res.to_csv(res_path)
#         logger.info(f"Saved results → {res_path}")
#         logger.info(f"Saved plot    → {plot_path}")

# if __name__ == "__main__":
#     main()

