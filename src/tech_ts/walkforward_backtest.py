import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid GUI thread errors
import matplotlib.pyplot as plt
import logging

from tech_ts.data.fetch_data import download_prices
from tech_ts.data.constants    import TICKERS
from tech_ts.data.date_config  import START_DATE, END_DATE
from tech_ts.indicators.indicators import add_indicator
from tech_ts.signal_generation.signal_generation import (
    generate_rsi_signal,
    generate_macd_signal,
    generate_bollinger_signal,
    generate_volatility_signal,
    combine_signals
)
from tech_ts.regime_detection.regime_detection import fit_hmm_regimes, predict_hmm_regimes
from tech_ts.regime_detection.ml_classifier import train_classifier, predict_classifier, evaluate_classifier
from tech_ts.parameter_tuning.rolling_tuning import rolling_parameter_tuning
from tech_ts.back_testing.back_testing import backtest_strategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------
# Walk-Forward Backtesting Orchestration
# -----------------------------------------

def run_walkforward_pipeline(
    tickers=TICKERS,
    start=START_DATE,
    end=END_DATE,
    window_size=252,
    step_size=63,
    test_size=63,
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
    prices = download_prices(tickers, start=start, end=end)
    prices = prices.dropna()
    close = prices.iloc[:, 0]  # e.g., SPY by default
    print(close.index[0])
    
    # Store aggregated metrics
    all_metrics = []
    results_list = []

    n = len(close)
    start_idx = 0

    while start_idx + window_size + test_size <= n:
        # Define indices for train and test
        train_idx = range(start_idx, start_idx + window_size)
        test_idx = range(start_idx + window_size, start_idx + window_size + test_size)

        train_data = close.iloc[train_idx].to_frame(name='Close')
        test_data = close.iloc[test_idx].to_frame(name='Close')

        # 2a. Tune parameters for each indicator on train set
        param_grids = {
            'RSI': {'window': [10, 14, 20]},
            'MACD': {'fast': [10,12], 'slow': [26,30], 'signal': [9]},
            'BollingerBands': {'window': [20, 30]},
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
        for ind, params in best_params.items():
            df = add_indicator(df, ind, **params)
            df_train = add_indicator(df_train, ind, **params)

        # 2c. Generate signals
        signals_test = {}
        signals_test['rsi'] = generate_rsi_signal(df, column='RSI', lower=30, upper=70)
        signals_test['macd'] = generate_macd_signal(df)
        signals_test['bb'] = generate_bollinger_signal(df, column='BollingerBands', lower=-1, upper=1)
        signals_test['vol'] = generate_volatility_signal(df, column='Volatility')

        signals_train = {}
        signals_train['rsi'] = generate_rsi_signal(df_train, column='RSI', lower=30, upper=70)
        signals_train['macd'] = generate_macd_signal(df_train)
        signals_train['bb'] = generate_bollinger_signal(df_train, column='BollingerBands', lower=-1, upper=1)
        signals_train['vol'] = generate_volatility_signal(df_train, column='Volatility')

        # 2d. Regime detection (HMM)
        X_ind_train = pd.DataFrame(signals_train)
        X_ind_test = pd.DataFrame(signals_test)
        model = fit_hmm_regimes(X_ind_train.values, n_states=3)
        train_states, _ = predict_hmm_regimes(model, X_ind_train.values)   # pseudo-labels for train
        states, probs = predict_hmm_regimes(model, X_ind_test.values)
        df['regime_hmm'] = states
        # ML classifier on same features
        # print("Class distribution:", np.bincount(train_states))
        # print("Unique classes:", np.unique(train_states, return_counts=True))
        clf, X_test, y_test = train_classifier(X_ind_train, train_states)
        ml_preds = predict_classifier(clf, X_ind_test)
        df['regime_ml'] = ml_preds
        ml_metrics = evaluate_classifier(states, ml_preds)

        # 2e. Combine signals with equal weights
        composite = combine_signals(signals_test)

        # 2f. Backtest composite strategy on test period
        results_df, metrics = backtest_strategy(close.iloc[test_idx], composite, transaction_cost)
        metrics.update(
            {'hmm_accuracy': np.mean(probs.argmax(axis=1) == states),
             'ml_accuracy': ml_metrics['accuracy']}
        )

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
    return all_results, metrics_df


def main():
    from pathlib import Path

    for ticker in TICKERS:
        logger.info(f"\n===== Running walkforward for {ticker} =====")
        results, metrics = run_walkforward_pipeline(tickers=[ticker])  # note: pass as list
        print(results.head(20))
        print(results[results['strategy_return'].isna()].index)
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

        # Aggregate others
        agg_trades = metrics["total_trades"].sum()
        agg_hit = np.average(metrics["hit_rate"].dropna(), weights=metrics["total_trades"] + 1e-10)
        agg_holding = np.average(metrics["avg_holding_period"], weights=metrics["total_trades"] + 1e-10)

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
        plot_path = Path(f"metrics_{ticker}_plot.png")
        plt.savefig(plot_path)
        logger.info(f"Saved plot to {plot_path}")

        metrics.to_csv(f"walkforward_metrics_{ticker}.csv")


if __name__ == "__main__":
    main()
