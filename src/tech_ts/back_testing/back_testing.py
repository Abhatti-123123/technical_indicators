import pandas as pd
import numpy as np

# -----------------------------------------
# Strategy Backtesting and Performance Metrics
# -----------------------------------------

def backtest_strategy(prices, signals, transaction_cost=0.0, annualization=252):
    """
    Backtest a strategy given price series and discrete signals.

    Parameters:
    - prices (pd.Series): Price series indexed by date.
    - signals (pd.Series): Strategy signals (-1, 0, 1) indexed by date.
    - transaction_cost (float): Cost per trade as fraction of trade notional.
    - annualization (int): Number of trading periods in a year (e.g., 252).

    Returns:
    - pd.DataFrame: DataFrame with 'strategy_return' and 'cumulative_return'.
    - dict: Performance metrics.
    """
    assert isinstance(prices, pd.Series), "prices must be a pandas Series"
    assert isinstance(signals, pd.Series), "signals must be a pandas Series"
    # Compute period returns
    returns = prices.pct_change().fillna(0)

    # Align signals
    sig_events = signals.replace(0, np.nan)
    # positions = signals.shift(1).fillna(0)
    positions  = sig_events.shift(1).ffill().fillna(0)

    # Transaction costs: cost when position changes
    trades = (positions != positions.shift()).astype(int).fillna(0)
    trade_cost = trades * transaction_cost * 0

    # Strategy returns net of transaction costs
    strat_returns = positions * returns
    strat_returns[trades != 0] -= trade_cost[trades != 0]

    # Cumulative returns
    cum_returns = (1 + strat_returns).cumprod() - 1

    # Performance metrics
    total_return = cum_returns.iloc[-1]
    # CAGR
    periods = len(strat_returns)
    cagr = (1 + total_return) ** (annualization / periods) - 1 if periods > 0 else np.nan
    # Annualized volatility
    vol = strat_returns.std() * np.sqrt(annualization)
    # Sharpe ratio (assume risk-free ~0)
    sharpe = strat_returns.mean() / (strat_returns.std() + 1e-10) * np.sqrt(annualization)
    # Max drawdown
    running_max = cum_returns.cummax()
    drawdown = cum_returns - running_max
    max_drawdown = drawdown.min()
    # Total trades
    total_trades = int(trades.sum())
    # Hit rate: fraction of positive strategy returns when in market (positions !=0)
    mask = positions != 0
    hit_rate = strat_returns[mask].gt(0).sum() / mask.sum() if mask.sum() > 0 else np.nan
    # Average holding period (in periods)
    holdings = positions.copy()
    holdings[holdings != 0] = 1
    # Compute lengths of consecutive ones
    groups = (holdings != holdings.shift()).cumsum()
    lengths = holdings.groupby(groups).sum()[holdings.groupby(groups).first().eq(1)]
    avg_holding = lengths.mean() if not lengths.empty else 0

    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'hit_rate': hit_rate,
        'avg_holding_period': avg_holding
    }

    results_df = pd.DataFrame({'strategy_return': strat_returns, 'cumulative_return': cum_returns, 'positions': positions, 'trade_ccost': trade_cost})
    return results_df, metrics

# -----------------------------------------
# Unit Tests for Backtesting
# -----------------------------------------

if __name__ == '__main__':
    import unittest

    class TestBacktesting(unittest.TestCase):
        def setUp(self):
            # Simple price series: linear uptrend
            dates = pd.date_range('2020-01-01', periods=5)
            prices = pd.Series([100, 110, 120, 130, 140], index=dates)
            # Always long
            signals_long = pd.Series([1,1,1,1,1], index=dates)
            # Alternating signals: trade each day
            signals_alt = pd.Series([1,-1,1,-1,1], index=dates)
            self.prices = prices
            self.sig_long = signals_long
            self.sig_alt = signals_alt

        def test_always_long(self):
            df, metrics = backtest_strategy(self.prices, self.sig_long, transaction_cost=0)
            # Total return = (140/100 -1) = 0.4
            self.assertAlmostEqual(metrics['total_return'], 0.4)
            # No trades costless
            self.assertEqual(metrics['total_trades'], 0)
            # Hit rate = 1.0
            self.assertEqual(metrics['hit_rate'], 1.0)

        def test_transaction_costs(self):
            # With costs, alternating signals incur trades
            df, metrics = backtest_strategy(self.prices, self.sig_alt, transaction_cost=0.01)
            # At least 4 trades (4 sign changes)
            self.assertEqual(metrics['total_trades'], 4)
            # Metrics dict contains keys
            for key in ['cagr', 'sharpe', 'max_drawdown']:
                self.assertIn(key, metrics)

    unittest.main(verbosity=2)
