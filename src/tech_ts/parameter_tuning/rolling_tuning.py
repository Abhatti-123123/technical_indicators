import pandas as pd
import numpy as np
from itertools import product
from tech_ts.indicators.indicators import INDICATOR_FUNCTIONS

# -----------------------------------------
# Rolling Window Parameter Tuning
# -----------------------------------------

def default_metric(series, returns):
    """
    Default metric: Pearson correlation between lagged series and returns.

    Parameters:
    - series (pd.Series): Indicator series.
    - returns (pd.Series): Forward returns.

    Returns:
    - float: correlation coefficient.
    """
    return series.shift(1).corr(returns)


def rolling_parameter_tuning(
    data,
    indicator_name,
    param_grid,
    train_size,
    test_size,
    step,
    metric_func=default_metric
):
    """
    Perform rolling window parameter tuning for a given indicator.

    Parameters:
    - data (pd.DataFrame): DataFrame with 'Close' price series.
    - indicator_name (str): Name of the indicator in INDICATOR_FUNCTIONS.
    - param_grid (dict): Parameter grid, e.g., {'window': [10,14,20]}.
    - train_size (int): Number of observations in training window.
    - test_size (int): Number of observations in testing window.
    - step (int): Step size to move the rolling window.
    - metric_func (callable): Function(series, returns) -> metric float.

    Returns:
    - pd.DataFrame: Results with columns ['train_start', 'train_end', 'test_start', 'test_end', 'best_params', 'best_metric'].
    """
    n = len(data)
    results = []
    returns = data['Close'].pct_change().shift(-1)

    # Prepare parameter combinations
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]

    # Rolling windows
    start = 0
    while start + train_size + test_size <= n:
        train_idx = range(start, start + train_size)
        test_idx = range(start + train_size, start + train_size + test_size)

        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        test_returns = returns.iloc[test_idx]

        best_metric = -np.inf
        best_params = None

        # Grid search for this window
        for params in combinations:
            func = INDICATOR_FUNCTIONS[indicator_name]
            series = func(train_data['Close'], **params)
            # If tuple, take first series
            if isinstance(series, tuple):
                series = series[0]
            # Compute metric on test data
            full_series = func(data['Close'], **params)
            if isinstance(full_series, tuple):
                full_series = full_series[0]
            test_series = full_series.iloc[test_idx]
            metric = metric_func(test_series, test_returns)
            if metric is not None and metric > best_metric:
                best_metric = metric
                best_params = params

        results.append({
            'train_start': data.index[start],
            'train_end': data.index[start + train_size - 1],
            'test_start': data.index[start + train_size],
            'test_end': data.index[start + train_size + test_size - 1],
            'best_params': best_params,
            'best_metric': best_metric
        })

        start += step

    return pd.DataFrame(results)

# -----------------------------------------
# Unit Tests for Rolling Tuning
# -----------------------------------------

import unittest

class TestRollingTuning(unittest.TestCase):
    def setUp(self):
        # Synthetic price data: linear increasing
        dates = pd.date_range('2020-01-01', periods=30)
        prices = pd.Series(np.linspace(100, 130, 30), index=dates)
        self.data = pd.DataFrame({'Close': prices})

    def test_rolling_tuning_basic(self):
        # Tune RSI window parameter
        param_grid = {'window': [2, 3]}
        df_results = rolling_parameter_tuning(
            data=self.data,
            indicator_name='RSI',
            param_grid=param_grid,
            train_size=10,
            test_size=5,
            step=5
        )
        # Expect two rolling windows: [0-9]/[10-14], [5-14]/[15-19]
        self.assertEqual(len(df_results), 4)
        # Check columns exist
        for col in ['train_start', 'train_end', 'test_start', 'test_end', 'best_params', 'best_metric']:
            self.assertIn(col, df_results.columns)
        # best_params should be one of the grid combos
        for params in df_results['best_params']:
            self.assertIn(params['window'], [2, 3])

if __name__ == '__main__':
    unittest.main(verbosity=2)
