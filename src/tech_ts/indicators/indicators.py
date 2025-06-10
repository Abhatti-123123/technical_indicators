import pandas as pd
import numpy as np

# -----------------------------------------
# Indicator computations (modular and clear)
# -----------------------------------------

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_volatility(close, window=20):
    vol = close.pct_change().rolling(window=window).std()
    return vol

def compute_bollinger_bands(close, window=20):
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    bb = (close - ma) / (std + 1e-10)
    return bb

def rolling_zscore(series, window=252):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    zscore = (series - mean) / (std + 1e-10)
    return zscore

# -----------------------------------------
# Dictionary-based indicator addition
# -----------------------------------------

INDICATOR_FUNCTIONS = {
    'RSI': compute_rsi,
    'MACD': compute_macd,
    'Volatility': compute_volatility,
    'BollingerBands': compute_bollinger_bands
}

def add_indicator(df, indicator_name, **kwargs):
    """
    Add a new indicator column to the DataFrame dynamically.

    Parameters:
        df (pd.DataFrame): DataFrame containing price data.
        indicator_name (str): Indicator to compute.
        kwargs: Indicator-specific parameters.
    
    Returns:
        pd.DataFrame: DataFrame with indicator columns appended.
    """
    if indicator_name not in INDICATOR_FUNCTIONS:
        raise ValueError(f"Indicator '{indicator_name}' is not defined.")
    
    func = INDICATOR_FUNCTIONS[indicator_name]
    result = func(df['Close'], **kwargs)

    if isinstance(result, tuple):  # For indicators returning multiple series
        for i, series in enumerate(result, 1):
            df[f'{indicator_name}_{i}'] = series
    else:
        df[indicator_name] = result

    return df

# -----------------------------------------
# Unit tests
# -----------------------------------------

import unittest

class TestIndicators(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2020-01-01', periods=300)
        np.random.seed(0)
        prices = pd.Series(np.random.normal(loc=100, scale=5, size=300), index=dates)
        self.df = pd.DataFrame({'Close': prices})

    def test_rsi(self):
        rsi = compute_rsi(self.df['Close'])
        self.assertEqual(len(rsi), len(self.df))
        self.assertFalse(rsi.isna().all(), "RSI calculation failed, all values are NaN")

    def test_macd(self):
        macd, signal = compute_macd(self.df['Close'])
        self.assertEqual(len(macd), len(self.df))
        self.assertEqual(len(signal), len(self.df))
        self.assertFalse(macd.isna().all(), "MACD calculation failed, all values are NaN")

    def test_volatility(self):
        vol = compute_volatility(self.df['Close'])
        self.assertEqual(len(vol), len(self.df))
        self.assertFalse(vol.isna().all(), "Volatility calculation failed, all values are NaN")

    def test_bollinger_bands(self):
        bb = compute_bollinger_bands(self.df['Close'])
        self.assertEqual(len(bb), len(self.df))
        self.assertFalse(bb.isna().all(), "Bollinger Bands calculation failed, all values are NaN")

    def test_rolling_zscore(self):
        z = rolling_zscore(self.df['Close'])
        self.assertEqual(len(z), len(self.df))
        self.assertFalse(z.isna().all(), "Rolling Z-score calculation failed, all values are NaN")

    def test_dynamic_indicator_addition(self):
        df_new = add_indicator(self.df.copy(), 'RSI', window=14)
        self.assertIn('RSI', df_new.columns)
        self.assertFalse(df_new['RSI'].isna().all(), "Dynamic indicator addition failed, RSI is NaN")

        df_new = add_indicator(df_new, 'MACD', fast=12, slow=26, signal=9)
        self.assertIn('MACD_1', df_new.columns)
        self.assertIn('MACD_2', df_new.columns)

if __name__ == "__main__":
    unittest.main(verbosity=2)
