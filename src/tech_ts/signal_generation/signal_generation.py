import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------------
# Signal Generation Functions
# -----------------------------------------

def generate_rsi_signal(df, column='RSI', lower=30, upper=70):
    """
    Generate RSI-based trading signals:
    - Long (1) when RSI < lower
    - Neutral (0) when lower <= RSI <= upper
    - Short (-1) when RSI > upper
    """
    rsi = df[column]
    signal = pd.Series(0, index=df.index)
    signal[rsi < lower] = 1
    signal[rsi > upper] = -1
    return signal


def generate_macd_signal(df, macd_col='MACD_1', signal_col='MACD_2'):
    """
    Generate MACD crossover signals:
    - Long (1) when MACD crosses above signal line
    - Short (-1) when MACD crosses below signal line
    """
    macd = df[macd_col]
    sig = df[signal_col]
    crossover_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    crossover_down = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    signal = pd.Series(0, index=df.index)
    signal[crossover_up] = 1
    signal[crossover_down] = -1
    return signal


def generate_bollinger_signal(df, column='BollingerBands', lower=-1, upper=1):
    """
    Generate Bollinger Bands z-score signals:
    - Long (1) when z-score < lower
    - Neutral (0) when lower <= z-score <= upper
    - Short (-1) when z-score > upper
    """
    bb = df[column]
    signal = pd.Series(0, index=df.index)
    signal[bb < lower] = 1
    signal[bb > upper] = -1
    return signal


def generate_volatility_signal(df, column='Volatility', threshold=None):
    """
    Generate volatility-based regime signals:
    - High volatility (1) when vol > threshold
    - Low volatility (-1) otherwise

    If threshold is None, use median volatility.
    """
    vol = df[column]
    if threshold is None:
        threshold = vol.median()
    signal = pd.Series(-1, index=df.index)
    signal[vol > threshold] = 1
    return signal


def combine_signals(signals_dict, weights=None):
    """
    Combine multiple signals into a single composite signal.

    Parameters:
    - signals_dict: dict of pd.Series signals, e.g., {'rsi':..., 'macd':...}
    - weights: dict of weights for each signal; defaults to equal weight.

    Returns:
    - pd.Series: composite signal based on weighted sum, sign of sum.
    """
    df_signals = pd.DataFrame(signals_dict)
    if weights is None:
        weights = {col: 1/len(df_signals.columns) for col in df_signals.columns}
    # Weighted sum
    weighted_sum = sum(df_signals[col] * weights.get(col, 0) for col in df_signals.columns)
    composite = np.sign(weighted_sum).fillna(0)
    return composite


# ------------------------------------------------------------------
#  Train a reg-regularised logit on sign-of-forward-return
# ------------------------------------------------------------------
def train_nonlinear_weighter(X: pd.DataFrame,
                             fwd_ret: pd.Series,
                             degree: int = 2,
                             C: float   = 0.1):
    """
    X        : z-scored indicator signals (rows = dates, cols = indicators)
    fwd_ret  : forward % return aligned with X.index
    returns  : fitted sklearn Pipeline
    """
    y = (fwd_ret > 0).astype(int)               # binary ↑ / ↓
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler(with_mean=False)),  # keep sparsity sign
        ("logit", LogisticRegression(penalty="l2",
                                     C=C,
                                     max_iter=500,
                                     solver="lbfgs"))
    ])
    pipe.fit(X.values, y.values)
    return pipe


def score_to_position(proba: np.ndarray,
                      thresh: float = 0.3) -> np.ndarray:
    """
    proba  : model.predict_proba()[:, 1]  (P(↑))
    thresh : trade only when conviction > thresh
    returns: +1 / 0 / –1 vector
    """
    score = 2 * proba - 1                     # map [0,1] → [-1,+1]
    pos   = np.zeros_like(score, dtype=int)
    pos[ score >=  thresh] =  1
    pos[ score <= -thresh] = 0
    return pos

# -----------------------------------------
# Unit Tests for Signal Generation
# -----------------------------------------

import unittest

class TestSignalGeneration(unittest.TestCase):
    def setUp(self):
        # Create dummy data with known patterns
        dates = pd.date_range('2021-01-01', periods=10)
        self.df = pd.DataFrame(index=dates)
        # RSI pattern: [20, 25, 50, 75, 80,...]
        self.df['RSI'] = [20, 25, 50, 75, 80, 50, 25, 20, 75, 50]
        # MACD & signal: cross up at t=2, cross down at t=5
        self.df['MACD_1'] = [0, -1, 1, 2, 1, -1, -2, -1, 0, 1]
        self.df['MACD_2'] = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # Bollinger z-scores simple pattern
        self.df['BollingerBands'] = [-2, -1, 0, 1, 2, 0, -1, -2, 1, 0]
        # Volatility simple pattern
        self.df['Volatility'] = [1,2,3,4,5,4,3,2,1,2]

    def test_rsi_signal(self):
        sig = generate_rsi_signal(self.df, lower=30, upper=70)
        expected = [1,1,0,-1,-1,0,1,1,-1,0]
        self.assertListEqual(list(sig), expected)

    def test_macd_signal(self):
        sig = generate_macd_signal(self.df)
        # Only crossover down at index 1 and 2, down at 5 and 9
        expected = [0, -1, 1, 0, 0, -1, 0, 0, 0, 1]
        self.assertListEqual(list(sig), expected)

    def test_bollinger_signal(self):
        sig = generate_bollinger_signal(self.df, lower=-1, upper=1)
        expected = [1,0,0,0,-1,0,0,1,0,0]
        self.assertListEqual(list(sig), expected)

    def test_volatility_signal(self):
        sig = generate_volatility_signal(self.df, threshold=3)
        expected = [0,0,0,1,1,1,0,0,0,0]
        self.assertListEqual(list(sig), expected)

    def test_combine_signals_default_weights(self):
        signals = {'rsi': pd.Series([1, -1, 1], index=[0,1,2]),
                   'vol': pd.Series([1, 1, -1], index=[0,1,2])}
        comp = combine_signals(signals)
        # weighted equally: sums [2,0,0] => sign [1,0,0]
        expected = [1,0,0]
        self.assertListEqual(list(comp), expected)

    def test_combine_signals_custom_weights(self):
        signals = {'rsi': pd.Series([1, -1], index=[0,1]),
                   'vol': pd.Series([1, 1], index=[0,1])}
        weights = {'rsi': 0.75, 'vol': 0.25}
        comp = combine_signals(signals, weights)
        # weighted sums [1*0.75+1*0.25=1, -1*0.75+1*0.25=-0.5] => sign [1,-1]
        expected = [1,-1]
        self.assertListEqual(list(comp), expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)
