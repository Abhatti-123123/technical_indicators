# regime_detection.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# -----------------------------------------
# Hidden Markov Model Regime Detection
# -----------------------------------------

def fit_hmm_regimes(X, n_states=2, covariance_type='full', n_iter=100, random_state=42):
    """
    Fit a Gaussian HMM to input data for regime detection.

    Parameters:
        X (pd.DataFrame or np.ndarray): Data (T x features) to fit HMM on.
        n_states (int): Number of hidden regimes.
        covariance_type (str): Covariance type for HMM ('full', 'diag').
        n_iter (int): Maximum iterations for training.
        random_state (int): Seed for reproducibility.

    Returns:
        model (GaussianHMM): Trained HMM model.
    """
    model = GaussianHMM(n_components=n_states,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=random_state)
    model.fit(X)
    return model


def predict_hmm_regimes(model, X):
    """
    Predict hidden regime states and probabilities using a trained HMM.

    Parameters:
        model (GaussianHMM): Fitted HMM model.
        X (pd.DataFrame or np.ndarray): Data to predict on.

    Returns:
        states (np.ndarray): Most likely state for each sample.
        probs (np.ndarray): State probability matrix (T x n_states).
    """
    states = model.predict(X)
    probs = model.predict_proba(X)
    return states, probs

# -----------------------------------------
# Unit Tests for Regime Detection
# -----------------------------------------

import unittest

class TestHMMRegimeDetection(unittest.TestCase):
    def setUp(self):
        # Simulate synthetic data: two regimes with different means
        np.random.seed(0)
        # Regime 0: mean 0, low var; Regime 1: mean 5, high var
        data0 = np.random.normal(0, 1, size=(100, 1))
        data1 = np.random.normal(5, 1, size=(100, 1))
        self.X = np.vstack([data0, data1])

    def test_fit_and_predict(self):
        # Fit HMM with 2 states
        model = fit_hmm_regimes(self.X, n_states=2)
        states, probs = predict_hmm_regimes(model, self.X)
        # Check lengths
        self.assertEqual(len(states), self.X.shape[0])
        self.assertEqual(probs.shape, (self.X.shape[0], 2))
        # Check probabilities sum to 1
        sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-6), "Probabilities do not sum to 1")

    def test_state_distribution(self):
        # Fit and check that states split roughly half-half
        model = fit_hmm_regimes(self.X, n_states=2)
        states, _ = predict_hmm_regimes(model, self.X)
        # Expect roughly 100 samples per state
        counts = np.bincount(states)
        self.assertEqual(len(counts), 2)
        self.assertTrue(abs(counts[0] - counts[1]) <= 50, "State counts too imbalanced")

if __name__ == "__main__":
    unittest.main(verbosity=2)

