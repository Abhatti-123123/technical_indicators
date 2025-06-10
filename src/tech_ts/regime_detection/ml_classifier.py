import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------------------
# ML-based Regime Classifier
# -----------------------------------------

def train_classifier(X, y, test_size=0.3, random_state=42, **rf_kwargs):
    """
    Train a Random Forest classifier on given features.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target labels (regimes).
        test_size (float): Proportion for test split.
        random_state (int): Seed for reproducibility.
        rf_kwargs: Additional keyword args for RandomForestClassifier.
        
    Returns:
        model: Fitted classifier.
        X_test, y_test: Test data for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(random_state=random_state, **rf_kwargs)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def predict_classifier(model, X):
    """
    Predict labels for given features using trained model.
    """
    return model.predict(X)


def evaluate_classifier(y_true, y_pred):
    """
    Compute performance metrics.
    
    Returns:
        dict: accuracy, precision, recall, f1.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

# -----------------------------------------
# Unit Tests for ML Classifier
# -----------------------------------------

if __name__ == "__main__":
    import numpy as np
    import unittest

    class TestMLClassifier(unittest.TestCase):
        def setUp(self):
            # Synthetic separable data for two regimes
            np.random.seed(0)
            X0 = np.random.normal(0, 1, size=(100, 2))
            X1 = np.random.normal(5, 1, size=(100, 2))
            X = pd.DataFrame(np.vstack([X0, X1]), columns=['f1', 'f2'])
            y = pd.Series([0] * 100 + [1] * 100)
            self.X = X
            self.y = y

        def test_train_and_predict(self):
            model, X_test, y_test = train_classifier(
                self.X, self.y, test_size=0.2, random_state=0, n_estimators=10
            )
            preds = predict_classifier(model, X_test)
            # Ensure prediction length matches test set
            self.assertEqual(len(preds), len(y_test))
            # Check high accuracy on simple data
            metrics = evaluate_classifier(y_test, preds)
            self.assertGreater(metrics['accuracy'], 0.9)

    unittest.main(verbosity=2)
