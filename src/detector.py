"""
Anomaly Detector - ML-based transaction risk scoring using Isolation Forest.

This module provides a modular Detector class that uses scikit-learn's
Isolation Forest algorithm optimized for high-dimensional tabular data.
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Default model path
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


class Detector:
    """
    Isolation Forest-based anomaly detector for financial transactions.
    
    Optimized parameters based on scikit-learn best practices:
    - n_estimators=150: Balance between accuracy and speed
    - max_samples=256: Efficient subsampling (default 'auto' behavior)
    - contamination=0.01: ~1% expected fraud rate in banking
    
    Attributes:
        model: Trained IsolationForest model
        scaler: StandardScaler for feature normalization
        label_encoders: Dict of LabelEncoders for categorical features
    """
    
    # Feature columns for model training
    NUMERICAL_FEATURES = ["amount", "hour", "day_of_week"]
    CATEGORICAL_FEATURES = ["location", "merchant_type"]
    
    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: int = 256,
        contamination: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize the detector with Isolation Forest parameters.
        
        Args:
            n_estimators: Number of isolation trees
            max_samples: Samples per tree (256 is efficient default)
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._is_fitted = False
    
    def _extract_features(self, transaction: dict) -> dict:
        """
        Extract features from a transaction for model input.
        
        Args:
            transaction: Transaction dictionary with standard fields
            
        Returns:
            Dictionary of extracted features
        """
        # Parse timestamp
        timestamp_str = transaction.get("timestamp", "")
        try:
            from datetime import datetime
            # Handle ISO format with Z suffix
            timestamp_str = timestamp_str.replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(timestamp_str)
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
        except (ValueError, AttributeError):
            hour = 12  # Default to noon
            day_of_week = 0  # Default to Monday
        
        return {
            "amount": float(transaction.get("amount", 0)),
            "hour": hour,
            "day_of_week": day_of_week,
            "location": transaction.get("location", "Unknown"),
            "merchant_type": transaction.get("merchant_type", "Unknown")
        }
    
    def _prepare_dataframe(self, transactions: list[dict]) -> pd.DataFrame:
        """
        Convert transaction list to feature DataFrame.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        features = [self._extract_features(txn) for txn in transactions]
        return pd.DataFrame(features)
    
    def _encode_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Encode categorical features and scale numerical ones.
        
        Args:
            df: DataFrame with raw features
            fit: Whether to fit encoders/scaler (True for training)
            
        Returns:
            Numpy array of encoded features
        """
        df = df.copy()
        
        # Encode categorical features
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    le = self.label_encoders.get(col)
                    if le is not None:
                        df[col] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df[col] = 0
        
        # Get all feature columns
        all_features = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES
        feature_matrix = df[all_features].values
        
        # Scale features
        if fit:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        else:
            feature_matrix = self.scaler.transform(feature_matrix)
        
        return feature_matrix
    
    def fit(self, transactions: list[dict]) -> "Detector":
        """
        Train the Isolation Forest model on transaction data.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Self for method chaining
        """
        df = self._prepare_dataframe(transactions)
        X = self._encode_features(df, fit=True)
        self.model.fit(X)
        self._is_fitted = True
        return self
    
    def score(self, transaction: dict) -> float:
        """
        Score a single transaction for anomaly probability.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Anomaly score between 0 and 1 (higher = more anomalous)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before scoring. Call fit() first.")
        
        df = self._prepare_dataframe([transaction])
        X = self._encode_features(df, fit=False)
        
        # Get raw anomaly score from Isolation Forest
        # decision_function returns negative values for anomalies
        raw_score = self.model.decision_function(X)[0]
        
        # Convert to 0-1 scale (higher = more anomalous)
        # Typical raw scores range from -0.5 to 0.5
        normalized_score = 1 / (1 + np.exp(raw_score * 5))  # Sigmoid transformation
        
        return float(np.clip(normalized_score, 0, 1))
    
    def score_batch(self, transactions: list[dict]) -> list[float]:
        """
        Score multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of anomaly scores between 0 and 1
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before scoring. Call fit() first.")
        
        df = self._prepare_dataframe(transactions)
        X = self._encode_features(df, fit=False)
        
        raw_scores = self.model.decision_function(X)
        normalized_scores = 1 / (1 + np.exp(raw_scores * 5))
        
        return [float(np.clip(s, 0, 1)) for s in normalized_scores]
    
    def get_features_for_explanation(self, transaction: dict) -> tuple[np.ndarray, list[str]]:
        """
        Get scaled features for SHAP explainability.
        
        IMPORTANT: Returns SCALED features to match model training.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Tuple of (scaled_feature_array, feature_names)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before getting features.")
        
        # Get the properly encoded and scaled features (same as model input)
        df = self._prepare_dataframe([transaction])
        X_scaled = self._encode_features(df, fit=False)  # This returns scaled features
        
        # Feature names match the order in _encode_features
        feature_names = ["amount", "hour", "day_of_week", "is_weekend", 
                         "location_encoded", "merchant_type_encoded"]
        
        # Return the scaled features (what the model actually sees)
        return X_scaled[0], feature_names
    
    def get_training_features(self, transactions: list[dict]) -> np.ndarray:
        """
        Get scaled training features for SHAP background data.
        
        Args:
            transactions: List of training transactions
            
        Returns:
            Scaled feature matrix for SHAP
        """
        df = self._prepare_dataframe(transactions)
        X_scaled = self._encode_features(df, fit=False)  # Returns scaled features
        
        return X_scaled
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model (defaults to models/isolation_forest.joblib)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = str(DEFAULT_MODEL_DIR / "isolation_forest.joblib")
        
        model_state = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "is_fitted": self._is_fitted
        }
        joblib.dump(model_state, path)
        return path
    
    def load_model(self, path: Optional[str] = None) -> "Detector":
        """
        Load a trained model from disk.
        
        Args:
            path: Path to model file (defaults to models/isolation_forest.joblib)
            
        Returns:
            Self for method chaining
        """
        if path is None:
            path = str(DEFAULT_MODEL_DIR / "isolation_forest.joblib")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_state = joblib.load(path)
        self.model = model_state["model"]
        self.scaler = model_state["scaler"]
        self.label_encoders = model_state["label_encoders"]
        self._is_fitted = model_state["is_fitted"]
        
        return self


def train_detector(num_samples: int = 10000) -> Detector:
    """
    Train a new detector on simulated data and save it.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        Trained Detector instance
    """
    from src.simulator import generate_training_data
    
    print(f"Generating {num_samples} training transactions...")
    training_data = generate_training_data(num_samples)
    
    print("Training Isolation Forest model...")
    detector = Detector()
    detector.fit(training_data)
    
    print("Saving model...")
    path = detector.save_model()
    print(f"Model saved to: {path}")
    
    return detector


if __name__ == "__main__":
    # Train and save a model
    detector = train_detector()
    
    # Test scoring
    test_transaction = {
        "transaction_id": "txn_test001",
        "user_id": "user_0001",
        "amount": 15000,  # High amount
        "location": "Mumbai",
        "timestamp": "2026-01-14T03:30:00Z",  # Unusual hour
        "merchant_type": "electronics"
    }
    
    score = detector.score(test_transaction)
    print(f"\nTest transaction anomaly score: {score:.4f}")
