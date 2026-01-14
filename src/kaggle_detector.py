"""
Kaggle-trained Anomaly Detector - Trained on real Credit Card Fraud dataset.

Uses the PCA-transformed features (V1-V28) plus Amount from the Kaggle
European Credit Card Fraud dataset for high-recall fraud detection.
"""

import csv
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Default model path
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "creditcard.csv"


class KaggleDetector:
    """
    Isolation Forest detector trained on Kaggle Credit Card Fraud dataset.
    
    Uses the 28 PCA-transformed features (V1-V28) plus Amount for
    high-dimensional anomaly detection on real fraud patterns.
    
    Attributes:
        model: Trained IsolationForest model
        scaler: StandardScaler for feature normalization
    """
    
    # Features from the Kaggle dataset
    PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # V1 to V28
    FEATURE_COLUMNS = PCA_FEATURES + ["Amount"]  # 29 features total
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int = 512,
        contamination: float = 0.00173,  # Actual fraud rate in Kaggle data
        random_state: int = 42
    ):
        """
        Initialize the Kaggle detector.
        
        Args:
            n_estimators: Number of isolation trees (higher for complex data)
            max_samples: Samples per tree
            contamination: Fraud rate (0.173% in Kaggle dataset)
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit_from_csv(self, csv_path: Optional[str] = None) -> "KaggleDetector":
        """
        Train on the Kaggle Credit Card CSV file.
        
        Args:
            csv_path: Path to creditcard.csv
            
        Returns:
            Self for method chaining
        """
        if csv_path is None:
            csv_path = str(DEFAULT_DATA_PATH)
        
        print(f"Loading Kaggle dataset from: {csv_path}")
        
        # Load features in chunks for memory efficiency
        features = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature_row = [float(row[col]) for col in self.FEATURE_COLUMNS]
                features.append(feature_row)
        
        X = np.array(features)
        print(f"Loaded {len(X):,} transactions with {X.shape[1]} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        print("Training Isolation Forest model...")
        self.model.fit(X_scaled)
        self._is_fitted = True
        print("Training complete!")
        
        return self
    
    def score(self, pca_features: dict) -> float:
        """
        Score a transaction using PCA features.
        
        Args:
            pca_features: Dict with V1-V28 and Amount keys
            
        Returns:
            Anomaly score between 0 and 1 (higher = more anomalous)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")
        
        # Extract features in order
        feature_row = [pca_features.get(col, 0.0) for col in self.FEATURE_COLUMNS]
        X = np.array([feature_row])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get raw anomaly score
        raw_score = self.model.decision_function(X_scaled)[0]
        
        # Convert to 0-1 scale (higher = more anomalous)
        normalized_score = 1 / (1 + np.exp(raw_score * 5))
        
        return float(np.clip(normalized_score, 0, 1))
    
    def score_batch(self, pca_features_list: list[dict]) -> list[float]:
        """
        Score multiple transactions.
        
        Args:
            pca_features_list: List of dicts with V1-V28 and Amount
            
        Returns:
            List of anomaly scores
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")
        
        features = []
        for pca_features in pca_features_list:
            feature_row = [pca_features.get(col, 0.0) for col in self.FEATURE_COLUMNS]
            features.append(feature_row)
        
        X = np.array(features)
        X_scaled = self.scaler.transform(X)
        
        raw_scores = self.model.decision_function(X_scaled)
        normalized_scores = 1 / (1 + np.exp(raw_scores * 5))
        
        return [float(np.clip(s, 0, 1)) for s in normalized_scores]
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model
            
        Returns:
            Path where model was saved
        """
        if path is None:
            DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = str(DEFAULT_MODEL_DIR / "kaggle_isolation_forest.joblib")
        
        model_state = {
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self._is_fitted
        }
        joblib.dump(model_state, path)
        print(f"Model saved to: {path}")
        return path
    
    def load_model(self, path: Optional[str] = None) -> "KaggleDetector":
        """
        Load a trained model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Self for method chaining
        """
        if path is None:
            path = str(DEFAULT_MODEL_DIR / "kaggle_isolation_forest.joblib")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_state = joblib.load(path)
        self.model = model_state["model"]
        self.scaler = model_state["scaler"]
        self._is_fitted = model_state["is_fitted"]
        
        return self


class EnsembleDetector:
    """
    Ensemble detector combining transaction-based and PCA-based models.
    
    Uses weighted averaging of scores from both detectors for
    optimal precision-recall balance.
    """
    
    def __init__(
        self,
        transaction_weight: float = 0.3,
        kaggle_weight: float = 0.7
    ):
        """
        Initialize ensemble with weight configuration.
        
        Args:
            transaction_weight: Weight for transaction-based detector
            kaggle_weight: Weight for Kaggle PCA detector
        """
        # Import here to avoid circular dependency
        from src.detector import Detector
        
        self.transaction_detector = Detector()
        self.kaggle_detector = KaggleDetector()
        self.transaction_weight = transaction_weight
        self.kaggle_weight = kaggle_weight
        self._is_fitted = False
    
    def load_models(
        self,
        transaction_model_path: Optional[str] = None,
        kaggle_model_path: Optional[str] = None
    ) -> "EnsembleDetector":
        """
        Load both pre-trained models.
        
        Args:
            transaction_model_path: Path to transaction detector model
            kaggle_model_path: Path to Kaggle detector model
            
        Returns:
            Self for method chaining
        """
        self.transaction_detector.load_model(transaction_model_path)
        self.kaggle_detector.load_model(kaggle_model_path)
        self._is_fitted = True
        return self
    
    def score(
        self,
        transaction: dict,
        pca_features: Optional[dict] = None
    ) -> float:
        """
        Score a transaction using both models.
        
        Args:
            transaction: Standard transaction dict (amount, location, etc.)
            pca_features: Optional PCA features (V1-V28, Amount)
            
        Returns:
            Weighted average anomaly score
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must have both models loaded.")
        
        # Get transaction-based score
        txn_score = self.transaction_detector.score(transaction)
        
        # If PCA features provided, use Kaggle detector
        if pca_features is not None and self.kaggle_detector._is_fitted:
            kaggle_score = self.kaggle_detector.score(pca_features)
            
            # Weighted average
            combined_score = (
                self.transaction_weight * txn_score +
                self.kaggle_weight * kaggle_score
            )
        else:
            # Fall back to transaction-only score
            combined_score = txn_score
        
        return float(np.clip(combined_score, 0, 1))


if __name__ == "__main__":
    # Train and save model
    detector = KaggleDetector()
    detector.fit_from_csv()
    detector.save_model()
    
    # Test scoring
    print("\nTesting model...")
    test_features = {f"V{i}": 0.0 for i in range(1, 29)}
    test_features["Amount"] = 100.0
    
    score = detector.score(test_features)
    print(f"Test score for normal transaction: {score:.4f}")
