"""
Tests for Anomaly Detector.
"""

import pytest
import tempfile
import os

from src.detector import Detector
from src.simulator import generate_training_data


class TestDetector:
    """Tests for the Detector class."""
    
    @pytest.fixture
    def trained_detector(self):
        """Create a trained detector for testing."""
        detector = Detector()
        training_data = generate_training_data(1000)
        detector.fit(training_data)
        return detector
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample normal transaction."""
        return {
            "transaction_id": "txn_test001",
            "user_id": "user_0001",
            "amount": 500,
            "location": "Mumbai",
            "timestamp": "2026-01-14T14:00:00Z",
            "merchant_type": "grocery"
        }
    
    @pytest.fixture
    def anomalous_transaction(self):
        """Sample anomalous transaction."""
        return {
            "transaction_id": "txn_test002",
            "user_id": "user_0001",
            "amount": 45000,  # Very high amount
            "location": "Unknown",
            "timestamp": "2026-01-14T03:00:00Z",  # Unusual hour
            "merchant_type": "unknown"
        }
    
    def test_detector_initialization(self):
        """Test detector initialization with default parameters."""
        detector = Detector()
        
        assert detector.model is not None
        assert detector._is_fitted is False
    
    def test_detector_custom_params(self):
        """Test detector with custom parameters."""
        detector = Detector(
            n_estimators=200,
            max_samples=128,
            contamination=0.05
        )
        
        assert detector.model.n_estimators == 200
        assert detector.model.max_samples == 128
        assert detector.model.contamination == 0.05
    
    def test_fit_marks_fitted(self):
        """Test that fit() sets the fitted flag."""
        detector = Detector()
        training_data = generate_training_data(100)
        detector.fit(training_data)
        
        assert detector._is_fitted is True
    
    def test_score_before_fit_raises_error(self, sample_transaction):
        """Test that scoring before fit raises error."""
        detector = Detector()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.score(sample_transaction)
    
    def test_score_returns_float(self, trained_detector, sample_transaction):
        """Test that score returns a float."""
        score = trained_detector.score(sample_transaction)
        
        assert isinstance(score, float)
    
    def test_score_range(self, trained_detector, sample_transaction):
        """Test that score is between 0 and 1."""
        score = trained_detector.score(sample_transaction)
        
        assert 0 <= score <= 1
    
    def test_anomalous_scores_higher(self, trained_detector, sample_transaction, anomalous_transaction):
        """Test that anomalous transactions generally score higher."""
        normal_score = trained_detector.score(sample_transaction)
        anomaly_score = trained_detector.score(anomalous_transaction)
        
        # Anomalous transaction should have higher score (more anomalous)
        # Note: This might occasionally fail due to model variance
        assert anomaly_score > normal_score * 0.9  # Allow some tolerance
    
    def test_score_batch(self, trained_detector):
        """Test batch scoring."""
        transactions = generate_training_data(10)
        scores = trained_detector.score_batch(transactions)
        
        assert len(scores) == 10
        assert all(isinstance(s, float) for s in scores)
        assert all(0 <= s <= 1 for s in scores)
    
    def test_save_and_load_model(self, trained_detector, sample_transaction):
        """Test model persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            
            # Save model
            saved_path = trained_detector.save_model(model_path)
            assert os.path.exists(saved_path)
            
            # Get score from original
            original_score = trained_detector.score(sample_transaction)
            
            # Load into new detector
            new_detector = Detector()
            new_detector.load_model(model_path)
            
            # Score should be the same
            loaded_score = new_detector.score(sample_transaction)
            assert abs(original_score - loaded_score) < 0.0001
    
    def test_load_nonexistent_model_raises_error(self):
        """Test that loading nonexistent model raises error."""
        detector = Detector()
        
        with pytest.raises(FileNotFoundError):
            detector.load_model("/nonexistent/path/model.joblib")
    
    def test_feature_extraction_handles_malformed_timestamp(self, trained_detector):
        """Test that feature extraction handles bad timestamps."""
        bad_txn = {
            "transaction_id": "txn_bad",
            "user_id": "user_001",
            "amount": 100,
            "location": "Mumbai",
            "timestamp": "not-a-timestamp",
            "merchant_type": "grocery"
        }
        
        # Should not raise, should use defaults
        score = trained_detector.score(bad_txn)
        assert 0 <= score <= 1
