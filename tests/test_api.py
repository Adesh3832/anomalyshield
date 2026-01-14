"""
Tests for FastAPI Application.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_health(self, client):
        """Test root endpoint returns health status."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestDetectEndpoint:
    """Tests for the /detect endpoint."""
    
    @pytest.fixture
    def normal_transaction(self):
        """Normal transaction payload."""
        return {
            "transaction_id": "txn_test001",
            "user_id": "user_001",
            "amount": 500,
            "location": "Mumbai",
            "timestamp": "2026-01-14T14:00:00Z",
            "merchant_type": "grocery"
        }
    
    @pytest.fixture
    def high_risk_transaction(self):
        """High risk transaction payload."""
        return {
            "transaction_id": "txn_test002",
            "user_id": "user_001",
            "amount": 25000,
            "location": "Mumbai",
            "timestamp": "2026-01-14T03:00:00Z",  # Unusual hour
            "merchant_type": "electronics"
        }
    
    def test_detect_returns_200(self, client, normal_transaction):
        """Test detect endpoint returns 200."""
        response = client.post("/detect", json=normal_transaction)
        
        assert response.status_code == 200
    
    def test_detect_response_structure(self, client, normal_transaction):
        """Test detect endpoint response has required fields."""
        response = client.post("/detect", json=normal_transaction)
        data = response.json()
        
        assert "transaction_id" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "reason_codes" in data
        assert "recommendation" in data
    
    def test_detect_echoes_transaction_id(self, client, normal_transaction):
        """Test that transaction_id is echoed in response."""
        response = client.post("/detect", json=normal_transaction)
        data = response.json()
        
        assert data["transaction_id"] == normal_transaction["transaction_id"]
    
    def test_detect_risk_score_range(self, client, normal_transaction):
        """Test that risk_score is between 0 and 1."""
        response = client.post("/detect", json=normal_transaction)
        data = response.json()
        
        assert 0 <= data["risk_score"] <= 1
    
    def test_detect_risk_level_valid(self, client, normal_transaction):
        """Test that risk_level is a valid category."""
        response = client.post("/detect", json=normal_transaction)
        data = response.json()
        
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_detect_recommendation_valid(self, client, normal_transaction):
        """Test that recommendation is a valid action."""
        response = client.post("/detect", json=normal_transaction)
        data = response.json()
        
        assert data["recommendation"] in ["APPROVE", "REVIEW", "BLOCK"]
    
    def test_detect_reason_codes_is_list(self, client, normal_transaction):
        """Test that reason_codes is a list."""
        response = client.post("/detect", json=normal_transaction)
        data = response.json()
        
        assert isinstance(data["reason_codes"], list)
    
    def test_detect_high_risk_triggers_codes(self, client, high_risk_transaction):
        """Test that high risk transaction triggers reason codes."""
        response = client.post("/detect", json=high_risk_transaction)
        data = response.json()
        
        # Should have at least HIGH_AMOUNT and UNUSUAL_HOUR
        assert "HIGH_AMOUNT" in data["reason_codes"]
        assert "UNUSUAL_HOUR" in data["reason_codes"]
    
    def test_detect_missing_field_returns_422(self, client):
        """Test that missing required field returns 422."""
        incomplete = {
            "transaction_id": "txn_001",
            # Missing other required fields
        }
        
        response = client.post("/detect", json=incomplete)
        
        assert response.status_code == 422
    
    def test_detect_invalid_amount_returns_422(self, client, normal_transaction):
        """Test that negative amount returns 422."""
        normal_transaction["amount"] = -100
        
        response = client.post("/detect", json=normal_transaction)
        
        assert response.status_code == 422


class TestBatchDetectEndpoint:
    """Tests for the /batch-detect endpoint."""
    
    def test_batch_detect_multiple(self, client):
        """Test batch detection with multiple transactions."""
        transactions = [
            {
                "transaction_id": f"txn_{i}",
                "user_id": "user_001",
                "amount": 100 * i,
                "location": "Mumbai",
                "timestamp": "2026-01-14T14:00:00Z",
                "merchant_type": "grocery"
            }
            for i in range(5)
        ]
        
        response = client.post("/batch-detect", json=transactions)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
    
    def test_batch_detect_empty_list(self, client):
        """Test batch detection with empty list."""
        response = client.post("/batch-detect", json=[])
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_batch_detect_over_limit(self, client):
        """Test batch detection over 100 limit."""
        transactions = [
            {
                "transaction_id": f"txn_{i}",
                "user_id": "user_001",
                "amount": 100,
                "location": "Mumbai",
                "timestamp": "2026-01-14T14:00:00Z",
                "merchant_type": "grocery"
            }
            for i in range(101)
        ]
        
        response = client.post("/batch-detect", json=transactions)
        
        assert response.status_code == 400
        assert "100" in response.json()["detail"]
