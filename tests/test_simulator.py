"""
Tests for Transaction Simulator.
"""

import json
import pytest
from datetime import datetime

from src.simulator import Transaction, TransactionSimulator, generate_training_data


class TestTransaction:
    """Tests for the Transaction class."""
    
    def test_transaction_creation(self):
        """Test basic transaction creation."""
        txn = Transaction(
            transaction_id="txn_001",
            user_id="user_001",
            amount=100.00,
            location="Mumbai",
            timestamp="2026-01-14T15:00:00Z",
            merchant_type="grocery"
        )
        
        assert txn.transaction_id == "txn_001"
        assert txn.user_id == "user_001"
        assert txn.amount == 100.00
        assert txn.location == "Mumbai"
        assert txn.merchant_type == "grocery"
    
    def test_transaction_to_dict(self):
        """Test transaction dictionary conversion."""
        txn = Transaction(
            transaction_id="txn_001",
            user_id="user_001",
            amount=100.00,
            location="Mumbai",
            timestamp="2026-01-14T15:00:00Z",
            merchant_type="grocery"
        )
        
        result = txn.to_dict()
        
        assert isinstance(result, dict)
        assert result["transaction_id"] == "txn_001"
        assert result["amount"] == 100.00
        assert "is_anomaly" not in result  # Should not expose internal flag
    
    def test_transaction_to_json(self):
        """Test transaction JSON serialization."""
        txn = Transaction(
            transaction_id="txn_001",
            user_id="user_001",
            amount=100.00,
            location="Mumbai",
            timestamp="2026-01-14T15:00:00Z",
            merchant_type="grocery"
        )
        
        json_str = txn.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["transaction_id"] == "txn_001"
        assert parsed["amount"] == 100.00


class TestTransactionSimulator:
    """Tests for the TransactionSimulator class."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization with default parameters."""
        sim = TransactionSimulator()
        
        assert sim.num_users == 100
        assert sim.anomaly_rate == 0.01
        assert len(sim.user_ids) == 100
    
    def test_simulator_custom_params(self):
        """Test simulator with custom parameters."""
        sim = TransactionSimulator(num_users=50, anomaly_rate=0.05)
        
        assert sim.num_users == 50
        assert sim.anomaly_rate == 0.05
    
    def test_generate_transaction_fields(self):
        """Test that generated transactions have all required fields."""
        sim = TransactionSimulator()
        txn = sim.generate_transaction()
        
        assert txn.transaction_id.startswith("txn_")
        assert txn.user_id.startswith("user_")
        assert isinstance(txn.amount, float)
        assert txn.amount > 0
        assert isinstance(txn.location, str)
        assert isinstance(txn.timestamp, str)
        assert isinstance(txn.merchant_type, str)
    
    def test_generate_forced_anomaly(self):
        """Test forcing an anomalous transaction."""
        sim = TransactionSimulator()
        txn = sim.generate_transaction(force_anomaly=True)
        
        assert txn.is_anomaly is True
    
    def test_generate_batch(self):
        """Test batch transaction generation."""
        sim = TransactionSimulator()
        batch = sim.generate_batch(100)
        
        assert len(batch) == 100
        assert all(isinstance(t, Transaction) for t in batch)
    
    def test_velocity_attack_generation(self):
        """Test velocity attack scenario generation."""
        sim = TransactionSimulator()
        attack = sim.generate_velocity_attack(num_transactions=5)
        
        assert len(attack) == 5
        # All should be from same user
        user_ids = set(t.user_id for t in attack)
        assert len(user_ids) == 1
        # Should have different cities
        cities = set(t.location for t in attack)
        assert len(cities) > 1
        # All marked as anomalies
        assert all(t.is_anomaly for t in attack)


class TestGenerateTrainingData:
    """Tests for the training data generation function."""
    
    def test_generate_training_data_count(self):
        """Test that correct number of samples are generated."""
        data = generate_training_data(500)
        
        assert len(data) == 500
    
    def test_generate_training_data_format(self):
        """Test that training data is in correct format."""
        data = generate_training_data(10)
        
        assert all(isinstance(d, dict) for d in data)
        required_fields = ["transaction_id", "user_id", "amount", "location", "timestamp", "merchant_type"]
        for d in data:
            for field in required_fields:
                assert field in d
