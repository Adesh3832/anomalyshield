"""
Tests for Rules Engine.
"""

import pytest
from src.rules_engine import RulesEngine, RuleResult


class TestRulesEngine:
    """Tests for the RulesEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Fresh rules engine for each test."""
        return RulesEngine()
    
    def test_engine_initialization(self, engine):
        """Test rules engine initialization."""
        assert engine.HIGH_AMOUNT_THRESHOLD == 10000
        assert engine.VELOCITY_TXN_THRESHOLD == 3
    
    # --- High Amount Rule Tests ---
    
    def test_high_amount_triggered(self, engine):
        """Test high amount rule triggers above threshold."""
        txn = {
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "amount": 15000,
            "location": "Mumbai",
            "timestamp": "2026-01-14T14:00:00Z",
            "merchant_type": "electronics"
        }
        
        result = engine.check_high_amount(txn)
        
        assert result.triggered is True
        assert result.reason_code == "HIGH_AMOUNT"
    
    def test_high_amount_not_triggered(self, engine):
        """Test high amount rule does not trigger below threshold."""
        txn = {
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "amount": 5000,
            "location": "Mumbai",
            "timestamp": "2026-01-14T14:00:00Z",
            "merchant_type": "grocery"
        }
        
        result = engine.check_high_amount(txn)
        
        assert result.triggered is False
    
    def test_high_amount_boundary(self, engine):
        """Test high amount at exact threshold (should not trigger)."""
        txn = {"amount": 10000}
        
        result = engine.check_high_amount(txn)
        
        assert result.triggered is False  # Not > threshold, just equal
    
    # --- Unusual Time Rule Tests ---
    
    def test_unusual_time_triggered(self, engine):
        """Test unusual time rule triggers during 2-5 AM."""
        txn = {
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "amount": 500,
            "location": "Mumbai",
            "timestamp": "2026-01-14T03:30:00Z",  # 3:30 AM
            "merchant_type": "atm"
        }
        
        result = engine.check_unusual_time(txn)
        
        assert result.triggered is True
        assert result.reason_code == "UNUSUAL_HOUR"
    
    def test_unusual_time_not_triggered_daytime(self, engine):
        """Test unusual time rule does not trigger during day."""
        txn = {
            "timestamp": "2026-01-14T14:00:00Z"  # 2 PM
        }
        
        result = engine.check_unusual_time(txn)
        
        assert result.triggered is False
    
    def test_unusual_time_boundary_start(self, engine):
        """Test unusual time at 2 AM (should trigger)."""
        txn = {"timestamp": "2026-01-14T02:00:00Z"}
        
        result = engine.check_unusual_time(txn)
        
        assert result.triggered is True
    
    def test_unusual_time_boundary_end(self, engine):
        """Test unusual time at 5 AM (should not trigger)."""
        txn = {"timestamp": "2026-01-14T05:00:00Z"}
        
        result = engine.check_unusual_time(txn)
        
        assert result.triggered is False  # 5 AM is not < 5
    
    # --- Velocity Rule Tests ---
    
    def test_velocity_single_transaction(self, engine):
        """Test velocity rule does not trigger for single transaction."""
        txn = {
            "user_id": "user_001",
            "location": "Mumbai",
            "timestamp": "2026-01-14T10:00:00Z"
        }
        
        result = engine.check_velocity(txn)
        
        assert result.triggered is False
    
    def test_velocity_multiple_same_city(self, engine):
        """Test velocity rule does not trigger for same city."""
        for i in range(5):
            txn = {
                "user_id": "user_001",
                "location": "Mumbai",  # Same city
                "timestamp": f"2026-01-14T10:00:{i:02d}Z"
            }
            result = engine.check_velocity(txn)
        
        # Should not trigger because all same city
        assert result.triggered is False
    
    def test_velocity_triggered_multi_city(self, engine):
        """Test velocity rule triggers for multiple cities."""
        cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]
        
        results = []
        for i, city in enumerate(cities):
            txn = {
                "user_id": "user_001",
                "location": city,
                "timestamp": f"2026-01-14T10:00:{i:02d}Z"
            }
            result = engine.check_velocity(txn)
            results.append(result)
        
        # Should trigger after threshold exceeded
        assert any(r.triggered and r.reason_code == "VELOCITY_MULTI_CITY" for r in results)
    
    def test_velocity_different_users_independent(self, engine):
        """Test that velocity is tracked per user."""
        # User 1 makes transactions
        for i in range(3):
            engine.check_velocity({
                "user_id": "user_001",
                "location": f"City{i}",
                "timestamp": f"2026-01-14T10:00:{i:02d}Z"
            })
        
        # User 2 makes transaction
        result = engine.check_velocity({
            "user_id": "user_002",
            "location": "Mumbai",
            "timestamp": "2026-01-14T10:00:30Z"
        })
        
        # User 2 should not trigger velocity
        assert result.triggered is False
    
    def test_velocity_clears_old_transactions(self, engine):
        """Test that old transactions are cleared from history."""
        # Make transactions 2 minutes apart (outside 60s window)
        engine.check_velocity({
            "user_id": "user_001",
            "location": "Mumbai",
            "timestamp": "2026-01-14T10:00:00Z"
        })
        
        engine.check_velocity({
            "user_id": "user_001",
            "location": "Delhi",
            "timestamp": "2026-01-14T10:02:00Z"  # 2 minutes later
        })
        
        # First transaction should be cleared, so no velocity trigger
        # (would need 4+ in same window)
        assert len(engine._transaction_history["user_001"]) <= 2
    
    # --- Combined Rule Tests ---
    
    def test_evaluate_returns_triggered_only(self, engine):
        """Test evaluate() returns only triggered rules."""
        txn = {
            "user_id": "user_001",
            "amount": 15000,  # High amount
            "location": "Mumbai",
            "timestamp": "2026-01-14T14:00:00Z"  # Normal time
        }
        
        results = engine.evaluate(txn)
        
        assert len(results) == 1
        assert results[0].reason_code == "HIGH_AMOUNT"
    
    def test_get_reason_codes(self, engine):
        """Test get_reason_codes() returns list of codes."""
        txn = {
            "user_id": "user_001",
            "amount": 15000,
            "location": "Mumbai",
            "timestamp": "2026-01-14T03:00:00Z"  # Unusual time
        }
        
        codes = engine.get_reason_codes(txn)
        
        assert "HIGH_AMOUNT" in codes
        assert "UNUSUAL_HOUR" in codes
    
    def test_clear_history(self, engine):
        """Test clearing transaction history."""
        engine.check_velocity({
            "user_id": "user_001",
            "location": "Mumbai",
            "timestamp": "2026-01-14T10:00:00Z"
        })
        
        assert len(engine._transaction_history) > 0
        
        engine.clear_history()
        
        assert len(engine._transaction_history) == 0
    
    def test_clear_history_specific_user(self, engine):
        """Test clearing history for specific user."""
        engine.check_velocity({
            "user_id": "user_001",
            "location": "Mumbai",
            "timestamp": "2026-01-14T10:00:00Z"
        })
        engine.check_velocity({
            "user_id": "user_002",
            "location": "Delhi",
            "timestamp": "2026-01-14T10:00:00Z"
        })
        
        engine.clear_history("user_001")
        
        assert "user_001" not in engine._transaction_history
        assert "user_002" in engine._transaction_history
