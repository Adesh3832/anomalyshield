"""
Rules Engine - Hard-coded compliance rules for velocity and pattern checks.

This module implements banking compliance rules that operate independently
of the ML model, providing deterministic reason codes for flagged transactions.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuleResult:
    """Result of a rule check."""
    triggered: bool
    reason_code: str
    description: str


class RulesEngine:
    """
    Compliance rules engine for transaction monitoring.
    
    Implements the following rules:
    1. Velocity Check: >3 transactions in 1 minute from different cities
    2. High Amount: Single transaction > $10,000
    3. Unusual Time: Transaction between 2-5 AM local time
    
    The engine maintains a transaction history for velocity checks.
    """
    
    # Rule thresholds
    VELOCITY_WINDOW_SECONDS = 60  # 1 minute
    VELOCITY_TXN_THRESHOLD = 3  # More than 3 transactions
    VELOCITY_CITY_THRESHOLD = 2  # From at least 2 different cities
    HIGH_AMOUNT_THRESHOLD = 10000  # $10,000
    UNUSUAL_HOUR_START = 2  # 2 AM
    UNUSUAL_HOUR_END = 5  # 5 AM
    
    def __init__(self):
        """Initialize the rules engine with empty transaction history."""
        # Store recent transactions per user: {user_id: [(timestamp, location), ...]}
        self._transaction_history: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse ISO format timestamp string."""
        try:
            timestamp_str = timestamp_str.replace("Z", "+00:00")
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, AttributeError):
            return None
    
    def _cleanup_old_transactions(self, user_id: str, current_time: datetime) -> None:
        """Remove transactions older than the velocity window."""
        cutoff = current_time - timedelta(seconds=self.VELOCITY_WINDOW_SECONDS)
        self._transaction_history[user_id] = [
            (ts, loc) for ts, loc in self._transaction_history[user_id]
            if ts > cutoff
        ]
    
    def check_velocity(self, transaction: dict) -> RuleResult:
        """
        Check for velocity attack: >3 transactions in 1 minute from different cities.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            RuleResult indicating if rule was triggered
        """
        user_id = transaction.get("user_id", "")
        location = transaction.get("location", "")
        timestamp = self._parse_timestamp(transaction.get("timestamp", ""))
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Clean up old transactions
        self._cleanup_old_transactions(user_id, timestamp)
        
        # Add current transaction
        self._transaction_history[user_id].append((timestamp, location))
        
        # Get recent transactions
        recent = self._transaction_history[user_id]
        
        # Check if threshold exceeded
        if len(recent) > self.VELOCITY_TXN_THRESHOLD:
            # Check for multiple cities
            unique_cities = [loc for _, loc in recent]
            unique_city_set = set(unique_cities)
            if len(unique_city_set) >= self.VELOCITY_CITY_THRESHOLD:
                # Build detailed description with city sequence
                city_sequence = " → ".join(unique_cities)
                return RuleResult(
                    triggered=True,
                    reason_code="VELOCITY_MULTI_CITY",
                    description=f"Transaction #{len(recent)} in {self.VELOCITY_WINDOW_SECONDS}s window. Cities: {city_sequence}. This is transaction from {location} - flagged because user has {len(recent)} transactions across {len(unique_city_set)} different cities within 60 seconds."
                )
        
        return RuleResult(
            triggered=False,
            reason_code="VELOCITY_MULTI_CITY",
            description="Velocity check passed"
        )
    
    def check_high_amount(self, transaction: dict) -> RuleResult:
        """
        Check for high-value transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            RuleResult indicating if rule was triggered
        """
        try:
            amount = float(transaction.get("amount", 0))
        except (TypeError, ValueError):
            amount = 0
        
        if amount > self.HIGH_AMOUNT_THRESHOLD:
            return RuleResult(
                triggered=True,
                reason_code="HIGH_AMOUNT",
                description=f"Transaction amount ${amount:,.2f} exceeds ${self.HIGH_AMOUNT_THRESHOLD:,} threshold"
            )
        
        return RuleResult(
            triggered=False,
            reason_code="HIGH_AMOUNT",
            description="Amount check passed"
        )
    
    def check_unusual_time(self, transaction: dict) -> RuleResult:
        """
        Check for transaction during unusual hours (2-5 AM).
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            RuleResult indicating if rule was triggered
        """
        timestamp = self._parse_timestamp(transaction.get("timestamp", ""))
        
        if timestamp is None:
            return RuleResult(
                triggered=False,
                reason_code="UNUSUAL_HOUR",
                description="Unable to parse timestamp"
            )
        
        hour = timestamp.hour
        
        if self.UNUSUAL_HOUR_START <= hour < self.UNUSUAL_HOUR_END:
            return RuleResult(
                triggered=True,
                reason_code="UNUSUAL_HOUR",
                description=f"Transaction at {hour}:00 is during unusual hours ({self.UNUSUAL_HOUR_START}-{self.UNUSUAL_HOUR_END} AM)"
            )
        
        return RuleResult(
            triggered=False,
            reason_code="UNUSUAL_HOUR",
            description="Time check passed"
        )
    
    def evaluate(self, transaction: dict) -> list[RuleResult]:
        """
        Evaluate all rules against a transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            List of RuleResults for triggered rules only
        """
        results = []
        
        # Check all rules
        velocity_result = self.check_velocity(transaction)
        if velocity_result.triggered:
            results.append(velocity_result)
        
        amount_result = self.check_high_amount(transaction)
        if amount_result.triggered:
            results.append(amount_result)
        
        time_result = self.check_unusual_time(transaction)
        if time_result.triggered:
            results.append(time_result)
        
        return results
    
    def get_reason_codes(self, transaction: dict) -> list[str]:
        """
        Get reason codes for triggered rules.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            List of reason code strings
        """
        results = self.evaluate(transaction)
        return [r.reason_code for r in results]
    
    def clear_history(self, user_id: Optional[str] = None) -> None:
        """
        Clear transaction history.
        
        Args:
            user_id: Specific user to clear, or None to clear all
        """
        if user_id:
            self._transaction_history.pop(user_id, None)
        else:
            self._transaction_history.clear()


if __name__ == "__main__":
    # Demo: Test the rules engine
    engine = RulesEngine()
    
    # Test high amount
    high_amount_txn = {
        "transaction_id": "txn_001",
        "user_id": "user_001",
        "amount": 15000,
        "location": "Mumbai",
        "timestamp": "2026-01-14T14:00:00Z",
        "merchant_type": "electronics"
    }
    
    print("=== High Amount Test ===")
    codes = engine.get_reason_codes(high_amount_txn)
    print(f"Reason codes: {codes}")
    
    # Test unusual time
    unusual_time_txn = {
        "transaction_id": "txn_002",
        "user_id": "user_002",
        "amount": 500,
        "location": "Delhi",
        "timestamp": "2026-01-14T03:30:00Z",
        "merchant_type": "atm"
    }
    
    print("\n=== Unusual Time Test ===")
    codes = engine.get_reason_codes(unusual_time_txn)
    print(f"Reason codes: {codes}")
    
    # Test velocity attack
    print("\n=== Velocity Attack Test ===")
    engine.clear_history()
    
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]
    for i, city in enumerate(cities):
        txn = {
            "transaction_id": f"txn_velocity_{i}",
            "user_id": "user_003",
            "amount": 500,
            "location": city,
            "timestamp": f"2026-01-14T10:00:{i:02d}Z",
            "merchant_type": "grocery"
        }
        codes = engine.get_reason_codes(txn)
        print(f"Transaction {i+1} from {city}: {codes}")
