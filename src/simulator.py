"""
Transaction Simulator - Generates synthetic financial transaction data.

This module creates realistic transaction streams for testing the anomaly
detection system, with configurable anomaly injection rates.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Optional
import json


# Configuration constants
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
MERCHANT_TYPES = ["grocery", "electronics", "restaurant", "fuel", "travel", "entertainment", "healthcare", "utilities"]
NORMAL_AMOUNT_RANGE = (10, 5000)
ANOMALY_AMOUNT_RANGE = (10000, 50000)
ANOMALY_RATE = 0.01  # 1% of transactions will be anomalous


class Transaction:
    """Represents a single financial transaction."""
    
    def __init__(
        self,
        transaction_id: str,
        user_id: str,
        amount: float,
        location: str,
        timestamp: str,
        merchant_type: str,
        is_anomaly: bool = False
    ):
        self.transaction_id = transaction_id
        self.user_id = user_id
        self.amount = amount
        self.location = location
        self.timestamp = timestamp
        self.merchant_type = merchant_type
        self.is_anomaly = is_anomaly
    
    def to_dict(self) -> dict:
        """Convert transaction to dictionary format."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "amount": self.amount,
            "location": self.location,
            "timestamp": self.timestamp,
            "merchant_type": self.merchant_type
        }
    
    def to_json(self) -> str:
        """Convert transaction to JSON string."""
        return json.dumps(self.to_dict())


class TransactionSimulator:
    """
    Generates a stream of synthetic financial transactions.
    
    Attributes:
        num_users: Number of unique users in the simulation
        anomaly_rate: Proportion of transactions that should be anomalous
    """
    
    def __init__(self, num_users: int = 100, anomaly_rate: float = ANOMALY_RATE):
        self.num_users = num_users
        self.anomaly_rate = anomaly_rate
        self.user_ids = [f"user_{str(i).zfill(4)}" for i in range(num_users)]
        self.user_locations = {uid: random.choice(CITIES) for uid in self.user_ids}
    
    def generate_transaction(
        self,
        user_id: Optional[str] = None,
        force_anomaly: bool = False,
        base_time: Optional[datetime] = None
    ) -> Transaction:
        """
        Generate a single transaction.
        
        Args:
            user_id: Specific user ID, or random if None
            force_anomaly: If True, generate an anomalous transaction
            base_time: Base timestamp for the transaction
            
        Returns:
            Transaction object
        """
        if user_id is None:
            user_id = random.choice(self.user_ids)
        
        is_anomaly = force_anomaly or (random.random() < self.anomaly_rate)
        
        # Generate transaction ID
        txn_id = f"txn_{uuid.uuid4().hex[:12]}"
        
        # Generate timestamp
        if base_time is None:
            base_time = datetime.now()
        time_offset = timedelta(seconds=random.randint(0, 3600))
        timestamp = (base_time - time_offset).isoformat() + "Z"
        
        # Generate amount (anomalous transactions have higher amounts)
        if is_anomaly and random.random() < 0.5:  # 50% of anomalies are high-amount
            amount = round(random.uniform(*ANOMALY_AMOUNT_RANGE), 2)
        else:
            amount = round(random.uniform(*NORMAL_AMOUNT_RANGE), 2)
        
        # Generate location (anomalous transactions may come from unusual locations)
        if is_anomaly and random.random() < 0.5:  # 50% of anomalies have unusual location
            # Pick a city different from user's home city
            home_city = self.user_locations.get(user_id, CITIES[0])
            other_cities = [c for c in CITIES if c != home_city]
            location = random.choice(other_cities)
        else:
            location = self.user_locations.get(user_id, random.choice(CITIES))
        
        # Generate merchant type
        merchant_type = random.choice(MERCHANT_TYPES)
        
        return Transaction(
            transaction_id=txn_id,
            user_id=user_id,
            amount=amount,
            location=location,
            timestamp=timestamp,
            merchant_type=merchant_type,
            is_anomaly=is_anomaly
        )
    
    def generate_batch(self, count: int) -> list[Transaction]:
        """Generate a batch of transactions."""
        return [self.generate_transaction() for _ in range(count)]
    
    def generate_velocity_attack(self, user_id: Optional[str] = None, num_transactions: int = 5) -> list[Transaction]:
        """
        Generate a velocity attack scenario: multiple transactions from different cities in quick succession.
        
        Args:
            user_id: User ID for the attack
            num_transactions: Number of rapid transactions
            
        Returns:
            List of transactions simulating a velocity attack
        """
        if user_id is None:
            user_id = random.choice(self.user_ids)
        
        base_time = datetime.now()
        transactions = []
        
        # Use different cities for each transaction
        cities_to_use = random.sample(CITIES, min(num_transactions, len(CITIES)))
        
        for i, city in enumerate(cities_to_use):
            txn = Transaction(
                transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
                user_id=user_id,
                amount=round(random.uniform(*NORMAL_AMOUNT_RANGE), 2),
                location=city,
                timestamp=(base_time + timedelta(seconds=i * 10)).isoformat() + "Z",  # 10 seconds apart
                merchant_type=random.choice(MERCHANT_TYPES),
                is_anomaly=True
            )
            transactions.append(txn)
        
        return transactions


def generate_training_data(num_transactions: int = 10000) -> list[dict]:
    """
    Generate training data for the anomaly detection model.
    
    Args:
        num_transactions: Number of transactions to generate
        
    Returns:
        List of transaction dictionaries suitable for model training
    """
    simulator = TransactionSimulator()
    transactions = simulator.generate_batch(num_transactions)
    return [txn.to_dict() for txn in transactions]


if __name__ == "__main__":
    # Demo: Generate sample transactions
    simulator = TransactionSimulator()
    
    print("=== Normal Transactions ===")
    for _ in range(3):
        txn = simulator.generate_transaction()
        print(txn.to_json())
    
    print("\n=== Anomalous Transaction ===")
    anomaly = simulator.generate_transaction(force_anomaly=True)
    print(anomaly.to_json())
    
    print("\n=== Velocity Attack ===")
    attack = simulator.generate_velocity_attack()
    for txn in attack:
        print(txn.to_json())
