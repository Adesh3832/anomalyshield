"""
Kaggle Credit Card Fraud Dataset Streamer

Provides a generator-based streaming interface to simulate real-time
transaction processing from the Kaggle European Credit Card Fraud dataset.
"""

import csv
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional


# Cities and merchant types for synthetic field generation
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
MERCHANT_TYPES = ["grocery", "electronics", "restaurant", "fuel", "travel", "entertainment", "healthcare", "utilities"]


class KaggleTransaction:
    """Represents a transaction from the Kaggle dataset with ground truth."""
    
    def __init__(
        self,
        row_id: int,
        amount: float,
        time_elapsed: float,
        is_fraud: bool,
        pca_features: dict[str, float]
    ):
        self.row_id = row_id
        self.amount = amount
        self.time_elapsed = time_elapsed
        self.is_fraud = is_fraud
        self.pca_features = pca_features
    
    def to_api_request(self, base_time: Optional[datetime] = None) -> dict:
        """
        Convert to TransactionRequest format for the /detect API.
        
        Uses synthetic fields for location and merchant_type since the
        Kaggle dataset doesn't contain these. Amount is preserved as-is.
        
        Args:
            base_time: Starting timestamp for the simulation
            
        Returns:
            Dictionary compatible with TransactionRequest schema
        """
        if base_time is None:
            base_time = datetime(2026, 1, 14, 0, 0, 0)
        
        # Generate synthetic timestamp using the Time column (seconds elapsed)
        transaction_time = base_time + timedelta(seconds=self.time_elapsed)
        
        # Generate synthetic location and merchant type based on amount patterns
        # Higher amounts more likely in electronics/travel, lower in grocery/fuel
        if self.amount > 1000:
            merchant_weights = [0.05, 0.35, 0.1, 0.05, 0.3, 0.05, 0.05, 0.05]
        elif self.amount > 200:
            merchant_weights = [0.15, 0.2, 0.2, 0.1, 0.1, 0.15, 0.05, 0.05]
        else:
            merchant_weights = [0.3, 0.05, 0.3, 0.2, 0.02, 0.08, 0.03, 0.02]
        
        merchant_type = random.choices(MERCHANT_TYPES, weights=merchant_weights, k=1)[0]
        location = random.choice(CITIES)
        
        return {
            "transaction_id": f"kaggle_{self.row_id:06d}",
            "user_id": f"user_{(self.row_id % 1000):04d}",  # Simulate 1000 users
            "amount": round(self.amount, 2),
            "location": location,
            "timestamp": transaction_time.isoformat() + "Z",
            "merchant_type": merchant_type
        }
    
    @property
    def ground_truth_label(self) -> str:
        """Return FRAUD or NORMAL based on the Class label."""
        return "FRAUD" if self.is_fraud else "NORMAL"


def stream_kaggle_transactions(
    csv_path: str,
    delay_seconds: float = 0.0,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42
) -> Generator[KaggleTransaction, None, None]:
    """
    Stream transactions from the Kaggle Credit Card CSV file.
    
    Yields transactions one-by-one to simulate a real-time data stream.
    
    Args:
        csv_path: Path to creditcard.csv
        delay_seconds: Delay between yielding transactions (for simulation)
        limit: Maximum number of transactions to yield (None = all)
        shuffle: Whether to shuffle the data (useful for balanced sampling)
        seed: Random seed for reproducibility
        
    Yields:
        KaggleTransaction objects with ground truth labels
    """
    random.seed(seed)
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Kaggle dataset not found: {csv_path}")
    
    # Read all rows if shuffling, otherwise stream directly
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        if shuffle:
            rows = list(reader)
            random.shuffle(rows)
        else:
            rows = reader
        
        count = 0
        for row in rows:
            if limit is not None and count >= limit:
                break
            
            # Extract PCA features (V1-V28)
            pca_features = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
            
            transaction = KaggleTransaction(
                row_id=count,
                amount=float(row["Amount"]),
                time_elapsed=float(row["Time"]),
                is_fraud=int(row["Class"]) == 1,
                pca_features=pca_features
            )
            
            yield transaction
            count += 1
            
            if delay_seconds > 0:
                time.sleep(delay_seconds)


def get_dataset_stats(csv_path: str) -> dict:
    """
    Get statistics about the Kaggle dataset.
    
    Args:
        csv_path: Path to creditcard.csv
        
    Returns:
        Dictionary with dataset statistics
    """
    total = 0
    fraud_count = 0
    amounts = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if int(row["Class"]) == 1:
                fraud_count += 1
            amounts.append(float(row["Amount"]))
    
    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "normal_count": total - fraud_count,
        "fraud_rate": fraud_count / total if total > 0 else 0,
        "min_amount": min(amounts) if amounts else 0,
        "max_amount": max(amounts) if amounts else 0,
        "avg_amount": sum(amounts) / len(amounts) if amounts else 0
    }


if __name__ == "__main__":
    import sys
    
    # Default path
    csv_path = "data/creditcard.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    print(f"Loading dataset from: {csv_path}")
    
    try:
        stats = get_dataset_stats(csv_path)
        print("\n=== Dataset Statistics ===")
        print(f"Total transactions: {stats['total_transactions']:,}")
        print(f"Fraud cases: {stats['fraud_count']:,} ({stats['fraud_rate']:.4%})")
        print(f"Normal cases: {stats['normal_count']:,}")
        print(f"Amount range: ${stats['min_amount']:.2f} - ${stats['max_amount']:.2f}")
        print(f"Average amount: ${stats['avg_amount']:.2f}")
        
        print("\n=== Sample Transactions ===")
        for i, txn in enumerate(stream_kaggle_transactions(csv_path, limit=5)):
            api_request = txn.to_api_request()
            print(f"\n[{i+1}] {txn.ground_truth_label}")
            print(f"    Amount: ${txn.amount:.2f}")
            print(f"    API Request: {api_request}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the dataset first:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/")
        print("  unzip data/creditcardfraud.zip -d data/")
