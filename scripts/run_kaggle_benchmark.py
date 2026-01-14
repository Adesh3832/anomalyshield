#!/usr/bin/env python3
"""
Kaggle Credit Card Fraud Benchmark

Streams the Kaggle dataset through the AnomalyShield API and evaluates
detection performance against ground truth labels.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.kaggle_streamer import (
    stream_kaggle_transactions,
    get_dataset_stats,
    KaggleTransaction
)


class BenchmarkResults:
    """Tracks and computes benchmark metrics."""
    
    def __init__(self):
        self.true_positives = 0  # Fraud detected as HIGH risk
        self.false_positives = 0  # Normal detected as HIGH risk
        self.true_negatives = 0  # Normal detected as LOW risk
        self.false_negatives = 0  # Fraud detected as LOW risk
        self.predictions = []
        self.latencies = []
        
    def add_result(
        self, 
        is_fraud: bool, 
        risk_score: float, 
        risk_level: str,
        recommendation: str,
        latency_ms: float
    ):
        """Record a prediction result."""
        self.predictions.append({
            "is_fraud": is_fraud,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation
        })
        self.latencies.append(latency_ms)
        
        # Classify based on recommendation (BLOCK/REVIEW = flagged)
        predicted_fraud = recommendation in ["BLOCK", "REVIEW"]
        
        if is_fraud and predicted_fraud:
            self.true_positives += 1
        elif is_fraud and not predicted_fraud:
            self.false_negatives += 1
        elif not is_fraud and predicted_fraud:
            self.false_positives += 1
        else:
            self.true_negatives += 1
    
    @property
    def total(self) -> int:
        return len(self.predictions)
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / Total"""
        correct = self.true_positives + self.true_negatives
        return correct / self.total if self.total > 0 else 0
    
    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0
    
    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "=" * 60)
        print("📊 ANOMALYSHIELD KAGGLE BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\n📈 Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                 FRAUD    NORMAL")
        print(f"Actual FRAUD      {self.true_positives:5d}     {self.false_negatives:5d}")
        print(f"       NORMAL     {self.false_positives:5d}     {self.true_negatives:5d}")
        
        print(f"\n📏 Metrics:")
        print(f"   Precision:  {self.precision:.4f} (of flagged transactions, how many were fraud)")
        print(f"   Recall:     {self.recall:.4f} (of actual fraud, how many were caught)")
        print(f"   F1 Score:   {self.f1_score:.4f} (harmonic mean of precision and recall)")
        print(f"   Accuracy:   {self.accuracy:.4f} (overall correct classifications)")
        
        print(f"\n⏱️  Performance:")
        print(f"   Total transactions: {self.total:,}")
        print(f"   Avg latency: {self.avg_latency_ms:.2f}ms per transaction")
        print(f"   Throughput: {1000 / self.avg_latency_ms:.1f} transactions/second" if self.avg_latency_ms > 0 else "   Throughput: N/A")
        
        print("\n" + "=" * 60)


def run_benchmark(
    api_url: str,
    csv_path: str,
    limit: int = 1000,
    delay: float = 0.0,
    verbose: bool = False,
    balanced: bool = False
):
    """
    Run the streaming benchmark against the AnomalyShield API.
    
    Args:
        api_url: Base URL of the API (e.g., http://localhost:8000)
        csv_path: Path to creditcard.csv
        limit: Number of transactions to process
        delay: Delay between transactions (seconds)
        verbose: Print each transaction result
        balanced: Sample equal fraud/normal transactions
    """
    print(f"🚀 Starting Kaggle Benchmark")
    print(f"   API: {api_url}")
    print(f"   Dataset: {csv_path}")
    print(f"   Limit: {limit:,} transactions")
    print(f"   Delay: {delay}s between requests")
    
    # Check API health
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        health.raise_for_status()
        print(f"   API Status: ✅ Healthy")
    except requests.RequestException as e:
        print(f"   API Status: ❌ Unreachable ({e})")
        print("\n⚠️  Please start the API first:")
        print("   python -m src.api")
        return None
    
    # Get dataset stats
    print(f"\n📁 Dataset Statistics:")
    stats = get_dataset_stats(csv_path)
    print(f"   Total: {stats['total_transactions']:,} transactions")
    print(f"   Fraud: {stats['fraud_count']:,} ({stats['fraud_rate']:.4%})")
    
    # Stream and benchmark
    results = BenchmarkResults()
    detect_url = f"{api_url}/detect"
    
    print(f"\n🔄 Streaming {limit} transactions...")
    start_time = time.time()
    
    fraud_seen = 0
    normal_seen = 0
    
    for txn in stream_kaggle_transactions(csv_path, delay_seconds=0, limit=None, shuffle=balanced):
        # If balanced sampling, try to get equal fraud/normal
        if balanced:
            if txn.is_fraud and fraud_seen >= limit // 2:
                continue
            if not txn.is_fraud and normal_seen >= limit // 2:
                continue
        
        # Convert to API request format
        api_request = txn.to_api_request()
        
        # Call the API
        req_start = time.time()
        try:
            response = requests.post(detect_url, json=api_request, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            print(f"   ⚠️  Request failed for {api_request['transaction_id']}: {e}")
            continue
        
        latency_ms = (time.time() - req_start) * 1000
        
        # Record result
        results.add_result(
            is_fraud=txn.is_fraud,
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            recommendation=result["recommendation"],
            latency_ms=latency_ms
        )
        
        if txn.is_fraud:
            fraud_seen += 1
        else:
            normal_seen += 1
        
        # Verbose output
        if verbose:
            emoji = "🔴" if txn.is_fraud else "🟢"
            flag = "⚠️" if result["recommendation"] != "APPROVE" else "✓"
            print(f"   {emoji} ${txn.amount:>10.2f} | Risk: {result['risk_score']:.3f} | {result['recommendation']:6s} {flag}")
        else:
            # Progress indicator
            if results.total % 100 == 0:
                elapsed = time.time() - start_time
                rate = results.total / elapsed if elapsed > 0 else 0
                print(f"   Processed: {results.total:,} ({rate:.1f} txn/s) - Fraud caught: {results.true_positives}/{fraud_seen}")
        
        # Check if we've hit the limit
        if results.total >= limit:
            break
        
        # Apply delay if specified
        if delay > 0:
            time.sleep(delay)
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed in {total_time:.2f} seconds")
    
    # Print summary
    results.print_summary()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AnomalyShield against Kaggle Credit Card Fraud dataset"
    )
    parser.add_argument(
        "--api-url", 
        default="http://localhost:8000",
        help="Base URL of the AnomalyShield API"
    )
    parser.add_argument(
        "--csv-path",
        default="data/creditcard.csv",
        help="Path to the Kaggle creditcard.csv file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of transactions to process"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between transactions (seconds) to simulate real-time"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each transaction result"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Sample equal fraud/normal transactions"
    )
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"❌ Dataset not found: {args.csv_path}")
        print("\nPlease download the dataset first:")
        print("  1. Install Kaggle CLI: pip install kaggle")
        print("  2. Set API token: export KAGGLE_API_TOKEN=<your-token>")
        print("  3. Download: kaggle datasets download -d mlg-ulb/creditcardfraud -p data/")
        print("  4. Extract: unzip data/creditcardfraud.zip -d data/")
        sys.exit(1)
    
    run_benchmark(
        api_url=args.api_url,
        csv_path=args.csv_path,
        limit=args.limit,
        delay=args.delay,
        verbose=args.verbose,
        balanced=args.balanced
    )


if __name__ == "__main__":
    main()
