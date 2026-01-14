#!/usr/bin/env python3
"""
Train the Kaggle detector on the Credit Card Fraud dataset.

This script trains an Isolation Forest model on the real Kaggle dataset
using PCA features (V1-V28) plus Amount.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kaggle_detector import KaggleDetector


def train_model(csv_path: str, output_path: str = None):
    """Train and save the Kaggle detector."""
    
    print("=" * 60)
    print("🔧 TRAINING KAGGLE ANOMALY DETECTOR")
    print("=" * 60)
    
    # Initialize detector
    detector = KaggleDetector(
        n_estimators=200,  # More trees for better accuracy
        max_samples=512,   # Larger samples for complex patterns
        contamination=0.00173  # Actual fraud rate in dataset
    )
    
    # Train from CSV
    detector.fit_from_csv(csv_path)
    
    # Save model
    saved_path = detector.save_model(output_path)
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"   Model saved to: {saved_path}")
    
    # Quick validation
    print("\n🧪 Quick Validation:")
    
    # Test with sample normal transaction (all zeros)
    normal_features = {f"V{i}": 0.0 for i in range(1, 29)}
    normal_features["Amount"] = 50.0
    normal_score = detector.score(normal_features)
    print(f"   Normal transaction (Amount=$50): {normal_score:.4f}")
    
    # Test with unusual features (extreme values)
    anomaly_features = {f"V{i}": 5.0 for i in range(1, 29)}  # Unusual PCA values
    anomaly_features["Amount"] = 5000.0
    anomaly_score = detector.score(anomaly_features)
    print(f"   Unusual transaction (Amount=$5000): {anomaly_score:.4f}")
    
    return detector


def main():
    parser = argparse.ArgumentParser(
        description="Train Kaggle anomaly detector on Credit Card Fraud dataset"
    )
    parser.add_argument(
        "--csv-path",
        default="data/creditcard.csv",
        help="Path to creditcard.csv"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save trained model"
    )
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"❌ Dataset not found: {args.csv_path}")
        print("   Please download the Kaggle dataset first.")
        sys.exit(1)
    
    train_model(args.csv_path, args.output)


if __name__ == "__main__":
    main()
