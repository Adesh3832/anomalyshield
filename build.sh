#!/bin/bash
# Render build script for AnomalyShield API

echo "🔨 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🤖 Training Isolation Forest model..."
python -c "
from src.detector import train_detector
print('Training model with 10,000 samples...')
detector = train_detector(10000)
print('✅ Model trained and saved successfully!')
"

echo "✅ Build completed successfully!"
