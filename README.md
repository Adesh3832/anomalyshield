# 🛡️ AnomalyShield - Transaction Anomaly Detection

**Real-time ML-powered fraud detection with explainable AI (SHAP), OFAC sanctions screening, and regulatory compliance.**

[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](https://anomalyshield-dashboard.onrender.com)
[![API](https://img.shields.io/badge/API-Docs-blue)](https://anomalyshield.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

## 🌐 Live Deployment

- **Dashboard**: https://anomalyshield-dashboard.onrender.com
- **API**: https://anomalyshield.onrender.com
- **API Docs**: https://anomalyshield.onrender.com/docs

---

## 📌 Current vs. Production Architecture

### Current Deployment (Live on Render)
**REST API with synchronous processing** - suitable for demos and moderate traffic.

```
User → Dashboard → FastAPI API → [ML + SHAP + OFAC] → Response
```

- ✅ **Deployed**: Production-ready REST API
- ✅ **Features**: Isolation Forest, SHAP explainability, OFAC screening
- ✅ **Capacity**: ~100 requests/sec
- ✅ **Latency**: < 500ms per transaction

### Future Production Scale (Kafka + Spark)
**Event-driven streaming architecture** - for high-volume enterprise deployment (see [Scaling to Production](#-scaling-to-production-with-kafka-and-spark) section below).

```
Transactions → Kafka → Spark Streaming → ML Model → Alerts
```

- 📈 **Capacity**: 50K+ transactions/sec
- 📈 **Distributed**: Multi-node Spark cluster
- 📈 **Async**: Kafka message queue

---

## 🚀 Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
PYTHONPATH=. uvicorn src.api:app --reload
```

## 📊 API Usage

### Detect Anomaly

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "user_id": "user_001",
    "amount": 5000,
    "location": "Mumbai",
    "timestamp": "2026-01-14T15:00:00Z",
    "merchant_type": "electronics"
  }'
```

**Response:**
```json
{
  "transaction_id": "txn_001",
  "risk_score": 0.42,
  "risk_level": "MEDIUM",
  "reason_codes": [],
  "recommendation": "APPROVE"
}
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Running Tests

```bash
source venv/bin/activate
PYTHONPATH=. pytest tests/ -v
```

## 🏗️ Current System Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │────▶│   FastAPI       │
│   (Browser)     │     │   /detect       │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
          ┌─────────────┐ ┌───────────┐ ┌──────────┐
          │ Isolation   │ │   SHAP    │ │   OFAC   │
          │   Forest    │ │ Explainer │ │ Screener │
          └──────┬──────┘ └─────┬─────┘ └────┬─────┘
                 │              │            │
                 └──────────────┼────────────┘
                                ▼
                      ┌─────────────────┐
                      │  Risk Score +   │
                      │  Explainability │
                      │  + Sanctions    │
                      └─────────────────┘
```

**Key Components:**
- **Isolation Forest**: Unsupervised ML anomaly detection
- **SHAP Explainer**: Feature contribution analysis (WHY flagged)
- **OFAC Screener**: Sanctions watchlist + high-risk MCC detection
- **Rules Engine**: Velocity checks, amount thresholds, unusual hours


## 📁 Project Structure

```
Transaction_anamoly/
├── src/
│   ├── simulator.py      # Transaction data generator
│   ├── detector.py       # Isolation Forest ML model
│   ├── rules_engine.py   # Compliance rules (velocity/amount/time)
│   └── api.py            # FastAPI endpoints
├── models/
│   └── isolation_forest.joblib  # Trained model (auto-generated)
├── tests/                # pytest test suite
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Detector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 150 | Number of isolation trees |
| `max_samples` | 256 | Samples per tree |
| `contamination` | 0.01 | Expected fraud rate (1%) |

### Rule Thresholds

| Rule | Threshold | Reason Code |
|------|-----------|-------------|
| High Amount | > $10,000 | `HIGH_AMOUNT` |
| Velocity | >3 txns/min from different cities | `VELOCITY_MULTI_CITY` |
| Unusual Time | 2-5 AM | `UNUSUAL_HOUR` |

---

# 🚀 Scaling to Production with Kafka and Spark

## Production Architecture

```
                                    ┌─────────────────────────────────┐
                                    │        Kubernetes Cluster       │
┌──────────────┐                    │  ┌─────────────────────────────┐│
│   Payment    │                    │  │    Spark Structured         ││
│   Gateway    │──▶ Kafka Topic ───▶│  │    Streaming Cluster        ││
│   (Source)   │   "transactions"   │  │  ┌─────────────────────────┐││
└──────────────┘                    │  │  │ Detector (broadcasted)  │││
                                    │  │  │ Rules Engine (per exec) │││
                                    │  │  └─────────────────────────┘││
                                    │  └──────────────┬──────────────┘│
                                    │                 │               │
                                    │                 ▼               │
                                    │  ┌─────────────────────────────┐│
                                    │  │   Kafka Topic "alerts"      ││
                                    │  └──────────────┬──────────────┘│
                                    └─────────────────┼───────────────┘
                                                      │
                           ┌──────────────────────────┼──────────────────────────┐
                           ▼                          ▼                          ▼
                 ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
                 │  Compliance      │      │  Real-time       │      │  Data Lake       │
                 │  Dashboard       │      │  Blocking API    │      │  (Delta Lake)    │
                 └──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Step 1: Kafka Integration

### Producer (Transaction Ingestion)

```python
from confluent_kafka import Producer
import json

producer = Producer({'bootstrap.servers': 'kafka:9092'})

def send_transaction(txn: dict):
    producer.produce(
        topic='transactions',
        key=txn['user_id'].encode(),  # Partition by user for ordering
        value=json.dumps(txn).encode()
    )
    producer.flush()
```

### Consumer Configuration

```python
kafka_config = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'anomaly-detection',
    'auto.offset.reset': 'latest',
    'enable.auto.commit': False  # Manual commit after processing
}
```

## Step 2: Spark Structured Streaming

### PySpark Job

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StringType, FloatType

# Initialize Spark with Kafka
spark = SparkSession.builder \
    .appName("TransactionAnomalyDetection") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Define schema
schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("user_id", StringType()) \
    .add("amount", FloatType()) \
    .add("location", StringType()) \
    .add("timestamp", StringType()) \
    .add("merchant_type", StringType())

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON
transactions = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Broadcast the trained model to all executors
from src.detector import Detector
detector = Detector().load_model("s3://models/isolation_forest.joblib")
broadcast_detector = spark.sparkContext.broadcast(detector)

# UDF for scoring
@udf(returnType=FloatType())
def score_transaction(txn_json):
    import json
    txn = json.loads(txn_json)
    return broadcast_detector.value.score(txn)

# Apply scoring
scored = transactions.withColumn(
    "risk_score", 
    score_transaction(to_json(struct("*")))
)

# Write high-risk to alerts topic
alerts = scored.filter(col("risk_score") > 0.7)

query = alerts.selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "alerts") \
    .option("checkpointLocation", "s3://checkpoints/anomaly") \
    .start()

query.awaitTermination()
```

## Step 3: Velocity Checks with Spark State

```python
from pyspark.sql.functions import window, count, collect_set

# Windowed aggregation for velocity detection
velocity_alerts = transactions \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(
        col("user_id"),
        window(col("timestamp"), "1 minute", "10 seconds")
    ) \
    .agg(
        count("*").alias("txn_count"),
        collect_set("location").alias("unique_cities")
    ) \
    .filter(
        (col("txn_count") > 3) & 
        (size(col("unique_cities")) >= 2)
    )
```

## Step 4: Model Serving at Scale

### Option A: MLflow Model Registry

```python
import mlflow

# Register model
mlflow.sklearn.log_model(
    detector.model,
    "isolation_forest",
    registered_model_name="transaction-anomaly-detector"
)

# Serve via REST
mlflow models serve -m "models:/transaction-anomaly-detector/Production" -p 5001
```

### Option B: Kubernetes Deployment

```yaml
# detector-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detector
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: detector
        image: your-registry/anomaly-detector:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detector
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
```

## Step 5: Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

TRANSACTIONS_PROCESSED = Counter(
    'transactions_processed_total',
    'Total transactions processed',
    ['risk_level']
)

SCORING_LATENCY = Histogram(
    'scoring_latency_seconds',
    'Time to score a transaction'
)

# In your scoring function
with SCORING_LATENCY.time():
    score = detector.score(txn)
TRANSACTIONS_PROCESSED.labels(risk_level=get_risk_level(score)).inc()
```

### Grafana Dashboard

Key metrics to monitor:
- Transactions per second (TPS)
- 95th percentile latency
- High-risk transaction rate
- Model prediction distribution

## Capacity Planning

| Component | Specification | Capacity |
|-----------|--------------|----------|
| Kafka | 3 brokers, 12 partitions | 100K msg/sec |
| Spark | 10 executors, 4 cores each | 50K txns/sec |
| API Pods | 3 replicas, 2 vCPU each | 10K req/sec |

## Next Steps for Production

1. **Feature Store**: Implement Feast or Tecton for real-time feature serving
2. **A/B Testing**: Use shadow mode to compare new models before deployment
3. **Drift Detection**: Monitor for data drift with Evidently or WhyLabs
4. **Explainability**: Add SHAP values for compliance audit trails

---

## 📄 License

MIT License - See LICENSE file for details.
