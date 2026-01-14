"""
FastAPI Application - Transaction Anomaly Detection API.

Provides the /detect endpoint for real-time transaction risk scoring
with ML-based anomaly detection and rule-based compliance checks.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.detector import Detector, train_detector
from src.rules_engine import RulesEngine
from src.simulator import generate_training_data
from src.ofac_screener import get_ofac_screener, OFACScreener
from src.explainer import ModelExplainer, initialize_explainer, get_explainer


# Global instances
detector: Optional[Detector] = None
rules_engine: Optional[RulesEngine] = None
ofac_screener: Optional[OFACScreener] = None
ml_explainer: Optional[ModelExplainer] = None


# Pydantic models for request/response
class TransactionRequest(BaseModel):
    """Input transaction for anomaly detection."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    location: str = Field(..., description="City where transaction occurred")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    merchant_type: str = Field(..., description="Category of merchant")
    
    # New fields for enhanced fraud detection
    user_name: Optional[str] = Field(None, description="User's full name for OFAC screening")
    country: Optional[str] = Field(None, description="Country for sanctions screening")
    mcc_code: Optional[str] = Field(None, description="Merchant Category Code")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "transaction_id": "txn_12345",
                "user_id": "user_001",
                "amount": 5000.00,
                "location": "Mumbai",
                "timestamp": "2026-01-14T15:00:00Z",
                "merchant_type": "electronics",
                "user_name": "John Smith",
                "country": "India",
                "mcc_code": "5411"
            }]
        }
    }


class DetectionResponse(BaseModel):
    """Response from the anomaly detection endpoint."""
    transaction_id: str = Field(..., description="Echo of input transaction ID")
    risk_score: float = Field(..., ge=0, le=1, description="ML anomaly score (0-1)")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    reason_codes: list[str] = Field(default_factory=list, description="Triggered rule codes")
    reason_details: dict[str, str] = Field(default_factory=dict, description="Detailed descriptions for each reason code")
    sanctions_alerts: list[dict] = Field(default_factory=list, description="OFAC/Sanctions screening alerts")
    explainability: Optional[dict] = Field(None, description="SHAP-based ML explainability showing feature contributions")
    recommendation: str = Field(..., description="Action recommendation: APPROVE, REVIEW, or BLOCK")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "transaction_id": "txn_12345",
                "risk_score": 0.85,
                "risk_level": "HIGH",
                "reason_codes": ["VELOCITY_MULTI_CITY", "HIGH_AMOUNT"],
                "reason_details": {"VELOCITY_MULTI_CITY": "Transaction #5 in 60s window. Cities: Mumbai → Delhi → Bangalore"},
                "sanctions_alerts": [{"type": "OFAC_SDN_MATCH", "severity": "CRITICAL", "description": "Match found on OFAC SDN list"}],
                "explainability": {"summary": "Risk driven by Amount and Hour", "contributions": []},
                "recommendation": "BLOCK"
            }]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


def get_risk_level(score: float) -> str:
    """Convert numeric score to risk level."""
    if score < 0.3:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def get_recommendation(risk_score: float, reason_codes: list[str]) -> str:
    """Determine action recommendation based on score and rules."""
    # High-priority rules always trigger review
    if "VELOCITY_MULTI_CITY" in reason_codes:
        return "BLOCK"
    
    # Combine ML score with rule count
    if risk_score >= 0.8 or len(reason_codes) >= 2:
        return "REVIEW"
    elif risk_score >= 0.5 or len(reason_codes) >= 1:
        return "REVIEW"
    else:
        return "APPROVE"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize detector, rules engine, OFAC screener, and ML explainer on startup."""
    global detector, rules_engine, ofac_screener, ml_explainer
    
    model_path = Path(__file__).parent.parent / "models" / "isolation_forest.joblib"
    
    # Initialize detector
    detector = Detector()
    
    # Generate training data for both model and explainer
    training_data = generate_training_data(5000)  # Background data for SHAP (increased for diversity)
    
    if model_path.exists():
        # Load existing model
        print(f"Loading model from {model_path}")
        detector.load_model(str(model_path))
    else:
        # Train new model on startup
        print("No model found. Training new model...")
        full_training_data = generate_training_data(10000)
        detector.fit(full_training_data)
        detector.save_model(str(model_path))
        print(f"Model saved to {model_path}")
    
    # Initialize rules engine
    rules_engine = RulesEngine()
    
    # Initialize OFAC screener
    ofac_screener = get_ofac_screener()
    print("OFAC/Sanctions screener initialized")
    
    # Initialize ML Explainer with SHAP
    try:
        background_features = detector.get_training_features(training_data)
        ml_explainer = initialize_explainer(detector.model, background_features)
        print("SHAP ML Explainer initialized")
    except Exception as e:
        print(f"SHAP explainer initialization failed (will use fallback): {e}")
        ml_explainer = None
    
    print("API ready!")
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Real-time ML-based anomaly detection for banking compliance",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None and detector._is_fitted
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Alternative health check endpoint."""
    return await health_check()


@app.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(transaction: TransactionRequest):
    """
    Analyze a transaction for potential fraud.
    
    Returns:
        - **risk_score**: ML-based anomaly probability (0-1)
        - **risk_level**: Categorical risk (LOW/MEDIUM/HIGH)
        - **reason_codes**: Compliance rule violations
        - **recommendation**: Suggested action (APPROVE/REVIEW/BLOCK)
    """
    if detector is None or not detector._is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Detector model not initialized"
        )
    
    if rules_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Rules engine not initialized"
        )
    
    # Convert to dict for processing
    txn_dict = transaction.model_dump()
    
    # Get ML anomaly score
    risk_score = detector.score(txn_dict)
    
    # Get rule-based results with full descriptions
    rule_results = rules_engine.evaluate(txn_dict)
    reason_codes = [r.reason_code for r in rule_results]
    reason_details = {r.reason_code: r.description for r in rule_results}
    
    # Perform OFAC/Sanctions screening
    sanctions_alerts = []
    if ofac_screener is not None:
        screening_results = ofac_screener.screen_transaction(txn_dict)
        for result in screening_results:
            alert = {
                "type": result.reason_code,
                "severity": "CRITICAL" if result.risk_score >= 1.0 else ("HIGH" if result.risk_score >= 0.7 else "MEDIUM"),
                "list_name": result.list_name,
                "matched_entry": result.matched_entry,
                "description": result.description,
                "risk_score": result.risk_score
            }
            sanctions_alerts.append(alert)
            
            # Add to reason codes if not already present
            if result.reason_code and result.reason_code not in reason_codes:
                reason_codes.append(result.reason_code)
                reason_details[result.reason_code] = result.description
    
    # Determine risk level and recommendation
    risk_level = get_risk_level(risk_score)
    
    # If any CRITICAL sanctions alert, force BLOCK
    has_critical_sanction = any(a.get("severity") == "CRITICAL" for a in sanctions_alerts)
    has_high_sanction = any(a.get("severity") == "HIGH" for a in sanctions_alerts)
    
    if has_critical_sanction:
        recommendation = "BLOCK"
        risk_level = "HIGH"
    elif has_high_sanction:
        recommendation = "REVIEW"
        if risk_level == "LOW":
            risk_level = "MEDIUM"
    else:
        recommendation = get_recommendation(risk_score, reason_codes)
    
    # Compute ML Explainability using SHAP
    explainability = None
    if ml_explainer is not None and detector is not None:
        try:
            features, _ = detector.get_features_for_explanation(txn_dict)
            explainability = ml_explainer.explain(features)
        except Exception as e:
            print(f"Explainability computation failed: {e}")
            explainability = {"status": "error", "message": str(e)}
    
    return DetectionResponse(
        transaction_id=transaction.transaction_id,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        reason_codes=reason_codes,
        reason_details=reason_details,
        sanctions_alerts=sanctions_alerts,
        explainability=explainability,
        recommendation=recommendation
    )


@app.post("/batch-detect", response_model=list[DetectionResponse])
async def detect_batch(transactions: list[TransactionRequest]):
    """
    Analyze multiple transactions in batch.
    
    Limited to 100 transactions per request.
    """
    if len(transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 transactions per batch"
        )
    
    results = []
    for txn in transactions:
        result = await detect_anomaly(txn)
        results.append(result)
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
