"""
ML Explainability Module using SHAP.

Provides interpretable explanations for why the Isolation Forest
model flagged specific transactions as anomalous.

SHAP (SHapley Additive exPlanations) values show the contribution
of each feature to the final anomaly score.
"""

import numpy as np
from typing import Optional
import shap


class ModelExplainer:
    """
    SHAP-based explainer for Isolation Forest model.
    
    Computes feature contributions to explain why transactions
    are flagged as anomalous or normal.
    """
    
    # Feature names in the order used by the detector
    FEATURE_NAMES = [
        "amount",
        "hour",
        "day_of_week",
        "is_weekend",
        "location_encoded",
        "merchant_type_encoded"
    ]
    
    # Human-readable feature descriptions
    FEATURE_DESCRIPTIONS = {
        "amount": "Transaction Amount ($)",
        "hour": "Hour of Transaction",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend Transaction",
        "location_encoded": "Location Risk Factor",
        "merchant_type_encoded": "Merchant Category Risk"
    }
    
    def __init__(self, model, background_data: Optional[np.ndarray] = None):
        """
        Initialize the explainer with a trained model.
        
        Args:
            model: Trained Isolation Forest model
            background_data: Background dataset for SHAP (uses 100 samples)
        """
        self.model = model
        self._explainer = None
        self._background_data = background_data
    
    def initialize_explainer(self, background_data: np.ndarray) -> None:
        """
        Initialize SHAP explainer with background data.
        
        For Isolation Forest, we use TreeExplainer which is optimized
        for tree-based models.
        
        Args:
            background_data: Sample of training data for baseline
        """
        # Use a subset of background data (max 100 samples for speed)
        if len(background_data) > 100:
            indices = np.random.choice(len(background_data), 100, replace=False)
            background_data = background_data[indices]
        
        self._background_data = background_data
        
        # TreeExplainer is the most efficient for tree-based models
        self._explainer = shap.TreeExplainer(self.model)
    
    def explain(self, features: np.ndarray) -> dict:
        """
        Compute SHAP values for a single transaction.
        
        Args:
            features: Feature array for one transaction (1D array)
            
        Returns:
            Dictionary with feature contributions and explanations
        """
        if self._explainer is None:
            return self._fallback_explanation(features)
        
        # Ensure 2D array for SHAP
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        try:
            # Compute SHAP values
            shap_values = self._explainer.shap_values(features)
            
            # For Isolation Forest, shap_values is typically (1, n_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            return self._format_explanation(features[0], shap_values)
            
        except Exception as e:
            # Fallback if SHAP computation fails
            print(f"SHAP computation failed: {e}")
            return self._fallback_explanation(features[0])
    
    def _format_explanation(self, features: np.ndarray, shap_values: np.ndarray) -> dict:
        """Format SHAP values into a structured explanation."""
        
        # Normalize SHAP values to percentages (contribution to anomaly)
        total_impact = np.abs(shap_values).sum()
        if total_impact == 0:
            total_impact = 1
        
        contributions = []
        
        for i, (feature_name, shap_value, feature_value) in enumerate(
            zip(self.FEATURE_NAMES, shap_values, features)
        ):
            # IMPORTANT: Isolation Forest decision_function is NEGATIVE for anomalies
            # So NEGATIVE SHAP values push toward anomaly (increase risk)
            # And POSITIVE SHAP values push toward normal (decrease risk)
            # We invert the interpretation to match intuition
            contribution_pct = (shap_value / total_impact) * 100
            
            # Determine impact direction (INVERTED for Isolation Forest)
            if shap_value < -0.01:  # Negative SHAP = increases anomaly risk
                direction = "increases_risk"
                impact = "HIGH" if abs(contribution_pct) > 30 else "MEDIUM" if abs(contribution_pct) > 15 else "LOW"
            elif shap_value > 0.01:  # Positive SHAP = decreases anomaly risk
                direction = "decreases_risk"
                impact = "HIGH" if abs(contribution_pct) > 30 else "MEDIUM" if abs(contribution_pct) > 15 else "LOW"
            else:
                direction = "neutral"
                impact = "NONE"
            
            # Get context-aware display name
            if feature_name == "is_weekend":
                display_name = "Weekend Transaction" if feature_value >= 0.5 else "Weekday Transaction"
            else:
                display_name = self.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
            
            contributions.append({
                "feature": feature_name,
                "display_name": display_name,
                "value": round(float(feature_value), 2),
                "shap_value": round(float(shap_value), 4),
                "contribution_pct": round(float(abs(contribution_pct)), 1),
                "direction": direction,
                "impact": impact,
                "explanation": self._generate_explanation(feature_name, feature_value, shap_value)
            })
        
        # Sort by absolute contribution (most important first)
        contributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
        
        return {
            "status": "success",
            "method": "SHAP TreeExplainer",
            "contributions": contributions,
            "top_factors": [c for c in contributions if c["contribution_pct"] > 10],
            "summary": self._generate_summary(contributions)
        }
    
    def _generate_explanation(self, feature_name: str, value: float, shap_value: float) -> str:
        """Generate human-readable explanation for a feature contribution."""
        
        direction = "increases" if shap_value > 0 else "decreases"
        
        if feature_name == "amount":
            if value > 10000:
                return f"Amount ${value:,.0f} is unusually high, {direction} anomaly risk"
            elif value > 5000:
                return f"Amount ${value:,.0f} is above average, {direction} risk slightly"
            else:
                return f"Amount ${value:,.0f} is within normal range"
        
        elif feature_name == "hour":
            if 2 <= value <= 5:
                return f"Transaction at {int(value)}:00 (unusual hours), {direction} risk"
            elif 9 <= value <= 21:
                return f"Transaction at {int(value)}:00 (business hours), normal pattern"
            else:
                return f"Transaction at {int(value)}:00, {direction} risk slightly"
        
        elif feature_name == "is_weekend":
            if value == 1:
                return f"Weekend transaction, {direction} risk"
            else:
                return f"Weekday transaction, typical pattern"
        
        elif feature_name == "location_encoded":
            return f"Location risk factor: {value:.1f}, {direction} overall risk"
        
        elif feature_name == "merchant_type_encoded":
            return f"Merchant category factor: {value:.1f}, {direction} overall risk"
        
        else:
            return f"Feature {feature_name} = {value}, {direction} risk"
    
    def _generate_summary(self, contributions: list) -> str:
        """Generate a summary of the top contributing factors."""
        top = [c for c in contributions if c["contribution_pct"] > 15]
        
        if not top:
            return "No single feature dominates the risk assessment."
        
        factors = [c["display_name"] for c in top[:3]]
        
        if len(factors) == 1:
            return f"Risk primarily driven by: {factors[0]}"
        elif len(factors) == 2:
            return f"Risk driven by: {factors[0]} and {factors[1]}"
        else:
            return f"Risk driven by: {', '.join(factors[:-1])}, and {factors[-1]}"
    
    def _fallback_explanation(self, features: np.ndarray) -> dict:
        """Provide a rule-based fallback explanation when SHAP is unavailable."""
        
        contributions = []
        
        # Simple heuristic-based importance
        feature_values = dict(zip(self.FEATURE_NAMES, features))
        
        # Amount contribution
        amount = feature_values.get("amount", 0)
        amount_impact = min(amount / 10000, 1.0) * 40  # Max 40% contribution
        contributions.append({
            "feature": "amount",
            "display_name": "Transaction Amount ($)",
            "value": round(float(amount), 2),
            "shap_value": 0,
            "contribution_pct": round(amount_impact, 1),
            "direction": "increases_risk" if amount > 5000 else "neutral",
            "impact": "HIGH" if amount > 10000 else "MEDIUM" if amount > 5000 else "LOW",
            "explanation": f"Amount ${amount:,.0f} impact on risk"
        })
        
        # Hour contribution
        hour = feature_values.get("hour", 12)
        unusual_hour = 2 <= hour <= 5
        hour_impact = 25 if unusual_hour else 5
        contributions.append({
            "feature": "hour",
            "display_name": "Hour of Transaction",
            "value": round(float(hour), 2),
            "shap_value": 0,
            "contribution_pct": round(hour_impact, 1),
            "direction": "increases_risk" if unusual_hour else "neutral",
            "impact": "HIGH" if unusual_hour else "LOW",
            "explanation": f"Transaction at {int(hour)}:00"
        })
        
        # Other features with minimal impact in fallback
        for feature in ["day_of_week", "is_weekend", "location_encoded", "merchant_type_encoded"]:
            value = feature_values.get(feature, 0)
            contributions.append({
                "feature": feature,
                "display_name": self.FEATURE_DESCRIPTIONS.get(feature, feature),
                "value": round(float(value), 2),
                "shap_value": 0,
                "contribution_pct": 5,
                "direction": "neutral",
                "impact": "LOW",
                "explanation": f"{self.FEATURE_DESCRIPTIONS.get(feature, feature)}: {value}"
            })
        
        contributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
        
        return {
            "status": "fallback",
            "method": "Heuristic Analysis (SHAP unavailable)",
            "contributions": contributions,
            "top_factors": [c for c in contributions if c["contribution_pct"] > 15],
            "summary": "Analysis based on rule-based heuristics"
        }


# Singleton instance
_explainer: Optional[ModelExplainer] = None


def get_explainer() -> Optional[ModelExplainer]:
    """Get the singleton explainer instance."""
    global _explainer
    return _explainer


def initialize_explainer(model, background_data: np.ndarray) -> ModelExplainer:
    """Initialize the global explainer with model and background data."""
    global _explainer
    _explainer = ModelExplainer(model)
    _explainer.initialize_explainer(background_data)
    return _explainer


if __name__ == "__main__":
    # Demo explainer (requires trained model)
    print("ML Explainability Module")
    print("=" * 40)
    print("This module provides SHAP-based explanations")
    print("for why the Isolation Forest flags transactions.")
    print()
    print("Features explained:")
    for feature, desc in ModelExplainer.FEATURE_DESCRIPTIONS.items():
        print(f"  - {feature}: {desc}")
