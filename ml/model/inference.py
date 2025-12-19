"""
ML Inference Module
===================

Production inference pipeline combining XGBoost and LSTM models
for real-time risk prediction with explainability support.

Ensemble Strategy:
------------------
- XGBoost: Interpretable feature-based risk (60% weight)
- LSTM: Temporal pattern-based risk (40% weight)
- Final score: Weighted combination with confidence metrics

Explainability Features:
------------------------
- Feature importance ranking for XGBoost predictions
- Top contributing features per prediction
- Confidence intervals based on probability entropy
- Model disagreement detection

IMPORTANT LIMITATIONS:
----------------------
- Models are trained on SYNTHETIC data only
- Risk scores are NOT clinically validated
- This module is for RESEARCH/DEVELOPMENT use only
- Do NOT use for clinical decision-making

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from tensorflow import keras
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Model Loading
# ============================================================================

class NeuroCardiacInference:
    """
    Production inference class for NeuroCardiac Shield.

    Loads both XGBoost and LSTM models and provides unified prediction API.
    """

    def __init__(
        self,
        xgb_model_path: str = "ml/checkpoints/xgboost/xgboost_model.json",
        xgb_scaler_path: str = "ml/checkpoints/xgboost/scaler.pkl",
        lstm_model_path: str = "ml/checkpoints/lstm/lstm_model.keras",
        lstm_scaler_path: str = "ml/checkpoints/lstm/lstm_scaler.pkl"
    ):
        """
        Initialize inference engine.

        Args:
            xgb_model_path: Path to XGBoost model file
            xgb_scaler_path: Path to XGBoost feature scaler
            lstm_model_path: Path to LSTM Keras model
            lstm_scaler_path: Path to LSTM sequence scaler
        """
        self.xgb_model = None
        self.xgb_scaler = None
        self.lstm_model = None
        self.lstm_scaler = None

        # Load XGBoost model
        try:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_model_path)
            self.xgb_scaler = joblib.load(xgb_scaler_path)
            print(f"✓ XGBoost model loaded from {xgb_model_path}")
        except Exception as e:
            print(f"⚠ XGBoost model not loaded: {e}")

        # Load LSTM model
        try:
            self.lstm_model = keras.models.load_model(lstm_model_path)
            self.lstm_scaler = joblib.load(lstm_scaler_path)
            print(f"✓ LSTM model loaded from {lstm_model_path}")
        except Exception as e:
            print(f"⚠ LSTM model not loaded: {e}")

        # Risk category mapping
        self.risk_categories = ['LOW', 'MEDIUM', 'HIGH']

    def predict_xgboost(
        self,
        features: Dict[str, float]
    ) -> Tuple[np.ndarray, float, str]:
        """
        Run XGBoost inference on extracted features.

        Args:
            features: Dictionary of feature values

        Returns:
            probabilities: Class probabilities [P(LOW), P(MEDIUM), P(HIGH)]
            risk_score: Scalar risk score (0-1)
            risk_category: Risk category string
        """
        if self.xgb_model is None or self.xgb_scaler is None:
            raise RuntimeError("XGBoost model not loaded")

        # Convert features dict to DataFrame
        features_df = pd.DataFrame([features])

        # Scale features
        features_scaled = self.xgb_scaler.transform(features_df)

        # Predict
        probabilities = self.xgb_model.predict_proba(features_scaled)[0]
        predicted_class = int(np.argmax(probabilities))
        risk_category = self.risk_categories[predicted_class]

        # Compute scalar risk score (0-1)
        # Weighted combination: 0.0 for LOW, 0.5 for MEDIUM, 1.0 for HIGH
        risk_score = probabilities[0] * 0.0 + probabilities[1] * 0.5 + probabilities[2] * 1.0

        return probabilities, risk_score, risk_category

    def predict_lstm(
        self,
        sequence: np.ndarray
    ) -> Tuple[np.ndarray, float, str]:
        """
        Run LSTM inference on time-series sequence.

        Args:
            sequence: Input sequence (seq_length, n_channels)

        Returns:
            probabilities: Class probabilities
            risk_score: Scalar risk score (0-1)
            risk_category: Risk category string
        """
        if self.lstm_model is None or self.lstm_scaler is None:
            raise RuntimeError("LSTM model not loaded")

        # Ensure correct shape: (1, seq_length, n_channels)
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Scale sequence
        n_samples, seq_len, n_channels = sequence.shape
        sequence_reshaped = sequence.reshape(-1, n_channels)
        sequence_scaled = self.lstm_scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(n_samples, seq_len, n_channels)

        # Predict
        probabilities = self.lstm_model.predict(sequence_scaled, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        risk_category = self.risk_categories[predicted_class]

        # Scalar risk score
        risk_score = probabilities[0] * 0.0 + probabilities[1] * 0.5 + probabilities[2] * 1.0

        return probabilities, risk_score, risk_category

    def predict_ensemble(
        self,
        features: Dict[str, float],
        sequence: np.ndarray,
        xgb_weight: float = 0.6,
        lstm_weight: float = 0.4,
        include_explanations: bool = True
    ) -> Dict:
        """
        Ensemble prediction combining XGBoost and LSTM with explainability.

        Args:
            features: Feature dictionary for XGBoost
            sequence: Time-series sequence for LSTM
            xgb_weight: Weight for XGBoost prediction
            lstm_weight: Weight for LSTM prediction
            include_explanations: If True, include feature importance analysis

        Returns:
            Prediction dictionary with:
            - risk_score: Weighted ensemble score (0-1)
            - risk_category: LOW, MEDIUM, or HIGH
            - confidence: Prediction confidence (0-1)
            - probabilities: Per-class probabilities
            - model_breakdown: Individual model predictions
            - explanations: Feature importance and top contributors (if requested)
            - model_agreement: Whether models agree on category
        """
        # Individual model predictions
        xgb_probs, xgb_score, xgb_category = self.predict_xgboost(features)
        lstm_probs, lstm_score, lstm_category = self.predict_lstm(sequence)

        # Weighted ensemble
        ensemble_probs = xgb_weight * xgb_probs + lstm_weight * lstm_probs
        ensemble_score = xgb_weight * xgb_score + lstm_weight * lstm_score
        ensemble_class = int(np.argmax(ensemble_probs))
        ensemble_category = self.risk_categories[ensemble_class]

        # Confidence: Use entropy of probability distribution
        # Low entropy = high confidence
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10))
        max_entropy = np.log(len(ensemble_probs))
        confidence = 1.0 - (entropy / max_entropy)

        # Model agreement check
        model_agreement = (xgb_category == lstm_category)

        # Build result
        result = {
            'risk_score': float(ensemble_score),
            'risk_category': ensemble_category,
            'confidence': float(confidence),
            'probabilities': {
                'LOW': float(ensemble_probs[0]),
                'MEDIUM': float(ensemble_probs[1]),
                'HIGH': float(ensemble_probs[2])
            },
            'model_breakdown': {
                'xgboost': {
                    'score': float(xgb_score),
                    'category': xgb_category,
                    'probabilities': xgb_probs.tolist()
                },
                'lstm': {
                    'score': float(lstm_score),
                    'category': lstm_category,
                    'probabilities': lstm_probs.tolist()
                }
            },
            'model_agreement': model_agreement
        }

        # Add explainability if requested
        if include_explanations and self.xgb_model is not None:
            result['explanations'] = self._generate_explanations(features)

        return result

    def _generate_explanations(self, features: Dict[str, float]) -> Dict:
        """
        Generate feature-level explanations for prediction.

        This method identifies which features contributed most to the
        prediction, enabling interpretability of the risk score.

        Args:
            features: Input feature dictionary

        Returns:
            Dictionary containing:
            - top_positive_features: Features pushing toward HIGH risk
            - top_negative_features: Features pushing toward LOW risk
            - feature_importance_used: Global feature importance from model
        """
        explanations = {
            'top_contributors': [],
            'feature_importance_summary': {},
            'interpretation_notes': []
        }

        try:
            # Get feature importance from XGBoost model
            importance = self.xgb_model.feature_importances_

            # Load feature names from training (if available)
            feature_names = list(features.keys())

            # Create importance mapping
            if len(importance) == len(feature_names):
                importance_dict = dict(zip(feature_names, importance))

                # Sort by importance
                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Top 10 contributors
                top_10 = sorted_features[:10]
                explanations['top_contributors'] = [
                    {
                        'feature': name,
                        'importance': float(imp),
                        'value': float(features.get(name, 0))
                    }
                    for name, imp in top_10
                ]

                # Category summaries
                hrv_importance = sum(
                    imp for name, imp in importance_dict.items()
                    if name in ['mean_hr', 'sdnn', 'rmssd', 'pnn50',
                               'lf_power', 'hf_power', 'lf_hf_ratio']
                )
                eeg_importance = sum(
                    imp for name, imp in importance_dict.items()
                    if 'power' in name or 'ratio' in name or 'entropy' in name
                ) - hrv_importance

                explanations['feature_importance_summary'] = {
                    'hrv_contribution': float(hrv_importance),
                    'eeg_contribution': float(eeg_importance),
                    'ecg_morphology_contribution': float(
                        sum(imp for name, imp in importance_dict.items()
                            if name in ['mean_qrs_duration', 'qrs_amplitude',
                                       'mean_rr_interval'])
                    )
                }

                # Interpretation notes based on key feature values
                notes = []

                if features.get('lf_hf_ratio', 1.0) > 2.5:
                    notes.append(
                        "Elevated LF/HF ratio suggests sympathetic dominance (stress indicator)"
                    )
                if features.get('sdnn', 50) < 30:
                    notes.append(
                        "Low SDNN indicates reduced heart rate variability"
                    )
                if features.get('rmssd', 30) < 20:
                    notes.append(
                        "Low RMSSD suggests reduced parasympathetic activity"
                    )
                if features.get('mean_hr', 70) > 100:
                    notes.append(
                        "Elevated heart rate detected"
                    )

                # Check alpha/beta ratios
                alpha_beta_ratios = [
                    features.get(f'{ch}_alpha_beta_ratio', 1.0)
                    for ch in ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
                ]
                mean_ab_ratio = np.mean(alpha_beta_ratios)
                if mean_ab_ratio < 0.8:
                    notes.append(
                        "Low alpha/beta ratio may indicate cognitive stress or fatigue"
                    )

                explanations['interpretation_notes'] = notes

        except Exception as e:
            explanations['error'] = str(e)

        return explanations


# ============================================================================
# Standalone Prediction Functions
# ============================================================================

def load_xgboost_model(model_dir: str = "ml/checkpoints/xgboost"):
    """Load XGBoost model and scaler."""
    model = xgb.XGBClassifier()
    model.load_model(f"{model_dir}/xgboost_model.json")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    return model, scaler


def load_lstm_model(model_dir: str = "ml/checkpoints/lstm"):
    """Load LSTM model and scaler."""
    model = keras.models.load_model(f"{model_dir}/lstm_model.keras")
    scaler = joblib.load(f"{model_dir}/lstm_scaler.pkl")
    return model, scaler


def predict_risk(
    features: Dict[str, float],
    sequence: np.ndarray,
    inference_engine: Optional[NeuroCardiacInference] = None
) -> Dict:
    """
    Convenience function for production inference.

    Args:
        features: Extracted features (from cloud.signal_processing.features)
        sequence: Raw time-series (seq_length, n_channels)
        inference_engine: Optional pre-loaded inference engine

    Returns:
        Prediction dictionary
    """
    if inference_engine is None:
        inference_engine = NeuroCardiacInference()

    return inference_engine.predict_ensemble(features, sequence)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    try:
        engine = NeuroCardiacInference()
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

    sample_features = {}
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
    for ch in channel_names:
        sample_features[f'{ch}_delta_power'] = np.random.lognormal(2.0, 1.0)
        sample_features[f'{ch}_theta_power'] = np.random.lognormal(1.5, 0.8)
        sample_features[f'{ch}_alpha_power'] = np.random.lognormal(2.5, 1.2)
        sample_features[f'{ch}_beta_power'] = np.random.lognormal(1.0, 0.6)
        sample_features[f'{ch}_gamma_power'] = np.random.lognormal(0.5, 0.4)
        sample_features[f'{ch}_alpha_beta_ratio'] = np.random.gamma(2.0, 0.5)
        sample_features[f'{ch}_theta_alpha_ratio'] = np.random.gamma(1.5, 0.3)
        sample_features[f'{ch}_spectral_entropy'] = np.random.beta(5, 2)

    sample_features.update({
        'mean_alpha_power': np.random.lognormal(2.5, 1.0),
        'std_alpha_power': np.random.lognormal(1.0, 0.5),
        'mean_hr': 72.0,
        'sdnn': 45.2,
        'rmssd': 38.1,
        'pnn50': 12.5,
        'lf_power': 150.0,
        'hf_power': 120.0,
        'lf_hf_ratio': 1.25,
        'mean_qrs_duration': 88.0,
        'mean_rr_interval': 833.0,
        'qrs_amplitude': 1.15
    })

    sample_sequence = np.random.randn(250, 11) * 20.0

    result = engine.predict_ensemble(sample_features, sample_sequence)

    print(f"Risk Score: {result['risk_score']:.3f}")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Confidence: {result['confidence']:.3f}")
