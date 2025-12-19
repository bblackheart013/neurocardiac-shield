"""
ML Model Module
===============

This module provides machine learning models and inference capabilities
for the NeuroCardiac Shield physiological risk assessment system.

Components:
-----------
- train_xgboost: XGBoost classifier training pipeline
- train_lstm: Bidirectional LSTM training pipeline
- inference: Production inference engine with ensemble prediction

Quick Start:
------------
    from ml.model.inference import NeuroCardiacInference, predict_risk

    # Initialize inference engine
    engine = NeuroCardiacInference()

    # Run prediction
    result = engine.predict_ensemble(features, sequence)
    print(f"Risk: {result['risk_category']} ({result['risk_score']:.2f})")

Model Architecture:
-------------------
- XGBoost: 200 trees, max_depth=6, 76 input features
- LSTM: Bidirectional (128→64 units), 250×11 input sequence
- Ensemble: 60% XGBoost + 40% LSTM weighted combination

IMPORTANT LIMITATIONS:
----------------------
- Models are trained on SYNTHETIC data only
- Risk scores are NOT clinically validated
- Do NOT use for medical decision-making
- See docs/ML_PIPELINE.md for full documentation

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 2.0.0
"""

from .inference import (
    NeuroCardiacInference,
    load_xgboost_model,
    load_lstm_model,
    predict_risk
)

__all__ = [
    'NeuroCardiacInference',
    'load_xgboost_model',
    'load_lstm_model',
    'predict_risk'
]

__version__ = '2.0.0'
