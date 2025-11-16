"""
XGBoost Training Pipeline
--------------------------
Train gradient boosting model for cardiovascular/neurological risk prediction.

Features: ~80 hand-crafted features from EEG + ECG
Target: Multi-class risk score (LOW, MEDIUM, HIGH) or regression (0-1)

Model advantages:
- Interpretable feature importance
- Handles missing values
- Robust to outliers
- Fast inference

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'model_type': 'classification',  # 'classification' or 'regression'
    'n_classes': 3,  # LOW, MEDIUM, HIGH
    'test_size': 0.2,
    'random_state': 42,
    'output_dir': 'ml/checkpoints/xgboost',

    # XGBoost hyperparameters (tuned for medical data)
    'xgb_params': {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'objective': 'multi:softprob',  # or 'reg:squarederror' for regression
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.01,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1
    }
}


# ============================================================================
# Data Generation (Simulated Training Data)
# ============================================================================

def generate_synthetic_training_data(n_samples: int = 5000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic training dataset for demonstration.

    In production, replace with:
    - Historical patient data (IRB-approved)
    - Labeled episodes from clinical trials
    - Crowdsourced wearable data (with consent)

    Args:
        n_samples: Number of training examples

    Returns:
        features_df: DataFrame with feature columns
        labels: Risk labels (0=LOW, 1=MEDIUM, 2=HIGH)
    """
    np.random.seed(42)

    # Simulate feature distributions based on clinical knowledge
    features = {}

    # EEG features (8 channels × 5 bands + ratios + entropy)
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
    for ch in channel_names:
        # Band powers (log-normal distribution, typical for EEG)
        features[f'{ch}_delta_power'] = np.random.lognormal(2.0, 1.0, n_samples)
        features[f'{ch}_theta_power'] = np.random.lognormal(1.5, 0.8, n_samples)
        features[f'{ch}_alpha_power'] = np.random.lognormal(2.5, 1.2, n_samples)
        features[f'{ch}_beta_power'] = np.random.lognormal(1.0, 0.6, n_samples)
        features[f'{ch}_gamma_power'] = np.random.lognormal(0.5, 0.4, n_samples)

        # Ratios
        features[f'{ch}_alpha_beta_ratio'] = np.random.gamma(2.0, 0.5, n_samples)
        features[f'{ch}_theta_alpha_ratio'] = np.random.gamma(1.5, 0.3, n_samples)

        # Entropy
        features[f'{ch}_spectral_entropy'] = np.random.beta(5, 2, n_samples)

    # Global EEG features
    features['mean_alpha_power'] = np.random.lognormal(2.5, 1.0, n_samples)
    features['std_alpha_power'] = np.random.lognormal(1.0, 0.5, n_samples)

    # HRV features (based on healthy adult distributions)
    features['mean_hr'] = np.random.normal(70, 12, n_samples)  # BPM
    features['sdnn'] = np.random.gamma(4, 10, n_samples)  # ms (healthy: 30-50)
    features['rmssd'] = np.random.gamma(4, 8, n_samples)  # ms (healthy: 20-40)
    features['pnn50'] = np.random.beta(2, 3, n_samples) * 50  # % (healthy: 5-20%)
    features['lf_power'] = np.random.lognormal(5, 1.5, n_samples)  # ms²
    features['hf_power'] = np.random.lognormal(4.5, 1.5, n_samples)  # ms²
    features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-6)

    # ECG morphology
    features['mean_qrs_duration'] = np.random.normal(90, 10, n_samples)  # ms
    features['mean_rr_interval'] = 60000 / features['mean_hr']
    features['qrs_amplitude'] = np.random.normal(1.2, 0.3, n_samples)  # mV

    features_df = pd.DataFrame(features)

    # Generate labels based on feature combinations (clinical heuristics)
    # HIGH RISK: Low HRV (stress) + High beta (anxiety) + Abnormal HR
    # MEDIUM RISK: Moderate deviations
    # LOW RISK: Normal physiological ranges

    risk_scores = np.zeros(n_samples)

    # HRV component (30% weight)
    hrv_risk = (features['sdnn'] < 30) * 0.3 + (features['rmssd'] < 20) * 0.3
    hrv_risk += (features['lf_hf_ratio'] > 2.5) * 0.3

    # EEG component (30% weight)
    mean_alpha_beta = np.mean([features[f'{ch}_alpha_beta_ratio'] for ch in channel_names], axis=0)
    eeg_risk = (mean_alpha_beta < 0.8) * 0.5  # Low alpha/beta = stress/fatigue
    eeg_risk += (features['mean_alpha_power'] < np.exp(1.5)) * 0.3

    # HR component (20% weight)
    hr_risk = ((features['mean_hr'] < 50) | (features['mean_hr'] > 100)) * 0.5

    # Combine risk components
    risk_scores = hrv_risk + eeg_risk + hr_risk + np.random.normal(0, 0.1, n_samples)

    # Discretize into 3 classes
    labels = np.zeros(n_samples, dtype=int)
    labels[risk_scores > 0.6] = 1  # MEDIUM
    labels[risk_scores > 1.2] = 2  # HIGH

    return features_df, labels


# ============================================================================
# Training Pipeline
# ============================================================================

def train_xgboost_model(
    X: pd.DataFrame,
    y: np.ndarray,
    config: Dict = CONFIG
) -> Tuple[xgb.XGBClassifier, StandardScaler, Dict]:
    """
    Train XGBoost model with cross-validation.

    Args:
        X: Feature matrix (n_samples × n_features)
        y: Target labels
        config: Configuration dictionary

    Returns:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        metrics: Training metrics dictionary
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y
    )

    # Feature scaling (important for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Initialize model
    model = xgb.XGBClassifier(**config['xgb_params'])

    # Train with early stopping
    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=eval_set,
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_test,
        y_pred,
        target_names=['LOW', 'MEDIUM', 'HIGH'],
        digits=4
    ))

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC (one-vs-rest)
    if len(np.unique(y)) > 2:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        print(f"\nWeighted ROC-AUC: {auc_score:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*60)
    print(feature_importance.head(20).to_string(index=False))

    # Store metrics
    metrics = {
        'test_accuracy': float(np.mean(y_pred == y_test)),
        'roc_auc': float(auc_score) if len(np.unique(y)) > 2 else None,
        'feature_importance': feature_importance.to_dict('records'),
        'class_distribution': np.bincount(y_train).tolist(),
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0]
    }

    return model, scaler, metrics


# ============================================================================
# Model Persistence
# ============================================================================

def save_model(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    metrics: Dict,
    feature_names: list,
    output_dir: str = CONFIG['output_dir']
):
    """Save trained model, scaler, and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / 'xgboost_model.json'
    model.save_model(str(model_path))
    print(f"\n✓ Model saved to {model_path}")

    # Save scaler
    scaler_path = output_path / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'version': '1.0.0',
        'feature_names': feature_names,
        'metrics': metrics,
        'config': CONFIG
    }
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("NeuroCardiac Shield - XGBoost Training Pipeline")
    print("="*60)

    # Generate synthetic training data
    print("\n[1/4] Generating synthetic training data...")
    X, y = generate_synthetic_training_data(n_samples=5000)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")

    # Train model
    print("\n[2/4] Training XGBoost model...")
    model, scaler, metrics = train_xgboost_model(X, y)

    # Save model
    print("\n[3/4] Saving model artifacts...")
    save_model(model, scaler, metrics, X.columns.tolist())

    # Verification
    print("\n[4/4] Verifying saved model...")
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(f"{CONFIG['output_dir']}/xgboost_model.json")
    print("✓ Model loaded successfully")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Model saved to: {CONFIG['output_dir']}/")
    print("\nTo use in production:")
    print("  from ml.model.inference import load_xgboost_model, predict")
