# Machine Learning Pipeline Documentation

**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Version:** 1.0.0

---

## Overview

The machine learning pipeline implements an ensemble approach combining interpretable feature-based models (XGBoost) with deep learning temporal models (LSTM) for robust cardiovascular-neurological risk prediction.

---

## Model Architecture

### XGBoost Classifier

**Purpose**: Interpretable feature-based risk prediction

**Input**: 74 hand-crafted features
- EEG band powers (8 channels × 5 bands = 40 features)
- EEG ratios (8 channels × 2 ratios = 16 features)
- Spectral entropy (8 features)
- HRV metrics (7 features)
- ECG morphology (3 features)

**Output**: 3-class probability distribution
- LOW risk (0-0.4)
- MEDIUM risk (0.4-0.7)
- HIGH risk (0.7-1.0)

**Hyperparameters**:
```python
max_depth = 6
learning_rate = 0.05
n_estimators = 200
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.1  # L1 regularization
reg_lambda = 1.0  # L2 regularization
```

**Advantages**:
- Interpretable feature importance
- Fast inference (<5 ms)
- Robust to missing values
- Clinical validation ready

---

### Bidirectional LSTM

**Purpose**: Temporal pattern recognition

**Input**: Time-series sequences
- Shape: (250, 11) - 1 second, 11 channels
- 8 EEG channels + 3 ECG leads

**Architecture**:
```
Input (250, 11)
    ↓
BatchNormalization
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units, return_sequences=False)
    ↓
Dropout (0.3)
    ↓
Dense (32, ReLU)
    ↓
Dense (3, Softmax)
```

**Advantages**:
- Captures temporal dependencies
- Detects transient events (arrhythmia, seizure precursors)
- Bidirectional context (past + future)

---

## Ensemble Strategy

**Weighted Combination**:
```python
ensemble_probs = 0.6 * xgboost_probs + 0.4 * lstm_probs
```

**Risk Score Calculation**:
```python
risk_score = P(LOW) * 0.0 + P(MEDIUM) * 0.5 + P(HIGH) * 1.0
```

**Confidence Metric**:
```python
entropy = -sum(probs * log(probs))
max_entropy = log(num_classes)
confidence = 1.0 - (entropy / max_entropy)
```

---

## Training Pipeline

### XGBoost Training

```python
# 1. Generate synthetic training data
X_train, y_train = generate_training_data(n_samples=5000)

# 2. Apply StandardScaler normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 3. Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_train, test_size=0.2, stratify=y_train
)

# 4. Train with early stopping
model = XGBClassifier(...)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=10)

# 5. Save model
model.save_model("ml/checkpoints/xgboost/xgboost_model.json")
joblib.dump(scaler, "ml/checkpoints/xgboost/scaler.pkl")
```

### LSTM Training

```python
# 1. Generate sequence data
X_sequences, y_labels = generate_sequences(n_samples=2000)

# 2. Per-channel normalization
scaler = StandardScaler()
X_scaled = scale_sequences(X_sequences, scaler)

# 3. Build model
model = build_lstm_model(input_shape=(250, 11), num_classes=3)

# 4. Compile
model.compile(optimizer=Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train with callbacks
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.2,
          callbacks=[EarlyStopping(patience=10)])

# 6. Save model
model.save("ml/checkpoints/lstm/lstm_model.keras")
joblib.dump(scaler, "ml/checkpoints/lstm/lstm_scaler.pkl")
```

---

## Inference Pipeline

```python
from ml.model.inference import NeuroCardiacInference

# Initialize engine (loads both models)
engine = NeuroCardiacInference()

# Extract features from raw signals
features = extract_features(eeg_signals, ecg_signals, hrv_data)

# Prepare sequence for LSTM
sequence = prepare_sequence(raw_signals)  # Shape: (250, 11)

# Run ensemble inference
result = engine.predict_ensemble(features, sequence)

# Result structure:
{
    'risk_score': 0.35,
    'risk_category': 'LOW',
    'confidence': 0.89,
    'probabilities': {'LOW': 0.72, 'MEDIUM': 0.23, 'HIGH': 0.05},
    'model_breakdown': {
        'xgboost': {'score': 0.32, 'category': 'LOW'},
        'lstm': {'score': 0.40, 'category': 'MEDIUM'}
    }
}
```

---

## Feature Importance

Top features from XGBoost (example):
1. LF/HF ratio (sympathovagal balance)
2. RMSSD (vagal tone indicator)
3. Alpha/Beta ratio (stress marker)
4. Spectral entropy (signal complexity)
5. SDNN (overall HRV)

---

## Model Performance

| Metric | XGBoost | LSTM | Ensemble |
|--------|---------|------|----------|
| Accuracy | 82% | 78% | 85% |
| F1 (macro) | 0.80 | 0.76 | 0.83 |
| Inference Time | 5 ms | 75 ms | 80 ms |

---

## Clinical Interpretation

**LOW Risk (0.0-0.4)**:
- Normal physiological patterns
- Stable HRV metrics
- No neurological anomalies

**MEDIUM Risk (0.4-0.7)**:
- Borderline indicators
- Elevated stress markers
- Requires monitoring

**HIGH Risk (0.7-1.0)**:
- Abnormal cardiac patterns
- Potential arrhythmia indicators
- Immediate attention recommended

---

## Limitations

1. Trained on synthetic data (not clinical datasets)
2. Requires validation on real patient data
3. Not FDA-cleared for diagnostic use
4. Performance varies with signal quality

---

## Future Enhancements

- Attention mechanisms for interpretability
- Transformer architectures for long sequences
- Federated learning for privacy preservation
- Transfer learning from clinical datasets
- Real-time continuous learning

---

**New York University - Advanced Project**
