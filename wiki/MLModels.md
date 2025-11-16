# Machine Learning Models

## Ensemble Approach

Combines two complementary models:
- **XGBoost**: Interpretable feature analysis
- **LSTM**: Temporal pattern recognition

## XGBoost Classifier

**Input**: 74 hand-crafted features
- EEG band powers (40 features)
- EEG ratios (16 features)
- Spectral entropy (8 features)
- HRV metrics (7 features)
- ECG morphology (3 features)

**Output**: 3-class risk (LOW, MEDIUM, HIGH)

**Inference**: ~5 ms

## Bidirectional LSTM

**Input**: Time-series (250 timesteps × 11 channels)

**Architecture**:
```
Input → BatchNorm → BiLSTM(128) → Dropout(0.3) → BiLSTM(64) → Dropout(0.3) → Dense(32) → Dense(3)
```

**Inference**: ~75 ms

## Ensemble Strategy

```python
final = 0.6 × XGBoost + 0.4 × LSTM
```

## Risk Categories

| Score | Category | Action |
|-------|----------|--------|
| 0.0-0.4 | LOW | Normal |
| 0.4-0.7 | MEDIUM | Monitor |
| 0.7-1.0 | HIGH | Attention |

## Confidence Calculation

Based on prediction entropy:
- Low entropy = High confidence
- High entropy = Low confidence

## Training

Models trained on synthetic physiological data.
Real clinical validation required before deployment.
