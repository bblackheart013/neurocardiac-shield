# NeuroCardiac Shield — Machine Learning Pipeline

**Document Type**: Technical Specification
**Version**: 2.1.0
**Last Updated**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav Devram Chandgir

---

## Table of Contents

1. [Overview](#1-overview)
2. [Feature Engineering](#2-feature-engineering)
3. [XGBoost Model](#3-xgboost-model)
4. [LSTM Model](#4-lstm-model)
5. [Ensemble Strategy](#5-ensemble-strategy)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Reproducibility](#7-reproducibility)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [Ethical Considerations](#9-ethical-considerations)

---

## 1. Overview

The NeuroCardiac Shield ML pipeline implements a dual-model ensemble for physiological risk assessment. The design prioritizes interpretability over raw accuracy, reflecting the requirements of medical AI applications where clinicians must understand and validate predictions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE OVERVIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           RAW SIGNALS                                        │
│                    ┌──────────┬──────────┐                                  │
│                    │   EEG    │   ECG    │                                  │
│                    │  8 ch    │ 3 leads  │                                  │
│                    └────┬─────┴────┬─────┘                                  │
│                         │          │                                         │
│           ┌─────────────┴──────────┴─────────────┐                          │
│           │                                       │                          │
│           ▼                                       ▼                          │
│   ┌───────────────────┐               ┌───────────────────┐                 │
│   │ FEATURE EXTRACTION│               │ SEQUENCE PREP     │                 │
│   │ ─────────────────│               │ ─────────────     │                 │
│   │ 76 hand-crafted   │               │ Shape: (250, 11)  │                 │
│   │ features          │               │ 1-sec window      │                 │
│   └─────────┬─────────┘               └─────────┬─────────┘                 │
│             │                                   │                            │
│             ▼                                   ▼                            │
│   ┌───────────────────┐               ┌───────────────────┐                 │
│   │     XGBOOST       │               │     BI-LSTM       │                 │
│   │     ─────────     │               │     ───────       │                 │
│   │  200 trees        │               │  128→64 units     │                 │
│   │  Interpretable    │               │  Temporal         │                 │
│   └─────────┬─────────┘               └─────────┬─────────┘                 │
│             │                                   │                            │
│             │    P(LOW), P(MED), P(HIGH)       │                            │
│             │                                   │                            │
│             └─────────────┬─────────────────────┘                            │
│                           │                                                  │
│                           ▼                                                  │
│                  ┌─────────────────┐                                        │
│                  │    ENSEMBLE     │                                        │
│                  │    ────────     │                                        │
│                  │  60% XGB        │                                        │
│                  │  40% LSTM       │                                        │
│                  └────────┬────────┘                                        │
│                           │                                                  │
│                           ▼                                                  │
│                  ┌─────────────────┐                                        │
│                  │     OUTPUT      │                                        │
│                  │     ──────      │                                        │
│                  │ • Risk Score    │                                        │
│                  │ • Category      │                                        │
│                  │ • Confidence    │                                        │
│                  │ • Explanations  │                                        │
│                  └─────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Faculty Note — Why Interpretability Over Accuracy?**
>
> In medical ML, a model that achieves 95% accuracy but cannot explain its predictions is often less valuable than an 85% accurate model with clear feature attribution. Clinicians need to verify that predictions align with physiological reasoning. If a model flags "HIGH risk" but the explanation shows it's driven by noise rather than meaningful features (e.g., elevated LF/HF ratio), the clinician can identify and correct the error. This project deliberately weights the interpretable XGBoost model more heavily than the black-box LSTM.

---

## 2. Feature Engineering

### 2.1 Feature Architecture

All 76 features have documented clinical meaning, enabling physicians to validate predictions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        76-FEATURE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   EEG FEATURES (66 total)                                                   │
│   ───────────────────────                                                   │
│                                                                              │
│   Per-Channel Features (8 features × 8 channels = 64)                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Channel │ δ-pwr │ θ-pwr │ α-pwr │ β-pwr │ γ-pwr │ α/β  │ θ/α │ H(S) │   │
│   ├─────────┼───────┼───────┼───────┼───────┼───────┼──────┼─────┼──────┤   │
│   │   Fp1   │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   Fp2   │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   C3    │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   C4    │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   T3    │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   T4    │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   O1    │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   │   O2    │   ✓   │   ✓   │   ✓   │   ✓   │   ✓   │  ✓   │  ✓  │  ✓   │   │
│   └─────────┴───────┴───────┴───────┴───────┴───────┴──────┴─────┴──────┘   │
│                                                                              │
│   Legend: δ=delta, θ=theta, α=alpha, β=beta, γ=gamma, H(S)=spectral entropy│
│                                                                              │
│   Global EEG Features (2)                                                    │
│   ┌─────────────────────────────────────────┐                               │
│   │ mean_alpha_power  │ Cross-channel mean  │                               │
│   │ std_alpha_power   │ Cross-channel std   │                               │
│   └─────────────────────────────────────────┘                               │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                              │
│   HRV FEATURES (7 total)                                                    │
│   ──────────────────────                                                    │
│   ┌─────────────┬───────────────────────────────────────────────────────┐   │
│   │   Feature   │            Computation / Meaning                      │   │
│   ├─────────────┼───────────────────────────────────────────────────────┤   │
│   │ mean_hr     │ 60000 / mean(RR) — beats per minute                   │   │
│   │ sdnn        │ std(RR) — overall HRV, reduced = poor adaptability    │   │
│   │ rmssd       │ sqrt(mean(diff(RR)²)) — parasympathetic activity      │   │
│   │ pnn50       │ % of |diff(RR)| > 50ms — vagal tone marker            │   │
│   │ lf_power    │ ∫PSD(0.04-0.15Hz) — sympathetic + parasympathetic     │   │
│   │ hf_power    │ ∫PSD(0.15-0.4Hz) — respiratory/parasympathetic        │   │
│   │ lf_hf_ratio │ LF/HF — sympathovagal balance (stress indicator)      │   │
│   └─────────────┴───────────────────────────────────────────────────────┘   │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                              │
│   ECG MORPHOLOGY (3 total)                                                  │
│   ────────────────────────                                                  │
│   ┌─────────────────────┬───────────────────────────────────────────────┐   │
│   │ mean_qrs_duration   │ Width at 50% R amplitude (ms)                 │   │
│   │ mean_rr_interval    │ Mean inter-beat interval (ms)                 │   │
│   │ qrs_amplitude       │ Mean R-peak height (mV)                       │   │
│   └─────────────────────┴───────────────────────────────────────────────┘   │
│                                                                              │
│   TOTAL: 64 + 2 + 7 + 3 = 76 FEATURES                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 EEG Frequency Bands

Frequency bands follow IFCN (International Federation of Clinical Neurophysiology) standards:

| Band | Range (Hz) | Physiological Association | Clinical Relevance |
|------|-----------|--------------------------|-------------------|
| Delta (δ) | 0.5 – 4 | Deep sleep, pathology | Elevated in drowsy/pathological states |
| Theta (θ) | 4 – 8 | Drowsiness, memory encoding | Elevated during cognitive load |
| Alpha (α) | 8 – 13 | Relaxed wakefulness | Suppressed by eye opening, anxiety |
| Beta (β) | 13 – 30 | Active cognition, alertness | Elevated during concentration |
| Gamma (γ) | 30 – 100 | Perception, attention binding | Sensitive to cognitive processing |

### 2.3 HRV Reference Ranges

Reference values from Task Force of ESC and NASPE (1996):

| Metric | Normal Range | Interpretation of Low Values |
|--------|-------------|------------------------------|
| SDNN | 141 ± 39 ms (24h) | Reduced cardiac adaptability |
| RMSSD | 27 ± 12 ms | Reduced vagal/parasympathetic tone |
| pNN50 | Variable | Reduced beat-to-beat variability |
| LF/HF | 1.5 – 2.0 | Values > 2.5 suggest sympathetic dominance |

> **Faculty Note — Why Hand-Crafted Features?**
>
> While end-to-end deep learning has achieved impressive results in some domains, medical applications benefit from hand-crafted features for several reasons: (1) Features like SDNN and LF/HF ratio have decades of clinical research establishing their meaning; (2) Clinicians can directly inspect feature values to validate predictions; (3) Features are transferable across different patient populations without retraining. The cost is that we may miss subtle patterns that only deep learning would detect—but this trade-off favors safety and interpretability.

---

## 3. XGBoost Model

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         XGBOOST CLASSIFIER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT                                                                      │
│   ─────                                                                      │
│   76-dimensional feature vector                                              │
│   StandardScaler normalization (mean=0, std=1)                              │
│                                                                              │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │                                                                        │ │
│   │   GRADIENT BOOSTED TREES                                              │ │
│   │   ──────────────────────                                              │ │
│   │                                                                        │ │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐            │ │
│   │   │ Tree 1  │  │ Tree 2  │  │ Tree 3  │  ...  │ Tree200 │            │ │
│   │   │         │  │         │  │         │       │         │            │ │
│   │   │ max_d=6 │  │ max_d=6 │  │ max_d=6 │       │ max_d=6 │            │ │
│   │   └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘            │ │
│   │        │            │            │                 │                  │ │
│   │        └────────────┴────────────┴─────────────────┘                  │ │
│   │                           │                                            │ │
│   │                           │  Additive combination                      │ │
│   │                           │  with learning rate = 0.05                 │ │
│   │                           ▼                                            │ │
│   │                    ┌─────────────┐                                    │ │
│   │                    │   Softmax   │                                    │ │
│   │                    │ (3 classes) │                                    │ │
│   │                    └─────────────┘                                    │ │
│   │                                                                        │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│   OUTPUT                                                                     │
│   ──────                                                                     │
│   [P(LOW), P(MEDIUM), P(HIGH)]                                              │
│                                                                              │
│   + Feature importance scores for explainability                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_estimators | 200 | Balance between complexity and overfitting |
| max_depth | 6 | Captures feature interactions, prevents overfitting |
| learning_rate | 0.05 | Low rate with more trees for stability |
| subsample | 0.8 | Row subsampling reduces variance |
| colsample_bytree | 0.8 | Feature subsampling reduces correlation |
| min_child_weight | 3 | Prevents leaves with too few samples |
| reg_alpha | 0.01 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |

### 3.3 Training Results (Synthetic Data)

| Metric | Value | Notes |
|--------|-------|-------|
| Test Accuracy | 81.1% | 3-class classification |
| ROC-AUC (weighted) | 0.923 | One-vs-rest averaging |
| Training Samples | 4,000 | After 80/20 split |
| Test Samples | 1,000 | Held-out evaluation |

### 3.4 Top Feature Importances

The following features contribute most to predictions:

| Rank | Feature | Importance | Clinical Interpretation |
|------|---------|------------|------------------------|
| 1 | lf_hf_ratio | 0.0416 | Sympathovagal balance |
| 2 | rmssd | 0.0358 | Parasympathetic activity |
| 3 | sdnn | 0.0308 | Overall HRV |
| 4 | mean_hr | 0.0297 | Heart rate |
| 5 | mean_alpha_power | 0.0286 | Relaxation state |
| 6 | mean_rr_interval | 0.0250 | Inter-beat timing |
| 7 | T3_alpha_beta_ratio | 0.0222 | Left temporal arousal |
| 8 | C4_alpha_beta_ratio | 0.0221 | Right central arousal |
| 9 | C3_alpha_beta_ratio | 0.0210 | Left central arousal |
| 10 | O1_alpha_beta_ratio | 0.0203 | Left occipital state |

**Observation**: HRV features dominate the top importance rankings, which aligns with clinical literature showing HRV as a strong cardiovascular health marker.

---

## 4. LSTM Model

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BIDIRECTIONAL LSTM NETWORK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT                                                                      │
│   ─────                                                                      │
│   Shape: (batch_size, 250, 11)                                              │
│   250 timesteps = 1 second at 250 Hz                                        │
│   11 channels = 8 EEG + 3 ECG                                               │
│                                                                              │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  BatchNormalization                                                    │ │
│   │  Normalizes across the batch for training stability                    │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Bidirectional LSTM (128 units)                                        │ │
│   │  ─────────────────────────────                                         │ │
│   │  return_sequences=True → outputs shape (batch, 250, 256)              │ │
│   │                                                                        │ │
│   │     Forward LSTM ──────────────────────────▶                           │ │
│   │     ◀────────────────────────────── Backward LSTM                      │ │
│   │                                                                        │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Dropout (0.3)                                                         │ │
│   │  Regularization to prevent overfitting                                 │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Bidirectional LSTM (64 units)                                         │ │
│   │  ───────────────────────────                                           │ │
│   │  return_sequences=False → outputs shape (batch, 128)                   │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Dropout (0.3)                                                         │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Dense (32 units, ReLU activation)                                     │ │
│   │  Feature compression before classification                             │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Dropout (0.2)                                                         │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Dense (3 units, Softmax activation)                                   │ │
│   │  Output: [P(LOW), P(MEDIUM), P(HIGH)]                                  │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│   Total Parameters: ~850,000                                                │
│   Training: Adam optimizer, lr=0.001, 50 epochs, batch_size=32             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Training Results (Synthetic Data)

| Metric | Value | Concern Level |
|--------|-------|--------------|
| Test Accuracy | 99.75% | HIGH CONCERN |
| Test AUC | 0.9999 | HIGH CONCERN |
| Test Loss | 0.0089 | - |

> **Faculty Note — Why Is 99.75% Accuracy a Problem?**
>
> Near-perfect accuracy on a classification task is almost always a red flag, not an achievement. There are three likely explanations: (1) **Data leakage**: Training and test sets contain overlapping patterns; (2) **Trivially separable classes**: The synthetic data generator produces risk categories that are easily distinguished (e.g., by simple amplitude thresholds); (3) **Label encoding in features**: The generation process may inadvertently embed the label into the signal. This model would almost certainly fail on real clinical data where patterns are far more subtle. The honest acknowledgment of this limitation is more valuable than claiming high performance.

---

## 5. Ensemble Strategy

### 5.1 Weighted Combination

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENSEMBLE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│       XGBoost Prediction                      LSTM Prediction               │
│       ──────────────────                      ───────────────               │
│                                                                              │
│       P_xgb = [0.15, 0.45, 0.40]              P_lstm = [0.10, 0.30, 0.60]   │
│              [LOW]  [MED] [HIGH]                      [LOW]  [MED] [HIGH]   │
│                                                                              │
│              │                                        │                      │
│              │ × 0.60                                │ × 0.40               │
│              │                                        │                      │
│              ▼                                        ▼                      │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────┐       │
│       │                                                              │       │
│       │   P_ensemble = 0.60 × P_xgb + 0.40 × P_lstm                 │       │
│       │                                                              │       │
│       │             = 0.60 × [0.15, 0.45, 0.40]                     │       │
│       │             + 0.40 × [0.10, 0.30, 0.60]                     │       │
│       │                                                              │       │
│       │             = [0.09 + 0.04, 0.27 + 0.12, 0.24 + 0.24]       │       │
│       │             = [0.13, 0.39, 0.48]                            │       │
│       │                                                              │       │
│       └─────────────────────────────────────────────────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│       Risk Score Calculation:                                                │
│       ───────────────────────                                                │
│       risk_score = P(LOW)×0 + P(MED)×0.5 + P(HIGH)×1.0                      │
│                  = 0.13×0 + 0.39×0.5 + 0.48×1.0                             │
│                  = 0 + 0.195 + 0.48                                          │
│                  = 0.675                                                     │
│                                                                              │
│       Category Mapping:                                                      │
│       ─────────────────                                                      │
│       0.0 – 0.4  → LOW                                                      │
│       0.4 – 0.7  → MEDIUM  ← 0.675 falls here                               │
│       0.7 – 1.0  → HIGH                                                     │
│                                                                              │
│       Final Output: MEDIUM risk (score = 0.675)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Confidence Calculation

Confidence is computed using inverse normalized entropy:

```
entropy = -Σ P(class) × log(P(class))

For uniform distribution [0.33, 0.33, 0.33]:
    entropy = -3 × (0.33 × log(0.33)) = log(3) ≈ 1.10 (maximum)
    confidence = 0 (maximally uncertain)

For certain prediction [0.01, 0.01, 0.98]:
    entropy ≈ 0.11
    confidence = 1 - (0.11 / 1.10) ≈ 0.90 (highly confident)
```

### 5.3 Model Agreement Detection

The system flags when XGBoost and LSTM disagree on category:

| XGBoost Category | LSTM Category | Agreement | Action |
|------------------|---------------|-----------|--------|
| LOW | LOW | Yes | Proceed |
| MEDIUM | MEDIUM | Yes | Proceed |
| HIGH | HIGH | Yes | Proceed |
| LOW | MEDIUM | No | Flag for review |
| MEDIUM | HIGH | No | Flag for review |
| LOW | HIGH | No | Strong disagreement warning |

---

## 6. Inference Pipeline

### 6.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INFERENCE DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. API receives buffer of ~10 seconds of data                             │
│      └── 100 packets × 25 samples = 2500 samples per channel               │
│                                                                              │
│   2. Concatenate EEG and ECG arrays                                         │
│      └── EEG: (8, 2500), ECG: (3, 2500)                                     │
│                                                                              │
│   3. Convert int16 to physical units                                        │
│      └── EEG: value_µV = raw / 10.0                                         │
│      └── ECG: value_mV = raw / 1000.0                                       │
│                                                                              │
│                       ┌───────────────────────────────┐                     │
│                       │                               │                     │
│   4a. XGBoost Path    │   4b. LSTM Path               │                     │
│   ─────────────────   │   ─────────────               │                     │
│                       │                               │                     │
│   ┌─────────────────┐ │   ┌─────────────────┐         │                     │
│   │ Filter signals  │ │   │ Take last 250   │         │                     │
│   │ (bandpass+notch)│ │   │ samples (1 sec) │         │                     │
│   └────────┬────────┘ │   └────────┬────────┘         │                     │
│            │          │            │                  │                     │
│   ┌────────▼────────┐ │   ┌────────▼────────┐         │                     │
│   │ Detect R-peaks  │ │   │ Stack: (250,11) │         │                     │
│   │ Compute RR      │ │   │ 8 EEG + 3 ECG   │         │                     │
│   └────────┬────────┘ │   └────────┬────────┘         │                     │
│            │          │            │                  │                     │
│   ┌────────▼────────┐ │   ┌────────▼────────┐         │                     │
│   │ Extract 76      │ │   │ Scale with      │         │                     │
│   │ features        │ │   │ saved scaler    │         │                     │
│   └────────┬────────┘ │   └────────┬────────┘         │                     │
│            │          │            │                  │                     │
│   ┌────────▼────────┐ │   ┌────────▼────────┐         │                     │
│   │ Scale with      │ │   │ LSTM.predict()  │         │                     │
│   │ saved scaler    │ │   │                 │         │                     │
│   └────────┬────────┘ │   └────────┬────────┘         │                     │
│            │          │            │                  │                     │
│   ┌────────▼────────┐ │            │                  │                     │
│   │ XGB.predict()   │ │            │                  │                     │
│   └────────┬────────┘ │            │                  │                     │
│            │          │            │                  │                     │
│            └──────────┴─────┬──────┘                  │                     │
│                             │                         │                     │
│                       ┌─────▼─────┐                   │                     │
│   5. Ensemble         │           │                   │                     │
│   ───────────         │  Combine  │                   │                     │
│                       │  60%/40%  │                   │                     │
│                       └─────┬─────┘                   │                     │
│                             │                                               │
│   6. Generate Explanations  │                                               │
│   ────────────────────────  │                                               │
│                             ▼                                               │
│                       ┌───────────┐                                        │
│                       │  Return   │                                        │
│                       │  result   │                                        │
│                       └───────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Inference Output Structure

```python
{
    "risk_score": 0.675,
    "risk_category": "MEDIUM",
    "confidence": 0.78,
    "probabilities": {
        "LOW": 0.13,
        "MEDIUM": 0.39,
        "HIGH": 0.48
    },
    "model_breakdown": {
        "xgboost": {
            "score": 0.625,
            "category": "MEDIUM",
            "probabilities": [0.15, 0.45, 0.40]
        },
        "lstm": {
            "score": 0.75,
            "category": "HIGH",
            "probabilities": [0.10, 0.30, 0.60]
        }
    },
    "model_agreement": False,  # XGB=MEDIUM, LSTM=HIGH
    "explanations": {
        "top_contributors": [
            {"feature": "lf_hf_ratio", "importance": 0.042, "value": 2.8},
            {"feature": "rmssd", "importance": 0.036, "value": 22.1},
            {"feature": "sdnn", "importance": 0.031, "value": 35.4}
        ],
        "interpretation_notes": [
            "Elevated LF/HF ratio suggests sympathetic dominance",
            "Low RMSSD indicates reduced parasympathetic activity"
        ]
    }
}
```

---

## 7. Reproducibility

### 7.1 Random Seeds

All random operations use deterministic seeds:

```python
# Training scripts
np.random.seed(42)
tf.random.set_seed(42)

# Synthetic data generation
rng = np.random.default_rng(seed=42)
```

### 7.2 Model Artifacts

| File | Location | Purpose |
|------|----------|---------|
| xgboost_model.json | ml/checkpoints/xgboost/ | Trained XGBoost model |
| scaler.pkl | ml/checkpoints/xgboost/ | Feature StandardScaler |
| lstm_model.keras | ml/checkpoints/lstm/ | Trained LSTM model |
| lstm_scaler.pkl | ml/checkpoints/lstm/ | Sequence StandardScaler |
| metadata.json | ml/checkpoints/*/ | Training config and metrics |

### 7.3 Regenerating Models

```bash
cd ml
source venv/bin/activate

# Train XGBoost (generates 5000 synthetic samples)
python model/train_xgboost.py

# Train LSTM (generates 5000 synthetic sequences)
python model/train_lstm.py
```

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

| Limitation | Severity | Impact |
|------------|----------|--------|
| Synthetic training data | Critical | Models have no clinical validity |
| LSTM overfitting | High | 99.75% accuracy is unrealistic |
| No pathology detection | High | Cannot identify real abnormalities |
| Static risk thresholds | Medium | Not personalized to individuals |
| No probability calibration | Medium | Probabilities may not be reliable |

### 8.2 Required for Clinical Use

1. **Data**: IRB-approved real patient data with expert-annotated ground truth
2. **Validation**: Prospective clinical trials with sensitivity/specificity characterization
3. **Calibration**: Platt scaling or isotonic regression on validation set
4. **Regulatory**: FDA De Novo or 510(k) submission with clinical evidence
5. **Monitoring**: Post-market surveillance for model drift

### 8.3 Potential Improvements

| Enhancement | Complexity | Expected Benefit |
|-------------|------------|------------------|
| Attention mechanisms | Medium | Better temporal feature selection |
| Transfer learning | Medium | Pretrain on PhysioNet datasets |
| Uncertainty quantification | Medium | MC Dropout or ensemble disagreement |
| Online adaptation | High | Personalized baselines per patient |
| Multi-task learning | High | Joint prediction of multiple conditions |

---

## 9. Ethical Considerations

### 9.1 Intended Use

This ML pipeline is designed for:
- Academic research and education
- Demonstration of medical AI architecture
- Proof-of-concept for multi-modal monitoring

This ML pipeline is **NOT** designed for:
- Clinical diagnosis or screening
- Treatment recommendations
- Patient monitoring in any capacity
- Any decision affecting patient care

### 9.2 Bias and Fairness

The synthetic training data does not model:
- Age-related physiological differences
- Sex-based signal variations
- Ethnic/racial population differences
- Effects of pre-existing conditions or medications

Real deployment would require careful stratification across demographic groups and subgroup performance analysis.

### 9.3 Failure Modes

| Scenario | Risk | Mitigation |
|----------|------|------------|
| False positive (HIGH risk when healthy) | Patient anxiety, unnecessary tests | Require clinical confirmation |
| False negative (LOW risk when at risk) | Missed intervention | Never use as sole diagnostic |
| Model drift (performance degrades) | Silent failure | Continuous monitoring required |
| Adversarial input | Unpredictable output | Input validation, uncertainty flags |

---

**End of Document**
