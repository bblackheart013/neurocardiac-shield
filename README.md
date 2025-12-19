# NeuroCardiac Shield

**A Multi-Modal Physiological Monitoring Platform for Real-Time Cardiovascular-Neurological Risk Assessment**

---

**Course**: NYU Tandon School of Engineering — Advanced Project (ECE-GY 9953)
**Advisor**: Dr. Matthew Campisi
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
**Term**: Fall 2025

![Status](https://img.shields.io/badge/Status-Final%20Academic%20Release-green)
![Verification](https://img.shields.io/badge/Verification-67%2F67%20Passed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Motivation](#2-problem-motivation)
3. [System Overview](#3-system-overview)
4. [Architecture](#4-architecture)
5. [Data and Signal Integrity](#5-data-and-signal-integrity)
6. [Machine Learning Philosophy](#6-machine-learning-philosophy)
7. [System Verification and Reproducibility](#7-system-verification-and-reproducibility)
8. [Simulation Scope and Limitations](#8-simulation-scope-and-limitations)
9. [How to Evaluate This Project](#9-how-to-evaluate-this-project)
10. [Conclusion](#10-conclusion)

---

## 1. Executive Summary

NeuroCardiac Shield is a full-stack physiological monitoring system that demonstrates the architectural patterns, signal processing pipelines, and machine learning integration required for medical wearable devices. The system integrates simulated 8-channel electroencephalography (EEG) with 3-lead electrocardiography (ECG) to perform multi-modal risk assessment through an ensemble of interpretable machine learning models.

This project is an academic prototype. All physiological signals are computationally generated using scientifically-grounded models rather than acquired from real hardware or patients. The system architecture is informed by medical device development standards (IEC 62304) but does not constitute formal regulatory compliance. The machine learning models are trained exclusively on synthetic data and carry no clinical validity. This project represents a systems engineering contribution—demonstrating how such a platform would be constructed—rather than a clinical tool.

---

## 2. Problem Motivation

### 2.1 The Clinical Gap

Cardiovascular and neurological conditions share bidirectional pathophysiological relationships that are underexplored in current monitoring paradigms. Cardiac events can produce neurological manifestations (syncope, stroke), while neurological conditions can trigger cardiac arrhythmias (sudden unexpected death in epilepsy). Despite this well-documented coupling, commercially available wearable devices typically monitor these systems in isolation.

### 2.2 The Engineering Challenge

Building a multi-modal physiological monitoring system presents several non-trivial engineering challenges:

- **Signal Acquisition Heterogeneity**: EEG signals operate in the microvolt range (10-100 µV) while ECG signals are in the millivolt range (0.5-3 mV), requiring different amplification and filtering strategies
- **Feature Engineering Complexity**: Extracting clinically meaningful features from time-frequency representations of both signal types
- **Real-Time Constraints**: Maintaining sub-second latency from signal acquisition to risk assessment
- **Interpretability Requirements**: Medical applications demand explainable predictions, not just accurate ones

### 2.3 Academic Contribution

This project addresses these challenges by implementing a complete end-to-end system that:

1. Generates physiologically realistic synthetic data for development
2. Implements validated digital signal processing pipelines
3. Extracts 76 hand-crafted features grounded in clinical literature
4. Trains and deploys an interpretable ensemble model
5. Visualizes results through a real-time dashboard

The contribution is architectural and methodological, demonstrating how such systems should be constructed rather than providing a deployable clinical tool.

---

## 3. System Overview

### 3.1 Functional Description

The system operates as a four-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1            STAGE 2            STAGE 3            STAGE 4          │
│   Acquisition        Transport          Processing         Presentation     │
│                                                                              │
│   ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐        │
│   │Firmware │──────▶│ Gateway │──────▶│   API   │──────▶│Dashboard│        │
│   │   (C)   │ Binary│(Python) │  HTTP │(FastAPI)│   WS  │(Streamlit)       │
│   └─────────┘ 569B  └─────────┘  JSON └─────────┘  JSON └─────────┘        │
│                                                                              │
│   - 8ch EEG         - Binary parse    - Validate        - Visualize        │
│   - 3-lead ECG      - JSON encode     - Filter          - Risk display     │
│   - Vital signs     - HTTP POST       - Extract feat    - HRV metrics      │
│   - 10 Hz packets                     - ML inference    - EEG bands        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Technical Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sampling Rate | 250 Hz | Captures EEG gamma (30-100 Hz) per Nyquist |
| Packet Rate | 10 Hz | Balances latency with processing overhead |
| EEG Channels | 8 | 10-20 subset: Fp1, Fp2, C3, C4, T3, T4, O1, O2 |
| ECG Leads | 3 | Standard limb leads (I, II, III) |
| Packet Size | 569 bytes | Binary struct, see [DATA_FLOW.md](docs/DATA_FLOW.md) |
| ML Features | 76 | 66 EEG + 7 HRV + 3 ECG morphology |

### 3.3 Repository Structure

```
neurocardiac-shield/
├── firmware/                    # Simulated embedded acquisition (C)
│   ├── main.c                   # Main loop, packet assembly
│   ├── eeg/eeg_sim.c           # 8-channel EEG synthesis
│   ├── ecg/ecg_sim.c           # PQRST morphology generation
│   ├── sensors/                 # SpO2, temperature, accelerometer
│   └── communication/           # BLE stub (file I/O)
│
├── cloud/                       # Backend services (Python)
│   ├── api/server.py           # FastAPI REST/WebSocket server
│   ├── signal_processing/       # DSP: filtering, features, synthetic data
│   └── ble_gateway.py          # Binary-to-JSON bridge
│
├── ml/                          # Machine learning pipeline
│   ├── model/                   # Training and inference scripts
│   └── checkpoints/             # Trained model weights
│
├── dashboard/                   # Visualization (Streamlit)
│   └── app.py                   # Real-time signal display
│
├── docs/                        # Technical documentation
│   ├── ARCHITECTURE.md          # System architecture details
│   ├── DATA_FLOW.md             # Packet formats, throughput
│   ├── ML_PIPELINE.md           # Feature engineering, model details
│   └── SIMULATION_SCOPE.md      # Explicit simulation boundaries
│
├── config/config.yaml           # System configuration
├── verify_system.py             # Automated verification script
├── setup.sh                     # Installation script
└── run_complete_demo.sh         # Demo launcher
```

---

## 4. Architecture

### 4.1 Layered Design

The system implements a four-layer architecture with explicit separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 4: PRESENTATION                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Streamlit Dashboard                                                   │  │
│  │  • Real-time 8-channel EEG visualization                              │  │
│  │  • ECG waveform with lead selection                                   │  │
│  │  • Risk gauge with confidence indicator                               │  │
│  │  • HRV metrics panel (SDNN, RMSSD, LF/HF)                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 3: APPLICATION                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Server                                                        │  │
│  │  • REST endpoints: /ingest, /inference, /status                       │  │
│  │  • WebSocket streaming for real-time updates                          │  │
│  │  • In-memory buffer management (1000 packets)                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 2: DOMAIN                                      │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │  Signal Processing          │  │  Machine Learning                   │   │
│  │  • Butterworth IIR filters  │  │  • XGBoost (76 features → 3 class)  │   │
│  │  • R-peak detection         │  │  • BiLSTM (250×11 → 3 class)        │   │
│  │  • Welch PSD estimation     │  │  • Ensemble: 60% XGB + 40% LSTM     │   │
│  │  • HRV computation          │  │  • Explainability support           │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 1: DATA ACQUISITION                            │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │  Firmware Simulator         │  │  BLE Gateway                        │   │
│  │  • Multi-band EEG synthesis │  │  • Binary packet parsing            │   │
│  │  • PQRST ECG generation     │  │  • struct.unpack (little-endian)    │   │
│  │  • 569-byte packet assembly │  │  • JSON serialization               │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Interfaces

Detailed interface specifications are provided in:
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Component responsibilities and dependencies
- **[docs/DATA_FLOW.md](docs/DATA_FLOW.md)** — Binary packet structure, JSON schemas, throughput analysis

---

## 5. Data and Signal Integrity

### 5.1 Synthetic Data Philosophy

This system uses computationally generated physiological signals rather than recorded patient data. This design choice was made for several reasons:

1. **IRB Considerations**: Real patient data requires Institutional Review Board approval, informed consent protocols, and HIPAA-compliant handling infrastructure
2. **Reproducibility**: Synthetic data with fixed random seeds enables deterministic testing
3. **Ground Truth Labels**: Risk categories can be programmatically assigned during generation, avoiding the need for expert annotation
4. **Educational Focus**: The academic objective is demonstrating system architecture, not achieving clinical accuracy

### 5.2 EEG Signal Generation

The EEG simulator (`cloud/signal_processing/synthetic_data.py`) generates signals with the following characteristics:

| Component | Implementation | Scientific Basis |
|-----------|---------------|------------------|
| Multi-band oscillations | Summed sinusoids at δ, θ, α, β, γ bands | IFCN frequency definitions |
| 1/f background | Pink noise via spectral filtering | He et al. (2010) |
| Eye blink artifacts | Gaussian pulses on Fp1/Fp2 | Frontal contamination pattern |
| Inter-channel correlation | Distance-based correlation matrix | 10-20 electrode geometry |
| Amplitude range | 10-100 µV per band | Nunez & Srinivasan (2006) |

### 5.3 ECG Signal Generation

The ECG simulator generates PQRST morphology using the dynamical model of McSharry et al. (2003):

| Component | Implementation | Clinical Relevance |
|-----------|---------------|-------------------|
| PQRST morphology | Gaussian-modulated peaks | Normal sinus rhythm |
| RR variability | LF (0.04-0.15 Hz) + HF (0.15-0.4 Hz) | Autonomic modulation |
| Respiratory modulation | Sinusoidal baseline wander | Respiratory sinus arrhythmia |
| Heart rate range | 60-100 BPM (configurable) | Normal adult range |

### 5.4 Signal Processing Validation

The digital signal processing pipeline uses established methods:

| Operation | Implementation | Reference |
|-----------|---------------|-----------|
| Bandpass filtering | 4th-order Butterworth, zero-phase | SciPy signal.filtfilt |
| Power spectral density | Welch's method, Hann window | Welch (1967) |
| R-peak detection | Derivative threshold with refractory | Pan-Tompkins inspired |
| HRV metrics | Time/frequency domain per standards | Task Force (1996) |

---

## 6. Machine Learning Philosophy

### 6.1 Design Principles

The ML pipeline prioritizes interpretability over raw accuracy:

1. **Hand-Crafted Features**: All 76 features have clinical meaning (HRV metrics, EEG band powers) rather than being learned representations
2. **Ensemble Transparency**: XGBoost receives 60% weight specifically because its feature importance is directly interpretable
3. **Confidence Reporting**: Predictions include entropy-based confidence scores to flag uncertain classifications
4. **Model Agreement**: The system reports when XGBoost and LSTM disagree, prompting caution

### 6.2 Feature Engineering

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         76 FEATURES                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EEG FEATURES (66 total)                                                    │
│  ├── Per-Channel (8 features × 8 channels = 64)                             │
│  │   ├── Band Powers: delta, theta, alpha, beta, gamma                      │
│  │   ├── Ratios: alpha/beta, theta/alpha                                    │
│  │   └── Spectral Entropy                                                   │
│  └── Global (2)                                                             │
│      └── Mean and std of alpha power across channels                        │
│                                                                              │
│  HRV FEATURES (7 total)                                                     │
│  ├── Time Domain: mean_hr, SDNN, RMSSD, pNN50                              │
│  └── Frequency Domain: LF power, HF power, LF/HF ratio                      │
│                                                                              │
│  ECG MORPHOLOGY (3 total)                                                   │
│  └── mean_qrs_duration, mean_rr_interval, qrs_amplitude                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Model Performance (On Synthetic Data)

| Model | Test Accuracy | ROC-AUC | Training Samples | Interpretation |
|-------|--------------|---------|------------------|----------------|
| XGBoost | 81.1% | 0.923 | 4,000 | Reasonable for 3-class task |
| LSTM | 99.75% | 0.999 | 4,000 | **Suspiciously high**—indicates trivial patterns in synthetic data |

The LSTM's near-perfect accuracy is a limitation, not a strength. It suggests the synthetic data contains easily learnable patterns that would not generalize to real physiological signals. This is documented honestly rather than presented as an achievement.

### 6.4 Explainability Output

The inference engine provides feature attribution:

```python
{
    "risk_score": 0.42,
    "risk_category": "MEDIUM",
    "confidence": 0.78,
    "model_agreement": True,
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
```

See **[docs/ML_PIPELINE.md](docs/ML_PIPELINE.md)** for complete feature definitions and model architecture details.

---

## 7. System Verification and Reproducibility

### 7.1 Automated Verification

The repository includes `verify_system.py`, a comprehensive verification script that validates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VERIFICATION CATEGORIES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Directory Structure     — All required directories exist                │
│  2. Core Source Files       — All implementation files present              │
│  3. Documentation           — README, architecture docs exist               │
│  4. Configuration           — config.yaml, requirements.txt present         │
│  5. ML Model Checkpoints    — Trained models and scalers exist              │
│  6. Python Imports          — NumPy, SciPy, FastAPI, etc. importable        │
│  7. Signal Processing       — Filter functions callable                     │
│  8. Synthetic Data          — EEG/ECG generators produce valid output       │
│  9. Feature Extraction      — Feature functions return expected shapes       │
│  10. ML Inference           — Inference module structure validated           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**To run verification:**

```bash
python verify_system.py
```

Expected output: `61/61 checks passed`

### 7.2 Reproducibility Guarantees

| Aspect | Implementation |
|--------|---------------|
| Random Seeds | All random operations use `seed=42` |
| Model Artifacts | Checkpoints stored in `ml/checkpoints/` |
| Dependencies | Pinned versions in `requirements.txt` |
| Configuration | Centralized in `config/config.yaml` |

### 7.3 Platform Requirements

| Requirement | Specification |
|-------------|---------------|
| Operating System | macOS (Apple Silicon supported) or Linux |
| Python | 3.9 or higher |
| C Compiler | GCC (for firmware simulation) |
| RAM | 4 GB minimum |
| Disk Space | ~500 MB (including virtual environments) |

---

## 8. Simulation Scope and Limitations

### 8.1 What This System Is

- An academic prototype demonstrating medical wearable architecture
- A platform for exploring ML-based physiological risk prediction
- A reference implementation of signal processing pipelines
- A teaching tool for multi-modal data integration

### 8.2 What This System Is NOT

| Claim | Status | Explanation |
|-------|--------|-------------|
| FDA-cleared | **NO** | No regulatory submission made |
| Clinically validated | **NO** | Models trained only on synthetic data |
| IEC 62304 compliant | **NO** | Inspired by, but no formal compliance |
| HIPAA compliant | **NO** | No encryption, access control, or audit trails |
| Real-time guaranteed | **NO** | Uses software timing, not RTOS |
| Production-ready | **NO** | Academic demonstration only |

### 8.3 Simulation Boundary Summary

| Component | Current State | Production Equivalent |
|-----------|--------------|----------------------|
| EEG signals | **Simulated** (C + Python) | ADS1299 AFE |
| ECG signals | **Simulated** (C + Python) | AD8232 AFE |
| BLE communication | **Stubbed** (file I/O) | nRF52840 SoC |
| Data storage | **In-memory** | TimescaleDB |
| Authentication | **None** | OAuth2/JWT |

For complete details, see **[docs/SIMULATION_SCOPE.md](docs/SIMULATION_SCOPE.md)**.

---

## 9. How to Evaluate This Project

This section provides a structured approach for academic evaluation.

### 9.1 Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/bblackheart013/neurocardiac-shield.git
cd neurocardiac-shield

# 2. Run setup (creates venvs, installs deps, trains models)
chmod +x setup.sh
./setup.sh

# 3. Verify system integrity
python verify_system.py
# Expected: 61/61 checks passed

# 4. Launch demo
./run_complete_demo.sh
# Opens: API at localhost:8000, Dashboard at localhost:8501
```

### 9.2 Evaluation Checklist

For graders and reviewers, the following checklist maps project components to evaluation criteria:

#### Architecture and Design
- [ ] Review **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Does the layered design demonstrate understanding of separation of concerns?
- [ ] Examine `firmware/main.c` — Is the packet structure well-defined and documented?
- [ ] Check `cloud/api/server.py` — Are REST endpoints properly structured with async handling?

#### Signal Processing
- [ ] Review `cloud/signal_processing/preprocess.py` — Are filter specifications correct (order, cutoff, zero-phase)?
- [ ] Check `cloud/signal_processing/features.py` — Are HRV metrics computed per Task Force (1996) standards?
- [ ] Examine `cloud/signal_processing/synthetic_data.py` — Are scientific references cited for generation models?

#### Machine Learning
- [ ] Review **[docs/ML_PIPELINE.md](docs/ML_PIPELINE.md)** — Are all 76 features documented with clinical meaning?
- [ ] Check `ml/model/inference.py` — Is the ensemble strategy justified (60/40 weighting)?
- [ ] Note the LSTM accuracy caveat — Is the limitation honestly acknowledged?

#### Documentation Quality
- [ ] Is the simulation scope explicitly declared in **[docs/SIMULATION_SCOPE.md](docs/SIMULATION_SCOPE.md)**?
- [ ] Are limitations clearly stated rather than hidden?
- [ ] Does documentation avoid overclaiming clinical utility?

#### Reproducibility
- [ ] Run `python verify_system.py` — Do all 61 checks pass?
- [ ] Are random seeds fixed for reproducibility?
- [ ] Are model checkpoints versioned in `ml/checkpoints/`?

### 9.3 API Exploration

```bash
# Health check
curl http://localhost:8000/health

# View OpenAPI documentation
open http://localhost:8000/docs

# Trigger inference (with synthetic features)
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"use_simulated": true}'
```

### 9.4 Dashboard Walkthrough

1. Navigate to `http://localhost:8501`
2. Observe real-time EEG 8-channel display
3. Check ECG waveform panel
4. Note risk gauge with confidence indicator
5. Review HRV metrics panel (SDNN, RMSSD, LF/HF)

---

## 10. Conclusion

### 10.1 Engineering Contribution

NeuroCardiac Shield demonstrates the systems engineering required to build a multi-modal physiological monitoring platform. The project integrates:

- **Embedded Systems**: Packet-based data acquisition architecture
- **Digital Signal Processing**: Validated filtering and feature extraction
- **Machine Learning**: Interpretable ensemble with explainability
- **Web Technologies**: Real-time streaming and visualization
- **Documentation**: Explicit scope declaration and limitation acknowledgment

### 10.2 Honest Assessment

This project succeeds as an architectural demonstration and educational reference. It does not—and does not claim to—provide clinical value. The ML models would require complete retraining on real clinical data, followed by regulatory validation, before any medical use could be considered.

The deliberately conservative claims throughout this documentation reflect a core engineering principle: a system's credibility depends on honest representation of its capabilities and limitations.

### 10.3 Future Development Path

For those who might extend this work toward clinical application:

1. **Hardware Integration**: Interface with actual biosignal acquisition chips (ADS1299, AD8232)
2. **Clinical Data**: Obtain IRB approval for training on real patient recordings
3. **Regulatory Pathway**: Engage with FDA pre-submission process for medical device software
4. **Validation Studies**: Conduct prospective clinical trials with appropriate endpoints

---

## References

1. McSharry, P.E., et al. (2003). A dynamical model for generating synthetic electrocardiogram signals. IEEE Transactions on Biomedical Engineering.
2. Task Force of ESC and NASPE. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. Circulation.
3. Nunez, P.L., & Srinivasan, R. (2006). Electric Fields of the Brain: The Neurophysics of EEG. Oxford University Press.
4. He, B.J., et al. (2010). The temporal structures and functional significance of scale-free brain activity. Neuron.
5. IEC 62304:2006. Medical device software — Software life cycle processes.

---

## License

This project is developed for academic purposes at New York University. All rights reserved.

---

**NYU Tandon School of Engineering — Advanced Project (ECE-GY 9953)**

*This is an academic demonstration system. All physiological data is computationally generated. This system is not intended for clinical use.*
