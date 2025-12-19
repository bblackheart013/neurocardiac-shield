# NeuroCardiac Shield — System Architecture

**Document Type**: Technical Architecture Specification
**Version**: 2.1.0
**Last Updated**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav Devram Chandgir

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Context](#2-system-context)
3. [Architectural Layers](#3-architectural-layers)
4. [Component Specifications](#4-component-specifications)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Security Considerations](#8-security-considerations)
9. [Known Limitations](#9-known-limitations)

---

## 1. Executive Summary

NeuroCardiac Shield implements a four-layer architecture for multi-modal physiological monitoring. The system processes simulated EEG (8 channels) and ECG (3 leads) signals through a pipeline that spans from embedded acquisition to web-based visualization.

> **Faculty Note — Why This Architecture?**
>
> The layered architecture mirrors patterns used in production medical device systems while remaining tractable for academic implementation. Each layer has a single responsibility: acquisition, transport, processing, and presentation. This separation enables independent testing and future replacement of individual components (e.g., swapping simulated firmware for real hardware) without restructuring the entire system.

### Scope Statement

| This System Is | This System Is NOT |
|----------------|-------------------|
| Academic prototype for NYU ECE-GY 9953 | FDA-cleared for clinical use |
| Demonstration of medical wearable architecture | Validated against real patient data |
| Platform for exploring ML-based risk prediction | Suitable for medical diagnosis |
| Educational reference implementation | A replacement for professional medical care |

---

## 2. System Context

The following diagram shows NeuroCardiac Shield within its operational context:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM CONTEXT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────────────────┐                    │
│                          │                             │                    │
│    ┌──────────────┐      │    NEUROCARDIAC SHIELD      │      ┌──────────┐ │
│    │  SIMULATED   │      │    ═══════════════════      │      │   WEB    │ │
│    │   WEARABLE   │─────▶│                             │─────▶│ BROWSER  │ │
│    │   DEVICE     │      │  ┌───────┐    ┌───────┐    │      │          │ │
│    │              │      │  │  API  │───▶│  ML   │    │      └──────────┘ │
│    │  • 8ch EEG   │      │  │Server │    │Engine │    │                    │
│    │  • 3-lead ECG│      │  └───────┘    └───────┘    │                    │
│    │  • Vitals    │      │       │                     │                    │
│    └──────────────┘      │       ▼                     │                    │
│                          │  ┌───────────┐              │                    │
│                          │  │ Dashboard │              │                    │
│                          │  └───────────┘              │                    │
│                          │                             │                    │
│                          └─────────────────────────────┘                    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  IMPORTANT: All physiological data is SIMULATED.                    │   │
│   │  No real patient data is acquired, processed, or stored.            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architectural Layers

### 3.1 Four-Layer Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  LAYER 4 ─────────────────────────────────────────────────── PRESENTATION   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │   STREAMLIT DASHBOARD (dashboard/app.py)                              │  │
│  │   ────────────────────────────────────────                            │  │
│  │   • Real-time 8-channel EEG waveform display                          │  │
│  │   • ECG trace with selectable leads (I, II, III)                      │  │
│  │   • Risk gauge: LOW / MEDIUM / HIGH with confidence %                 │  │
│  │   • HRV metrics panel: HR, SDNN, RMSSD, pNN50, LF/HF                 │  │
│  │   • EEG band power bar chart (δ, θ, α, β, γ)                          │  │
│  │   • Auto-refresh at configurable interval                             │  │
│  │                                                                        │  │
│  │   Technology: Streamlit 1.28+, Plotly 5.18+                           │  │
│  │   Port: 8501                                                           │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      │ HTTP GET, WebSocket                   │
│                                      ▼                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 ─────────────────────────────────────────────────── APPLICATION    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │   FASTAPI SERVER (cloud/api/server.py)                                │  │
│  │   ─────────────────────────────────────                               │  │
│  │                                                                        │  │
│  │   REST Endpoints:                                                      │  │
│  │   ┌─────────────────────┬─────────┬───────────────────────────────┐   │  │
│  │   │ Endpoint            │ Method  │ Purpose                       │   │  │
│  │   ├─────────────────────┼─────────┼───────────────────────────────┤   │  │
│  │   │ /health             │ GET     │ Service health check          │   │  │
│  │   │ /api/v1/ingest      │ POST    │ Receive physiological packets │   │  │
│  │   │ /api/v1/device/{id} │ GET     │ Device connection status      │   │  │
│  │   │ /api/v1/inference   │ POST    │ Trigger ML prediction         │   │  │
│  │   │ /ws/stream          │ WS      │ Real-time data streaming      │   │  │
│  │   └─────────────────────┴─────────┴───────────────────────────────┘   │  │
│  │                                                                        │  │
│  │   Buffer: In-memory, 1000 packets (~100 seconds)                      │  │
│  │   Technology: FastAPI 0.104+, Uvicorn, Pydantic                       │  │
│  │   Port: 8000                                                           │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      │ Function calls                        │
│                                      ▼                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 ────────────────────────────────────────────────────────── DOMAIN  │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │                                │  │                                    │ │
│  │   SIGNAL PROCESSING            │  │   MACHINE LEARNING                 │ │
│  │   (cloud/signal_processing/)   │  │   (ml/model/)                      │ │
│  │   ─────────────────────────    │  │   ────────────────                 │ │
│  │                                │  │                                    │ │
│  │   preprocess.py:               │  │   XGBoost Classifier:              │ │
│  │   • filter_eeg(): 0.5-50 Hz    │  │   • 76 features → 3 classes        │ │
│  │   • filter_ecg(): 0.5-40 Hz    │  │   • 200 trees, depth 6             │ │
│  │   • detect_r_peaks()           │  │   • Feature importance available   │ │
│  │                                │  │                                    │ │
│  │   features.py:                 │  │   BiLSTM Network:                  │ │
│  │   • extract_eeg_features()     │  │   • 250×11 → 3 classes             │ │
│  │   • extract_hrv_features()     │  │   • 128→64 units, bidirectional   │ │
│  │   • compute_band_power()       │  │   • Dropout 0.3                    │ │
│  │                                │  │                                    │ │
│  │   synthetic_data.py:           │  │   inference.py:                    │ │
│  │   • EEGGenerator class         │  │   • NeuroCardiacInference class    │ │
│  │   • ECGGenerator class         │  │   • Ensemble: 60% XGB + 40% LSTM   │ │
│  │                                │  │   • Explainability support         │ │
│  │                                │  │                                    │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                                      │                                       │
│                                      │ Binary data                           │
│                                      ▼                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 ────────────────────────────────────────────── DATA ACQUISITION    │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │                                │  │                                    │ │
│  │   FIRMWARE SIMULATOR           │  │   BLE GATEWAY                      │ │
│  │   (firmware/)                  │  │   (cloud/ble_gateway.py)           │ │
│  │   ────────────────────         │  │   ──────────────────               │ │
│  │                                │  │                                    │ │
│  │   main.c:                      │  │   Functions:                       │ │
│  │   • Main acquisition loop      │  │   • Monitor /tmp/ble_data.bin      │ │
│  │   • 569-byte packet assembly   │  │   • Parse binary with struct       │ │
│  │   • 10 Hz output rate          │  │   • Convert to JSON                │ │
│  │                                │  │   • POST to /api/v1/ingest         │ │
│  │   eeg/eeg_sim.c:               │  │                                    │ │
│  │   • 8-channel synthesis        │  │   Polling: 100ms interval          │ │
│  │   • Multi-band oscillations    │  │                                    │ │
│  │                                │  │   Limitation:                      │ │
│  │   ecg/ecg_sim.c:               │  │   File-based IPC (not real BLE)    │ │
│  │   • PQRST morphology           │  │                                    │ │
│  │   • HRV modulation             │  │                                    │ │
│  │                                │  │                                    │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Responsibilities

| Layer | Responsibility | Failure Isolation |
|-------|---------------|-------------------|
| Presentation | User interaction, visualization | Dashboard crash doesn't affect API |
| Application | Request routing, data buffering | API crash doesn't affect firmware |
| Domain | Business logic, signal processing, ML | Processing errors don't crash API |
| Acquisition | Data generation, transport | Firmware crash doesn't affect cloud |

> **Faculty Note — Separation of Concerns**
>
> Each layer depends only on the layer below it. The dashboard never directly calls the firmware; it only interacts through the API. This unidirectional dependency graph makes the system easier to reason about and test. In a production system, this would also enable horizontal scaling of the API layer independently of the dashboard.

---

## 4. Component Specifications

### 4.1 Firmware Simulator

**Location**: `firmware/`

**Purpose**: Simulates embedded data acquisition that would run on STM32 or similar MCU.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIRMWARE COMPONENT DIAGRAM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │  eeg_sim.c  │     │  ecg_sim.c  │     │  sensors/   │                   │
│   │             │     │             │     │             │                   │
│   │ 8 channels  │     │ 3 leads     │     │ SpO2, Temp  │                   │
│   │ 250 Hz      │     │ 250 Hz      │     │ Accel       │                   │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│          │                   │                   │                           │
│          └───────────────────┼───────────────────┘                           │
│                              │                                               │
│                              ▼                                               │
│                    ┌─────────────────┐                                       │
│                    │     main.c      │                                       │
│                    │                 │                                       │
│                    │ • Collect data  │                                       │
│                    │ • Pack struct   │                                       │
│                    │ • Write binary  │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │  ble_stub.c     │                                       │
│                    │                 │                                       │
│                    │ Write to file:  │                                       │
│                    │ /tmp/ble_data   │                                       │
│                    └─────────────────┘                                       │
│                                                                              │
│   Output: 569-byte packets at 10 Hz                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Packet Structure** (569 bytes total):

```c
typedef struct __attribute__((packed)) {
    // Header (8 bytes)
    uint32_t timestamp_ms;              // 4 bytes - ms since boot
    uint16_t packet_id;                 // 2 bytes - sequence counter
    uint8_t  device_id;                 // 1 byte  - device identifier
    uint8_t  status_flags;              // 1 byte  - status bitmap

    // EEG (400 bytes)
    int16_t  eeg_data[8][25];           // 8 ch × 25 samples × 2 bytes

    // ECG (150 bytes)
    int16_t  ecg_data[3][25];           // 3 leads × 25 samples × 2 bytes

    // Vitals (10 bytes)
    uint8_t  spo2_percent;              // 1 byte
    int16_t  temperature_x10;           // 2 bytes (°C × 10)
    int16_t  accel_x_mg;                // 2 bytes
    int16_t  accel_y_mg;                // 2 bytes
    int16_t  accel_z_mg;                // 2 bytes

    // Integrity (2 bytes)
    uint16_t checksum;                  // CRC16-CCITT
} PhysiologicalDataPacket;
```

### 4.2 Signal Processing Pipeline

**Location**: `cloud/signal_processing/`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIGNAL PROCESSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   RAW DATA                           PROCESSED OUTPUT                        │
│   ────────                           ────────────────                        │
│                                                                              │
│   EEG: int16[8][25]                                                         │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                       │
│   │ Scale to µV     │  value_µV = raw / 10.0                                │
│   └────────┬────────┘                                                       │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │ Bandpass Filter │  0.5-50 Hz, 4th order Butterworth                     │
│   │ (Zero-phase)    │  scipy.signal.filtfilt()                              │
│   └────────┬────────┘                                                       │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │ Notch Filter    │  60 Hz, Q=30 (powerline rejection)                    │
│   └────────┬────────┘                                                       │
│            ▼                                                                 │
│   ┌─────────────────┐      ┌────────────────────────────────┐               │
│   │ Welch PSD       │─────▶│ 64 EEG features:               │               │
│   │ Band Integration│      │ • 5 band powers × 8 channels   │               │
│   └─────────────────┘      │ • 2 ratios × 8 channels        │               │
│                            │ • 1 entropy × 8 channels       │               │
│                            │ + 2 global features            │               │
│                            └────────────────────────────────┘               │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                              │
│   ECG: int16[3][25]                                                         │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                       │
│   │ Scale to mV     │  value_mV = raw / 1000.0                              │
│   └────────┬────────┘                                                       │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │ Bandpass Filter │  0.5-40 Hz, 4th order Butterworth                     │
│   └────────┬────────┘                                                       │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │ R-Peak Detect   │  Derivative threshold, 250ms refractory               │
│   └────────┬────────┘                                                       │
│            ▼                                                                 │
│   ┌─────────────────┐      ┌────────────────────────────────┐               │
│   │ RR Interval     │─────▶│ 10 ECG features:               │               │
│   │ Computation     │      │ • 7 HRV: SDNN, RMSSD, pNN50,   │               │
│   └─────────────────┘      │         LF, HF, LF/HF, mean_hr │               │
│                            │ • 3 morphology: QRS dur, RR,   │               │
│                            │   amplitude                     │               │
│                            └────────────────────────────────┘               │
│                                                                              │
│   TOTAL: 76 FEATURES                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Machine Learning Ensemble

**Location**: `ml/model/`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ML ENSEMBLE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌───────────────────────────────────┐                    │
│                    │        INPUT PROCESSING           │                    │
│                    └───────────────────────────────────┘                    │
│                                    │                                         │
│              ┌─────────────────────┴─────────────────────┐                  │
│              │                                           │                  │
│              ▼                                           ▼                  │
│   ┌─────────────────────┐                   ┌─────────────────────┐        │
│   │                     │                   │                     │        │
│   │  FEATURE VECTOR     │                   │  TIME SEQUENCE      │        │
│   │  ─────────────────  │                   │  ─────────────      │        │
│   │                     │                   │                     │        │
│   │  76 dimensions      │                   │  Shape: (250, 11)   │        │
│   │  StandardScaler     │                   │  StandardScaler     │        │
│   │                     │                   │                     │        │
│   └──────────┬──────────┘                   └──────────┬──────────┘        │
│              │                                         │                    │
│              ▼                                         ▼                    │
│   ┌─────────────────────┐                   ┌─────────────────────┐        │
│   │                     │                   │                     │        │
│   │      XGBOOST        │                   │      BI-LSTM        │        │
│   │      ────────       │                   │      ───────        │        │
│   │                     │                   │                     │        │
│   │  • 200 trees        │                   │  • BatchNorm        │        │
│   │  • max_depth=6      │                   │  • BiLSTM(128)      │        │
│   │  • lr=0.05          │                   │  • Dropout(0.3)     │        │
│   │  • L1+L2 reg        │                   │  • BiLSTM(64)       │        │
│   │                     │                   │  • Dense(32)        │        │
│   │  Interpretable      │                   │  • Softmax(3)       │        │
│   │  Feature Importance │                   │                     │        │
│   │                     │                   │  Temporal Patterns  │        │
│   └──────────┬──────────┘                   └──────────┬──────────┘        │
│              │                                         │                    │
│              │    [P(LOW), P(MED), P(HIGH)]           │                    │
│              │                                         │                    │
│              └──────────────────┬──────────────────────┘                    │
│                                 │                                            │
│                                 ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │                       │                                │
│                    │   WEIGHTED ENSEMBLE   │                                │
│                    │   ──────────────────  │                                │
│                    │                       │                                │
│                    │   P = 0.6×XGB + 0.4×LSTM                              │
│                    │                       │                                │
│                    │   Why 60/40?          │                                │
│                    │   XGB: interpretable  │                                │
│                    │   LSTM: temporal      │                                │
│                    │                       │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │                       │                                │
│                    │   OUTPUT              │                                │
│                    │   ──────              │                                │
│                    │                       │                                │
│                    │   • risk_score: 0-1   │                                │
│                    │   • category: L/M/H   │                                │
│                    │   • confidence: 0-1   │                                │
│                    │   • explanations      │                                │
│                    │   • model_agreement   │                                │
│                    │                       │                                │
│                    └───────────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Faculty Note — Why 60/40 Ensemble Weighting?**
>
> The ensemble weights are intentionally skewed toward XGBoost (60%) despite LSTM achieving higher accuracy on synthetic data. This reflects a design philosophy that prioritizes interpretability in medical contexts. XGBoost provides feature importance scores that can explain predictions ("elevated LF/HF ratio contributed most to HIGH risk classification"), whereas LSTM predictions are effectively black-box. In a clinical setting, explainability enables physician oversight and builds trust in the system's recommendations.

---

## 5. Data Flow Architecture

### 5.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1              STAGE 2              STAGE 3              STAGE 4    │
│   Acquisition          Transport            Processing           Display    │
│                                                                              │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐ │
│   │ Firmware │        │ Gateway  │        │   API    │        │Dashboard │ │
│   │   (C)    │        │ (Python) │        │(FastAPI) │        │(Streamlit)│ │
│   └────┬─────┘        └────┬─────┘        └────┬─────┘        └────┬─────┘ │
│        │                   │                   │                   │        │
│        │   Binary          │   HTTP/JSON       │   WebSocket       │        │
│        │   569 bytes       │   ~2 KB           │   JSON            │        │
│        │   10 Hz           │   10 Hz           │   10 Hz           │        │
│        │                   │                   │                   │        │
│        ▼                   ▼                   ▼                   ▼        │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐ │
│   │ Generate │        │ Read     │        │ Validate │        │ Render   │ │
│   │ 8ch EEG  │        │ binary   │        │ Pydantic │        │ Plotly   │ │
│   │ 3-lead   │        │ from     │        │ schema   │        │ charts   │ │
│   │ ECG      │        │ /tmp/    │        │          │        │          │ │
│   └────┬─────┘        └────┬─────┘        └────┬─────┘        └────┬─────┘ │
│        │                   │                   │                   │        │
│        ▼                   ▼                   ▼                   ▼        │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐ │
│   │ Pack     │        │ struct   │        │ Store in │        │ Update   │ │
│   │ struct   │        │ unpack   │        │ buffer   │        │ display  │ │
│   │ 569 B    │        │ to dict  │        │ (1000)   │        │ at 1 Hz  │ │
│   └────┬─────┘        └────┬─────┘        └────┬─────┘        └──────────┘ │
│        │                   │                   │                            │
│        ▼                   ▼                   ▼                            │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐                     │
│   │ Write to │        │ JSON     │        │ Extract  │                     │
│   │ /tmp/    │        │ encode   │        │ features │                     │
│   │ file     │        │          │        │ (76)     │                     │
│   └──────────┘        └────┬─────┘        └────┬─────┘                     │
│                            │                   │                            │
│                            ▼                   ▼                            │
│                       ┌──────────┐        ┌──────────┐                     │
│                       │ POST to  │        │ Run ML   │                     │
│                       │ /ingest  │        │ ensemble │                     │
│                       └──────────┘        └────┬─────┘                     │
│                                                │                            │
│                                                ▼                            │
│                                           ┌──────────┐                     │
│                                           │ Return   │                     │
│                                           │ risk +   │                     │
│                                           │ explain  │                     │
│                                           └──────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Throughput Analysis

| Stage | Data Size | Rate | Bandwidth |
|-------|-----------|------|-----------|
| Firmware → File | 569 bytes | 10 Hz | 5.69 KB/s |
| File → Gateway | 569 bytes | 10 Hz | 5.69 KB/s |
| Gateway → API | ~2 KB JSON | 10 Hz | ~20 KB/s |
| API → Dashboard | ~2 KB JSON | 10 Hz | ~20 KB/s |

**Sample Rate Breakdown**:
- Sampling: 250 Hz
- Samples per packet: 25
- Packet rate: 250 / 25 = 10 Hz
- Signal data per second: (8×250×2) + (3×250×2) = 5,500 bytes

---

## 6. Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TECHNOLOGY STACK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Layer          Technology          Version      Purpose                   │
│   ─────          ──────────          ───────      ───────                   │
│                                                                              │
│   Firmware       C (GCC)             C11          Signal simulation         │
│                                                                              │
│   Gateway        Python              3.9+         Protocol bridge           │
│                  requests            2.31+        HTTP client               │
│                                                                              │
│   API            FastAPI             0.104+       REST/WebSocket server     │
│                  Uvicorn             0.24+        ASGI server               │
│                  Pydantic            2.5+         Schema validation         │
│                                                                              │
│   Processing     NumPy               1.24+        Array operations          │
│                  SciPy               1.11+        DSP (filters, Welch)      │
│                  Pandas              2.1+         Data manipulation         │
│                                                                              │
│   ML Training    XGBoost             2.0+         Gradient boosting         │
│                  TensorFlow/Keras    2.15+        Deep learning (LSTM)      │
│                  scikit-learn        1.3+         Preprocessing, metrics    │
│                                                                              │
│   Dashboard      Streamlit           1.28+        Web application           │
│                  Plotly              5.18+        Interactive charts        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Deployment Architecture

### 7.1 Development Mode (Implemented)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LOCAL DEVELOPMENT DEPLOYMENT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           LOCAL MACHINE                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐    │   │
│   │   │ Firmware  │   │  Gateway  │   │    API    │   │ Dashboard │    │   │
│   │   │  Process  │   │  Process  │   │  :8000    │   │  :8501    │    │   │
│   │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘    │   │
│   │         │               │               │               │           │   │
│   │         │    file       │    HTTP       │    HTTP       │           │   │
│   │         └───────────────┴───────────────┴───────────────┘           │   │
│   │                         │                                            │   │
│   │                    /tmp/neurocardiac_ble_data.bin                   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Launch: ./run_complete_demo.sh                                            │
│   Stop:   CTRL+C                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Production Mode (Conceptual — Not Implemented)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION DEPLOYMENT (CONCEPTUAL)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐                                                        │
│   │   WEARABLE     │         This diagram shows where the                   │
│   │   DEVICE       │         system WOULD go, not where it IS.             │
│   │   ──────────   │                                                        │
│   │   • STM32 MCU  │                                                        │
│   │   • ADS1299    │         Current implementation stops at                │
│   │   • nRF52840   │         simulated firmware and file I/O.               │
│   └───────┬────────┘                                                        │
│           │ BLE 5.0                                                         │
│           ▼                                                                  │
│   ┌────────────────┐                                                        │
│   │  EDGE GATEWAY  │                                                        │
│   │  ────────────  │                                                        │
│   │  • Phone app   │                                                        │
│   │  • RPi bridge  │                                                        │
│   └───────┬────────┘                                                        │
│           │ HTTPS/WSS                                                       │
│           ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                        CLOUD PLATFORM                               │   │
│   │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │   │
│   │   │   Load   │   │   API    │   │  ML      │   │ Dashboard│       │   │
│   │   │ Balancer │──▶│ Cluster  │──▶│ Workers  │   │ (CDN)    │       │   │
│   │   └──────────┘   └──────────┘   └──────────┘   └──────────┘       │   │
│   │                        │                                            │   │
│   │                        ▼                                            │   │
│   │                  ┌──────────┐                                       │   │
│   │                  │TimescaleDB                                       │   │
│   │                  │(Persistent)                                      │   │
│   │                  └──────────┘                                       │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Status: NOT IMPLEMENTED — shown for architectural completeness            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Security Considerations

### 8.1 Current State (Development)

| Aspect | Status | Notes |
|--------|--------|-------|
| Authentication | ❌ None | API is fully open |
| Authorization | ❌ None | No role-based access |
| Encryption (transit) | ❌ HTTP | No TLS configured |
| Encryption (rest) | ❌ None | In-memory only |
| Input Validation | ✅ Pydantic | Schema enforcement |
| Logging | ⚠️ Basic | Console output only |
| Rate Limiting | ❌ None | No protection |

> **Faculty Note — Why No Security Implementation?**
>
> The security features listed as "NOT IMPLEMENTED" are intentional scope exclusions for this academic project, not oversights. Implementing authentication, TLS, and audit logging would add complexity without furthering the educational objectives of the project (demonstrating signal processing and ML integration). In a production medical device, these would be mandatory requirements under HIPAA and FDA cybersecurity guidance (21 CFR Part 11).

### 8.2 Production Requirements (Not Implemented)

For clinical deployment, the following would be required:

- [ ] OAuth2/JWT authentication
- [ ] Role-based access control (clinician, patient, admin)
- [ ] TLS 1.3 for all connections
- [ ] AES-256 encryption at rest
- [ ] HIPAA-compliant audit logging
- [ ] Rate limiting and DDoS protection
- [ ] Penetration testing

---

## 9. Known Limitations

### 9.1 Technical Limitations

| Limitation | Impact | Mitigation Path |
|------------|--------|----------------|
| No real hardware | Cannot validate with real signals | Integrate ADS1299 + nRF52840 |
| In-memory storage | Data lost on restart | Add TimescaleDB persistence |
| Single-threaded gateway | Limited throughput | Use asyncio or multiprocessing |
| No authentication | Insecure for deployment | Implement OAuth2/JWT |
| File-based IPC | Not production-viable | Implement real BLE stack |

### 9.2 Scientific Limitations

| Limitation | Impact | Mitigation Path |
|------------|--------|----------------|
| Synthetic training data | No clinical validity | Obtain IRB-approved real data |
| Simplified HRV | May not match standards | Validate against PhysioNet |
| LSTM overfitting | 99.75% accuracy is suspicious | Retrain on diverse real data |
| No pathology modeling | Cannot detect abnormalities | Add arrhythmia/seizure patterns |

### 9.3 Regulatory Limitations

| Claim | Status | Required for Compliance |
|-------|--------|------------------------|
| IEC 62304 | Inspired by, not compliant | Full documentation, V&V, risk analysis |
| HIPAA | Not applicable (no PHI) | Encryption, access control, audit trails |
| FDA clearance | Not submitted | 510(k) or De Novo pathway |

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Nov 2025 | MSF | Initial architecture |
| 2.0.0 | Dec 2025 | MSF | Added limitations, security |
| 2.1.0 | Dec 2025 | MSF | Final submission version with diagrams |

---

**End of Document**
