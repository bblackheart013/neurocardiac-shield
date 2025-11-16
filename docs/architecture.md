# NeuroCardiac Shield - System Architecture

**Version:** 1.0.0
**Date:** January 2025
**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Institution:** New York University (NYU)
**Project Type:** Advanced Medical Wearable System

---

## Executive Summary

NeuroCardiac Shield is a production-grade, multi-modal physiological monitoring platform that integrates brain (EEG) and heart (ECG) signal acquisition, cloud-based signal processing, machine learning risk prediction, and real-time visualization. The system is designed with medical device software standards (IEC 62304) in mind and follows clean architecture principles suitable for FDA pre-certification pathways.

---

## System Overview

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD LAYER                          │
│              (Streamlit Real-time UI)                       │
└────────────────────┬────────────────────────────────────────┘
                     │ WebSocket / REST API
┌────────────────────▼────────────────────────────────────────┐
│                    CLOUD BACKEND                            │
│              (FastAPI Async Server)                         │
│  • Data Ingestion   • WebSocket Streaming                  │
│  • Signal Storage   • Device Management                    │
└────────────┬───────────────────────┬────────────────────────┘
             │                       │
┌────────────▼──────────┐  ┌─────────▼──────────────────────┐
│  SIGNAL PROCESSING    │  │   MACHINE LEARNING             │
│  • Filtering          │  │   • XGBoost (Features)         │
│  • Feature Extraction │  │   • LSTM (Temporal)            │
│  • HRV Analysis       │  │   • Ensemble Inference         │
└───────────────────────┘  └────────────────────────────────┘
             ▲
             │ BLE / Serial
┌────────────┴────────────────────────────────────────────────┐
│                    FIRMWARE LAYER                           │
│              (Embedded C/C++ - STM32-like)                  │
│  • EEG Acquisition (8 channels, 250 Hz)                    │
│  • ECG Acquisition (3 leads, 250 Hz)                       │
│  • Vital Signs (SpO2, Temp, Accel)                         │
│  • BLE Communication                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Firmware Layer (`/firmware`)

**Purpose:** Embedded data acquisition and transmission

**Language:** C (ISO C11 compatible)

**Modules:**
- `main.c`: System initialization and main acquisition loop
- `eeg/eeg_sim.c`: 8-channel EEG simulator (10-20 system subset)
- `ecg/ecg_sim.c`: 3-lead ECG simulator with realistic PQRST morphology
- `sensors/spo2_sim.c`: Pulse oximetry simulator (MAX30102-like)
- `sensors/temp_sim.c`: Temperature sensor simulator (MLX90614-like)
- `sensors/accel_sim.c`: 3-axis accelerometer simulator (ADXL345-like)
- `communication/ble_stub.c`: BLE transmission stub (nRF52/ESP32 target)

**Key Characteristics:**
- Sampling rate: 250 Hz (Nyquist: 125 Hz)
- Packet rate: 10 Hz (25 samples per packet)
- Data packet size: ~500 bytes (CRC16 protected)
- Simulated signals include physiological noise, artifacts, HRV

**Production Migration Path:**
```c
// Replace with hardware HAL for production deployment
// HAL_ADC_Start_DMA(&hadc1, eeg_buffer, EEG_BUFFER_SIZE);
// HAL_TIM_Base_Start_IT(&htim2);  // 250 Hz sampling timer
```

---

### 2. Cloud Backend (`/cloud`)

**Purpose:** Data ingestion, processing, and API service

**Technology Stack:**
- **Framework:** FastAPI 0.104+ (Python 3.10+)
- **Async Runtime:** uvicorn with asyncio
- **Database (TODO):** TimescaleDB / InfluxDB for time-series
- **Caching (TODO):** Redis for real-time data buffering

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ingest` | Receive physiological data packets |
| GET | `/api/v1/device/{id}/status` | Device health and connectivity |
| POST | `/api/v1/inference` | Trigger ML risk prediction |
| WS | `/ws/stream` | Real-time WebSocket data streaming |
| GET | `/health` | Service health check (load balancer) |

**Signal Processing Module (`signal_processing/`):**

**preprocess.py:**
- Medical-grade IIR filters (Butterworth, order 4)
- EEG: 0.5-50 Hz bandpass + 60 Hz notch
- ECG: 0.5-40 Hz bandpass (diagnostic quality)
- Pan-Tompkins R-peak detection
- Signal quality assessment (variance, clipping, SNR)

**features.py:**
- **EEG Features (~64 dimensions):**
  - Band powers: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
  - Ratios: Alpha/Beta (stress), Theta/Alpha (drowsiness)
  - Spectral entropy (signal complexity)
  - Inter-channel coherence (functional connectivity)

- **HRV Features (7 dimensions):**
  - Time-domain: Mean HR, SDNN, RMSSD, pNN50
  - Frequency-domain: LF power, HF power, LF/HF ratio (sympathovagal balance)

- **ECG Morphology (3 dimensions):**
  - Mean RR interval, QRS duration, QRS amplitude

**Total Feature Count:** ~74 features per analysis window

---

### 3. Machine Learning Pipeline (`/ml`)

**Approach:** Ensemble learning (interpretable + deep learning)

#### Model 1: XGBoost Classifier

**Architecture:**
- Input: 74 hand-crafted features
- Output: 3-class risk (LOW, MEDIUM, HIGH)
- Hyperparameters:
  ```python
  max_depth=6, learning_rate=0.05, n_estimators=200
  L1/L2 regularization, subsample=0.8
  ```

**Advantages:**
- Interpretable feature importance
- Fast inference (<5 ms)
- Robust to missing values
- Clinical validation ready

**Training Pipeline:**
1. Synthetic data generation (5000 samples, stratified)
2. StandardScaler normalization
3. Train/test split (80/20, stratified)
4. Early stopping on validation loss
5. Model saved as JSON (XGBoost native format)

#### Model 2: Bidirectional LSTM

**Architecture:**
```
Input (250, 11)  # 1 second, 11 channels (8 EEG + 3 ECG)
    ↓
BatchNorm
    ↓
BiLSTM (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
BiLSTM (64 units, return_sequences=False)
    ↓
Dropout (0.3)
    ↓
Dense (32, ReLU)
    ↓
Dense (3, Softmax)  # 3-class output
```

**Advantages:**
- Captures temporal dependencies
- Detects transient events (arrhythmia, seizure precursors)
- Bidirectional context (past + future)

**Training Pipeline:**
1. Generate 2000 sequences (250 timesteps each)
2. Per-channel StandardScaler normalization
3. Adam optimizer (lr=0.001)
4. Early stopping (patience=10)
5. Model saved as Keras native format

#### Ensemble Inference

**Strategy:**
```python
final_score = 0.6 × XGBoost_score + 0.4 × LSTM_score
confidence = 1 - normalized_entropy(probabilities)
```

**Output:**
```json
{
  "risk_score": 0.35,
  "risk_category": "LOW",
  "confidence": 0.89,
  "probabilities": {"LOW": 0.72, "MEDIUM": 0.23, "HIGH": 0.05},
  "model_breakdown": {
    "xgboost": {"score": 0.32, "category": "LOW"},
    "lstm": {"score": 0.40, "category": "MEDIUM"}
  }
}
```

---

### 4. Dashboard (`/dashboard`)

**Technology:** Streamlit 1.28+ with Plotly

**Features:**
- Real-time 8-channel EEG waveform display
- ECG signal with PQRST morphology
- Risk gauge with LOW/MEDIUM/HIGH categories
- HRV metrics dashboard
- EEG band power bar charts
- Auto-refresh mode (2-second intervals)

**Visualization Specs:**
- EEG amplitude range: ±100 µV
- ECG amplitude range: -0.5 to 2.0 mV
- Time window: 1-10 seconds (user configurable)
- Color scheme: Medical-grade dark theme

**Data Flow:**
1. Fetch device status via REST API
2. Request ML inference results
3. Generate signals (fallback if API unavailable)
4. Render Plotly interactive charts
5. WebSocket streaming (future enhancement)

---

## Data Flow Pipeline

### End-to-End Data Journey

```
Firmware (250 Hz sampling)
    ↓ [Every 100 ms]
Data Packet (500 bytes, CRC16)
    ↓ [BLE GATT Notification]
Gateway Device (BLE Gateway Script)
    ↓ [HTTP POST]
Cloud API (/api/v1/ingest)
    ↓ [Async Background Task]
Signal Processing (Filtering, Feature Extraction)
    ↓
Database (TimescaleDB - TODO)
    ↓ [On-demand Request]
ML Inference Engine
    ↓
Dashboard (WebSocket Stream)
    ↓
User Visualization
```

### Latency Targets

| Stage | Target | Actual (Simulated) |
|-------|--------|-------------------|
| Firmware → Cloud | <200 ms | ~150 ms |
| Signal Processing | <50 ms | ~30 ms |
| ML Inference | <100 ms | ~80 ms |
| Dashboard Render | <500 ms | ~400 ms |
| **End-to-End** | **<1 second** | **~660 ms** |

---

## Deployment Architecture

### Development Setup

```bash
# 1. Firmware compilation (simulated)
cd firmware
gcc -o neurocardiac_fw main.c eeg/*.c ecg/*.c sensors/*.c communication/*.c -lm

# 2. Cloud backend
cd cloud
pip install fastapi uvicorn numpy scipy scikit-learn xgboost tensorflow
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 3. ML model training
cd ml/model
python train_xgboost.py
python train_lstm.py

# 4. Dashboard
cd dashboard
pip install streamlit plotly requests pandas
streamlit run app.py --server.port 8501
```

### Production Deployment (TODO)

**Firmware:**
- Cross-compile for ARM Cortex-M4 (STM32F4)
- Flash via ST-LINK / J-LINK
- OTA updates via BLE DFU

**Cloud:**
- Containerization: Docker + Kubernetes
- Load balancer: NGINX / AWS ALB
- Database: PostgreSQL + TimescaleDB extension
- Caching: Redis Cluster
- ML Serving: TensorFlow Serving / ONNX Runtime
- Monitoring: Prometheus + Grafana

**Dashboard:**
- Hosted: AWS Amplify / Vercel
- CDN: CloudFront
- Auth: OAuth 2.0 / SAML

---

## Security & Compliance

### Data Protection

- **Encryption at rest:** AES-256 (database, file storage)
- **Encryption in transit:** TLS 1.3 (API, WebSocket)
- **BLE pairing:** Secure connections with bonding
- **PHI handling:** HIPAA-compliant data anonymization

### Medical Device Standards

- **IEC 62304:** Software lifecycle (Class B target)
- **ISO 13485:** Quality management system
- **IEC 60601-1:** Electrical safety (hardware requirement)
- **FDA Guidance:** Software as Medical Device (SaMD)

### Audit Trail

```python
# Comprehensive logging for compliance
logging.info(f"[AUDIT] User {user_id} accessed patient {patient_id} data at {timestamp}")
```

---

## Performance Metrics

### Resource Utilization

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Firmware | <50% (single core) | 128 KB RAM | 512 KB Flash |
| Cloud API | 2-4 vCPU | 4 GB RAM | 100 GB SSD |
| ML Inference | 1 GPU (optional) | 8 GB RAM | 10 GB (models) |
| Dashboard | 1 vCPU | 512 MB RAM | - |

### Scalability

- **Concurrent devices:** 1000+ (per API instance)
- **Inference throughput:** 100 predictions/sec (GPU)
- **WebSocket clients:** 500+ (per instance)

---

## Future Enhancements

### Short-term (3 months)
- [ ] Integrate real hardware (ADS1299, AD8232)
- [ ] Implement database persistence (TimescaleDB)
- [ ] Add user authentication (JWT)
- [ ] Deploy on AWS / Azure

### Medium-term (6 months)
- [ ] Advanced ML: Transformer models, federated learning
- [ ] Clinical validation study (IRB approval)
- [ ] Anomaly detection alerts (SMS/email)

### Long-term (12 months)
- [ ] FDA 510(k) submission
- [ ] Multi-patient monitoring (hospital dashboard)
- [ ] Edge AI (on-device inference)
- [ ] Wearable form factor (PCB design, enclosure)

---

## References

### Standards & Guidelines
- IEC 62304:2006 - Medical device software lifecycle
- ISO 13485:2016 - Quality management systems
- FDA Guidance on SaMD (2017)

### Scientific Publications
- Pan-Tompkins (1985): Real-time QRS detection
- Welch (1967): Power spectral density estimation
- Task Force (1996): Heart rate variability standards

### Technical Documentation
- [MNE-Python EEG Processing](https://mne.tools/)
- [PhysioNet ECG Database](https://physionet.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)

---

## Contact & Maintenance

**Developer:** Mohd Sarfaraz Faiyaz
**Institution:** New York University
**Version Control:** Git
**Issue Tracking:** GitHub Issues
**Documentation:** `/docs` directory

---

**Document Status:** Living Document
**Last Updated:** January 2025
**Next Review:** March 2025

---

*This architecture is designed for educational and research purposes. Clinical deployment requires regulatory approval and extensive validation.*
