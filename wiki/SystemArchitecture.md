# System Architecture

## Overview

```
┌─────────────────────┐
│     Dashboard       │
│   (Streamlit UI)    │
└──────────┬──────────┘
           │ REST API
┌──────────▼──────────┐
│    Cloud Backend    │
│      (FastAPI)      │
├─────────┬───────────┤
│ Signal  │    ML     │
│Processing│ Inference │
└─────────┴───────────┘
           ▲
           │ HTTP POST
┌──────────┴──────────┐
│    BLE Gateway      │
│   (Python Script)   │
└──────────┬──────────┘
           │ Binary File
┌──────────▼──────────┐
│     Firmware        │
│   (C Simulator)     │
└─────────────────────┘
```

## Components

### Firmware Layer
- 8-channel EEG acquisition
- 3-lead ECG monitoring
- Vital signs (SpO2, Temperature, Accelerometer)
- 250 Hz sampling rate
- 10 Hz packet transmission

### Cloud Backend
- FastAPI async server
- Signal preprocessing (filters)
- Feature extraction
- Data buffering
- REST API endpoints

### ML Pipeline
- XGBoost (feature-based)
- LSTM (temporal patterns)
- Weighted ensemble (60/40)
- 3-class risk prediction

### Dashboard
- Real-time visualization
- EEG/ECG waveforms
- Risk indicators
- HRV metrics
- Band power charts

## Data Flow

1. Firmware generates signals
2. Gateway reads binary packets
3. API ingests JSON data
4. Features extracted
5. ML inference executed
6. Results displayed on dashboard

## Performance

- End-to-end latency: <1 second
- Packet rate: 10 Hz
- Inference time: ~80 ms
