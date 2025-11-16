# NeuroCardiac Shield

A multi-modal physiological monitoring platform integrating EEG and ECG signals for real-time cardiovascular-neurological risk assessment.

## Overview

NeuroCardiac Shield is a medical wearable system that combines brain activity monitoring (EEG) with cardiac signals (ECG) to provide comprehensive health assessment. The platform employs machine learning models for risk prediction and features a real-time visualization dashboard.

This project was developed as part of the NYU Advanced Project coursework, demonstrating end-to-end medical device software development with compliance to IEC 62304 standards.

## Features

- 8-channel EEG acquisition using the 10-20 electrode placement system (Fp1, Fp2, C3, C4, T3, T4, O1, O2)
- 3-lead ECG monitoring with PQRST morphology analysis
- Vital signs tracking: SpO2 saturation, body temperature, accelerometer data
- Heart Rate Variability (HRV) analysis: SDNN, RMSSD, pNN50, LF/HF ratio
- EEG frequency band decomposition: Delta, Theta, Alpha, Beta, Gamma
- Machine learning risk prediction using XGBoost and LSTM ensemble
- Real-time web dashboard with signal visualization and risk indicators
- RESTful API for data ingestion and inference

## Architecture

```
firmware/           Simulated embedded device (C)
    main.c                  Main acquisition loop
    eeg/                    EEG signal simulation
    ecg/                    ECG signal simulation
    sensors/                SpO2, temperature, accelerometer
    communication/          BLE transmission stub

cloud/              Backend services (Python)
    api/                    FastAPI server
    signal_processing/      Digital filters and feature extraction
    ble_gateway.py          Firmware-to-API bridge

ml/                 Machine learning pipeline
    model/                  Training and inference scripts
    checkpoints/            Trained model weights

dashboard/          Visualization interface
    app.py                  Streamlit dashboard

docs/               Technical documentation
config/             Configuration files
logs/               Runtime logs
```

## Technical Specifications

- Sampling rate: 250 Hz (Nyquist frequency for signals up to 125 Hz)
- Packet rate: 10 Hz (25 samples per packet)
- Packet size: 569 bytes per transmission
- EEG range: ±200 µV typical
- ECG range: ±3 mV typical
- Communication: BLE 5.0+ compatible (MTU 600 bytes)

## Requirements

- macOS (Apple Silicon supported) or Linux
- Python 3.9+
- GCC compiler
- 4GB RAM minimum

## Installation

Clone the repository:
```bash
git clone <repository-url>
cd NeuroCardiac-Shield
```

Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will create Python virtual environments, install all dependencies, and train the ML models.

## Usage

Start all system components:
```bash
./run_complete_demo.sh
```

This launches:
- API Server at http://localhost:8000
- Dashboard at http://localhost:8501
- Firmware simulator
- BLE Gateway bridge

Press CTRL+C to stop all components.

## API Endpoints

- GET /health - Service health check
- POST /api/v1/ingest - Receive physiological data packets
- GET /api/v1/device/{id}/status - Device connection status
- POST /api/v1/inference - Execute ML risk prediction

API documentation available at http://localhost:8000/docs

## Machine Learning Models

XGBoost Classifier
- Input: 74 hand-crafted features (HRV metrics, EEG band powers, spectral entropy)
- Output: 3-class risk prediction (LOW, MEDIUM, HIGH)

Bidirectional LSTM
- Input: Time-series sequences (250 timesteps x 11 channels)
- Architecture: Bidirectional LSTM layers (128 to 64 units)

Ensemble Strategy
- XGBoost weight: 60% (interpretable features)
- LSTM weight: 40% (temporal patterns)

## Compliance

This software follows medical device software development practices:
- IEC 62304 Class B design patterns
- HIPAA-aware data handling considerations
- GDPR-ready architecture

Note: This is a research prototype. Clinical deployment requires FDA clearance and full regulatory compliance.

## Authors

Mohd Sarfaraz Faiyaz - Author

Vaibhav Devram Chandgir - Contributor

## License

This project is developed for academic purposes at NYU. All rights reserved.

## Future Work

- Integration with actual hardware (STM32, nRF52840)
- Database persistence (TimescaleDB, InfluxDB)
- Real-time WebSocket streaming to dashboard
- CRC16 packet validation
- Heart rate calculation from ECG R-peaks
- Additional ML models (attention mechanisms, transformer architectures)
- Clinical validation studies
- FHIR HL7 interoperability

## Acknowledgments

- Signal processing based on scipy.signal and MNE-Python standards
- ECG algorithms adapted from Pan-Tompkins QRS detection
- Dashboard built with Streamlit and Plotly

---

NYU Advanced Project - Medical Device Software Development
