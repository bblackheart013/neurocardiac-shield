# Changelog

All notable changes to the NeuroCardiac Shield project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2025-11-16

### Added

#### Firmware Layer
- 8-channel EEG signal simulator (Fp1, Fp2, C3, C4, T3, T4, O1, O2)
- 3-lead ECG waveform generator with PQRST morphology
- SpO2 pulse oximetry simulator
- Body temperature sensor simulator
- 3-axis accelerometer simulator
- BLE transmission stub for data output
- Packet assembly (569 bytes per transmission)
- 250 Hz sampling rate implementation

#### Cloud Backend
- FastAPI asynchronous server
- POST /api/v1/ingest endpoint for data ingestion
- GET /api/v1/device/{id}/status for device monitoring
- POST /api/v1/inference for ML risk prediction
- GET /health for service health checks
- WebSocket endpoint for real-time streaming
- CORS middleware configuration
- Pydantic data validation models

#### Signal Processing
- Butterworth bandpass filters (EEG: 0.5-50 Hz, ECG: 0.5-40 Hz)
- 60 Hz notch filter for powerline noise
- Pan-Tompkins R-peak detection algorithm
- EEG frequency band decomposition (Delta, Theta, Alpha, Beta, Gamma)
- HRV feature extraction (SDNN, RMSSD, pNN50, LF/HF ratio)
- Spectral entropy calculation
- Signal quality assessment

#### Machine Learning
- XGBoost classifier with 74 input features
- Bidirectional LSTM network (128→64 units)
- StandardScaler normalization for both models
- Weighted ensemble inference (60% XGBoost, 40% LSTM)
- 3-class risk prediction (LOW, MEDIUM, HIGH)
- Confidence scoring via entropy calculation
- Model serialization (JSON, Keras, pickle)

#### Dashboard
- Streamlit web application
- 8-channel EEG real-time visualization
- ECG waveform display
- Risk score gauge indicator
- HRV metrics panel
- EEG band power bar charts
- Auto-refresh functionality
- Plotly interactive charts

#### Infrastructure
- Setup script for environment configuration
- Demo launcher script
- BLE gateway bridge application
- Configuration file (YAML)
- Comprehensive .gitignore
- Makefile for compilation

#### Documentation
- README.md with project overview
- Architecture documentation
- Usage guide
- Firmware technical reference
- Cloud API reference
- ML pipeline documentation
- Dashboard user guide
- System validation report
- GitHub Wiki pages
- Release notes

### Changed
- N/A (Initial release)

### Deprecated
- N/A (Initial release)

### Removed
- N/A (Initial release)

### Fixed
- N/A (Initial release)

### Security
- Input validation via Pydantic models
- Configurable CORS policy
- No credentials in repository
- PHI-aware data handling design

---

## [Unreleased]

### Planned
- Database persistence (TimescaleDB/InfluxDB)
- JWT authentication
- CRC16 packet validation
- Real hardware integration
- WebSocket dashboard streaming
- Heart rate calculation from R-peaks
- Clinical validation studies

---

**Authors:** Mohd Sarfaraz Faiyaz, Vaibhav Devram Chandgir
**Institution:** New York University
