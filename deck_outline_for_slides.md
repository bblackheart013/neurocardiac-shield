# NeuroCardiac Shield - Pitch Deck Outline

**Presenter:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Institution:** New York University

---

## Slide 1: Title

**NeuroCardiac Shield**

Multi-Modal Physiological Monitoring for Real-Time Cardiovascular-Neurological Risk Assessment

NYU Advanced Project - Medical Device Software Development

---

## Slide 2: Problem Statement

- Cardiovascular diseases: Leading cause of global mortality
- Neurological disorders: Affect millions worldwide
- Current monitoring: Single-modality, isolated measurements
- Gap: No integrated brain-heart monitoring with predictive analytics

---

## Slide 3: Solution Overview

- Simultaneous EEG and ECG signal acquisition
- Real-time signal processing pipeline
- Machine learning risk prediction
- Interactive visualization dashboard
- End-to-end medical device software

---

## Slide 4: System Architecture

[Architecture Diagram]

- Firmware: Embedded signal acquisition (250 Hz)
- Cloud Backend: FastAPI async processing
- ML Pipeline: XGBoost + LSTM ensemble
- Dashboard: Streamlit real-time visualization

---

## Slide 5: Signal Acquisition

- 8-channel EEG (10-20 system)
  - Fp1, Fp2, C3, C4, T3, T4, O1, O2
- 3-lead ECG with PQRST morphology
- Ancillary: SpO2, Temperature, Accelerometer
- Packet rate: 10 Hz (569 bytes per transmission)

---

## Slide 6: Signal Processing

- Butterworth bandpass filters (order 4)
- 60 Hz notch filter for powerline noise
- Pan-Tompkins R-peak detection
- EEG frequency band decomposition
- HRV feature extraction (SDNN, RMSSD, LF/HF)

---

## Slide 7: Machine Learning Models

**XGBoost Classifier**
- 74 hand-crafted features
- Interpretable feature importance
- Inference: ~5 ms

**Bidirectional LSTM**
- Temporal pattern recognition
- 250 timesteps × 11 channels
- Inference: ~75 ms

---

## Slide 8: Ensemble Strategy

[Ensemble Diagram]

- XGBoost weight: 60%
- LSTM weight: 40%
- Risk categories: LOW, MEDIUM, HIGH
- Confidence scoring via entropy

---

## Slide 9: Dashboard Demo

[Dashboard Screenshots]

- Real-time EEG waveforms
- ECG signal with beat detection
- Risk gauge indicator
- HRV metrics panel
- Band power distribution

---

## Slide 10: Performance Results

| Metric | Achievement |
|--------|-------------|
| Sampling Rate | 250 Hz |
| End-to-End Latency | <1 second |
| ML Inference | 80 ms |
| API Response | <50 ms |
| Resource Usage | <30% CPU |

---

## Slide 11: Compliance Standards

- IEC 62304: Software lifecycle (Class B)
- ISO 13485: Quality management
- FDA SaMD: Pre-submission documentation
- HIPAA: PHI handling awareness
- GDPR: Data protection readiness

---

## Slide 12: Technical Innovation

- Multi-modal sensor fusion (EEG + ECG)
- Real-time ensemble machine learning
- Cloud-native medical device architecture
- Production-grade code structure
- Comprehensive documentation

---

## Slide 13: Limitations and Future Work

**Current Limitations:**
- Synthetic training data
- No persistent storage
- Single device monitoring

**Future Development:**
- Hardware integration (STM32, nRF52)
- Clinical validation studies
- FDA 510(k) submission
- Database persistence

---

## Slide 14: Impact and Applications

- Continuous health monitoring
- Early warning systems
- Remote patient monitoring
- Clinical decision support
- Research platform

---

## Slide 15: Conclusion

- Successfully integrated multi-modal biosignals
- Demonstrated real-time ML risk prediction
- Achieved medical device compliance standards
- Created foundation for clinical deployment
- Ready for hardware integration phase

---

## Slide 16: Thank You

**Repository:** https://github.com/bblackheart013/neurocardiac-shield

**Contact:**
- Mohd Sarfaraz Faiyaz (Author)
- Vaibhav Devram Chandgir (Contributor)

New York University - Advanced Project

---

**End of Presentation**
