# Project Summary for NYU Professors

---

## NeuroCardiac Shield: Multi-Modal Physiological Monitoring Platform

**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Course:** Advanced Project - Medical Device Software Development
**Institution:** New York University
**Date:** November 2025

---

## Executive Summary

NeuroCardiac Shield is a comprehensive medical device software platform that integrates electroencephalography (EEG) and electrocardiography (ECG) signal acquisition with machine learning-based risk prediction. The system demonstrates end-to-end software development capabilities spanning embedded firmware simulation, cloud backend services, real-time signal processing, ensemble machine learning inference, and interactive data visualization.

The platform implements 8-channel EEG monitoring using the international 10-20 electrode placement standard (Fp1, Fp2, C3, C4, T3, T4, O1, O2) alongside 3-lead ECG acquisition with PQRST morphology analysis. Signal processing employs Butterworth bandpass filters, 60 Hz notch filtering, and Pan-Tompkins QRS detection algorithms. Feature extraction yields 74 dimensions including EEG frequency band powers (delta through gamma), heart rate variability metrics (SDNN, RMSSD, pNN50, LF/HF ratio), and spectral entropy measures.

Risk prediction utilizes a weighted ensemble combining XGBoost classifiers for interpretable feature analysis (60% weight) and bidirectional LSTM networks for temporal pattern recognition (40% weight), producing three-class risk stratification (LOW, MEDIUM, HIGH) with confidence scoring.

The architecture was designed with IEC 62304 Class B medical device software lifecycle compliance in mind, incorporating HIPAA-aware data handling patterns and GDPR-ready architectural principles. The system achieves end-to-end latency under one second with machine learning inference times of approximately 80 milliseconds.

The project deliverables include complete source code (C firmware, Python backend, machine learning models), comprehensive technical documentation, GitHub Actions CI/CD pipelines, and a system validation report confirming functional operation of all components.

Repository: https://github.com/bblackheart013/neurocardiac-shield

This work demonstrates proficiency in embedded systems programming, cloud-native architecture design, biomedical signal processing, production machine learning deployment, healthcare regulatory frameworks, and professional software engineering practices suitable for medical device development.

---

**End of Summary**
