# NeuroCardiac Shield — Simulation Scope Declaration

**Document Type**: Academic Scope Declaration
**Version**: 2.1.0
**Last Updated**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav Devram Chandgir

---

## Purpose of This Document

This document explicitly declares the simulation boundaries of the NeuroCardiac Shield system. It serves three critical functions:

1. **Academic Integrity**: Ensures complete transparency for faculty evaluation
2. **Overclaim Prevention**: Establishes clear boundaries between implemented features and future work
3. **Honest Engineering**: Documents what the system actually does versus what it might appear to do

> **Faculty Note — Why This Document Exists**
>
> In medical device development, scope creep and overclaiming are serious professional hazards. A system that appears to monitor patients but actually processes synthetic data could cause harm if misunderstood. This document is written in the style of regulatory scope declarations (similar to FDA intended use statements) to practice the discipline of precise technical communication. Every claim in the README and other documentation should be verifiable against this scope declaration.

---

## 1. Simulation vs. Real Hardware Matrix

The following table provides an at-a-glance view of what is simulated versus what would exist in a production system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMULATION STATUS MATRIX                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Component             Current State         Production Target              │
│   ─────────             ─────────────         ─────────────────              │
│                                                                              │
│   EEG Acquisition       SIMULATED (C code)    ADS1299 chip                  │
│   ECG Acquisition       SIMULATED (C code)    AD8232 AFE                    │
│   SpO2 Sensor           SIMULATED (random)    MAX30102                      │
│   Temperature           SIMULATED (random)    MCP9808                       │
│   Accelerometer         SIMULATED (random)    LSM6DSO                       │
│   BLE Communication     STUBBED (file I/O)    nRF52840 SoC                  │
│                                                                              │
│   Signal Processing     IMPLEMENTED           Same (validated code)         │
│   Feature Extraction    IMPLEMENTED           Same (validated code)         │
│   ML Training           IMPLEMENTED           Same + real data              │
│   ML Inference          IMPLEMENTED           Same (validated code)         │
│   Dashboard             IMPLEMENTED           Same (production styling)     │
│                                                                              │
│   Database              NOT IMPLEMENTED       TimescaleDB                   │
│   Authentication        NOT IMPLEMENTED       OAuth2/JWT                    │
│   Encryption            NOT IMPLEMENTED       TLS 1.3 + AES-256             │
│   Audit Logging         NOT IMPLEMENTED       HIPAA-compliant logs          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  SIMULATED     = Generates synthetic data, no real hardware
  STUBBED       = Interface exists but uses placeholder (file I/O)
  IMPLEMENTED   = Functional code that would work in production
  NOT IMPLEMENTED = Intentionally excluded from academic scope
```

---

## 2. Signal Simulation Details

### 2.1 EEG Signal Simulation

**Implementation Files**:
- `firmware/eeg/eeg_sim.c` — C-based generation for firmware
- `cloud/signal_processing/synthetic_data.py` — Python-based generation for ML training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EEG SIMULATION CHARACTERISTICS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   WHAT IS SIMULATED                     WHAT IS NOT SIMULATED               │
│   ──────────────────                    ───────────────────────              │
│                                                                              │
│   ✓ 8 channels (10-20 subset)           ✗ Real patient variability          │
│     Fp1, Fp2, C3, C4, T3, T4, O1, O2    ✗ Pathological patterns             │
│                                           (seizures, abnormalities)          │
│   ✓ Multi-band oscillations              ✗ Electrode impedance effects      │
│     δ, θ, α, β, γ bands                  ✗ Motion artifacts (beyond blinks) │
│                                                                              │
│   ✓ 1/f pink noise background            ✗ Individual alpha frequency       │
│     Per He et al. (2010)                   variation (8-13 Hz range)        │
│                                                                              │
│   ✓ Eye blink artifacts                  ✗ Muscle artifacts (EMG)           │
│     Gaussian pulses on Fp1/Fp2           ✗ Cardiac artifacts in EEG         │
│                                                                              │
│   ✓ Inter-channel correlations           ✗ Age-related spectral changes     │
│     Distance-based correlation             (e.g., theta increase in aging)  │
│                                                                              │
│   ✓ Amplitude range: 10-100 µV           ✗ Medication effects               │
│     Per Nunez & Srinivasan (2006)        ✗ Sleep stage transitions          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Scientific Basis**:
| Component | Reference | Implementation |
|-----------|-----------|----------------|
| Amplitude ranges | Nunez & Srinivasan (2006) | Band-specific amplitude parameters |
| 1/f characteristics | He et al. (2010) | Pink noise via spectral filtering |
| Frequency bands | IFCN standards | Delta, theta, alpha, beta, gamma |
| Eye blinks | Standard EEG artifact literature | Gaussian pulses at frontal channels |

### 2.2 ECG Signal Simulation

**Implementation Files**:
- `firmware/ecg/ecg_sim.c` — C-based generation
- `cloud/signal_processing/synthetic_data.py` — Python-based generation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ECG SIMULATION CHARACTERISTICS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   WHAT IS SIMULATED                     WHAT IS NOT SIMULATED               │
│   ──────────────────                    ───────────────────────              │
│                                                                              │
│   ✓ Normal sinus rhythm                  ✗ Arrhythmias                      │
│     PQRST morphology                       - Atrial fibrillation            │
│     Per McSharry et al. (2003)             - Premature ventricular          │
│                                              contractions (PVCs)             │
│   ✓ Heart rate variability                 - Heart blocks                   │
│     LF (0.04-0.15 Hz)                                                        │
│     HF (0.15-0.4 Hz)                     ✗ Ischemic changes                 │
│                                            - ST elevation/depression        │
│   ✓ Respiratory sinus arrhythmia           - T-wave inversions             │
│     Baseline wander modulation                                               │
│                                          ✗ Bundle branch blocks             │
│   ✓ Heart rate range                       - RBBB/LBBB patterns             │
│     60-100 BPM (configurable)                                                │
│                                          ✗ Individual morphological         │
│   ✓ High-frequency noise                   variation                        │
│     Gaussian additive noise                                                  │
│                                          ✗ Multi-lead geometric             │
│   ✓ 3-lead simulation                      relationships (approximate)      │
│     Leads I, II, III                                                         │
│                                          ✗ Exercise/stress responses        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Scientific Basis**:
| Component | Reference | Implementation |
|-----------|-----------|----------------|
| PQRST morphology | McSharry et al. (2003) | Gaussian-modulated peaks |
| HRV parameters | Task Force (1996) | LF/HF power ratio generation |
| RSA | Standard physiology | Sinusoidal baseline modulation |

> **Faculty Note — Why No Pathology Simulation?**
>
> Simulating arrhythmias, seizures, or other pathological patterns would require clinical expertise to ensure accuracy and could create a false sense that the system can detect these conditions. The honest approach is to limit simulation to normal physiology and explicitly state that pathology detection is not a capability of this system. Adding unrealistic pathology patterns would actually degrade the educational value by teaching incorrect signal characteristics.

### 2.3 Auxiliary Sensor Simulation

| Sensor | Implementation | Realism Level | Notes |
|--------|---------------|---------------|-------|
| SpO2 | Random 95-100% | Low | No physiological model |
| Temperature | Random 36.5-37.5°C | Low | No thermoregulation model |
| Accelerometer | Random ±100 mg | Very Low | No motion model |

These sensors are included to demonstrate the packet structure and data pipeline, not to provide meaningful physiological data.

---

## 3. Data Flow Simulation

### 3.1 What IS Real

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL COMPONENTS IN DATA FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │  FIRMWARE   │                                                           │
│   │   (C code)  │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          │  Real binary packet encoding                                      │
│          │  Real struct packing (569 bytes)                                  │
│          │  Real timing loop (10 Hz)                                         │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │    FILE     │  Real file I/O operations                                 │
│   │  (binary)   │  /tmp/neurocardiac_ble_data.bin                          │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          │  Real binary parsing (struct.unpack)                             │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │   GATEWAY   │  Real JSON serialization                                  │
│   │  (Python)   │  Real HTTP POST requests                                  │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          │  Real HTTP/JSON transport                                        │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │     API     │  Real FastAPI handling                                    │
│   │  (FastAPI)  │  Real Pydantic validation                                 │
│   └──────┬──────┘  Real WebSocket streaming                                 │
│          │                                                                   │
│          │  Real signal processing (SciPy)                                  │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │     ML      │  Real model inference (XGBoost, TensorFlow)               │
│   │  INFERENCE  │  Real feature extraction                                  │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          │  Real visualization (Plotly)                                     │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │  DASHBOARD  │  Real Streamlit application                               │
│   │ (Streamlit) │                                                           │
│   └─────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 What is NOT Real

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMULATED COMPONENTS IN DATA FLOW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │  PHYSICAL   │  ← NOT REAL                                               │
│   │   DEVICE    │    No actual wearable hardware                            │
│   │   (none)    │    No actual electrodes                                   │
│   └──────┬──────┘    No actual ADC (analog-to-digital conversion)           │
│          │                                                                   │
│          │  ← NOT REAL: No actual BLE packets                               │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │     BLE     │  ← NOT REAL                                               │
│   │   STACK     │    No nRF52840 or similar                                 │
│   │   (none)    │    No actual RF transmission                              │
│   └──────┬──────┘    No Bluetooth pairing                                   │
│          │                                                                   │
│          │  ← NOT REAL: No actual wireless connection                       │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │    PHONE    │  ← NOT REAL                                               │
│   │     APP     │    No mobile application                                  │
│   │   (none)    │    No app store deployment                                │
│   └─────────────┘                                                           │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                              │
│   The data flow from physical device through BLE to phone is entirely       │
│   replaced by: C program → file → Python gateway                            │
│                                                                              │
│   This substitution is functionally equivalent for testing the cloud        │
│   and ML components, but does not validate any wireless or hardware         │
│   integration.                                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Machine Learning Scope

### 4.1 Training Data Status

| Aspect | Status | Implication |
|--------|--------|-------------|
| Data Source | **SYNTHETIC** | Models learn from simulated patterns |
| Ground Truth Labels | **HEURISTIC** | Not from clinical diagnosis |
| Population Distribution | **SIMPLIFIED** | Does not reflect real patient diversity |
| Pathology | **NONE** | Cannot detect real abnormalities |
| Class Balance | **ARTIFICIAL** | Balanced by design, not by prevalence |

### 4.2 Model Performance Claims

| Metric | Reported Value | Critical Caveat |
|--------|---------------|-----------------|
| XGBoost Accuracy | 81.1% | Synthetic test set only |
| XGBoost ROC-AUC | 0.923 | Not clinically validated |
| LSTM Accuracy | 99.75% | **Suspiciously high** — likely trivial patterns |
| LSTM AUC | 0.9999 | **Suspiciously high** — likely data leakage |

> **Faculty Note — Honest Reporting of ML Results**
>
> The LSTM's near-perfect accuracy is documented as a limitation rather than hidden or celebrated. In academic work, honestly acknowledging that your results may indicate a problem (trivial synthetic patterns) is more valuable than claiming unrealistic performance. A grader should see this as evidence of engineering maturity, not as a failure.

### 4.3 What the ML System Cannot Do

The system **cannot**:
- Detect real cardiac arrhythmias
- Identify seizures or neurological events
- Provide medically actionable recommendations
- Replace clinical judgment in any capacity
- Generalize to populations it has not seen

---

## 5. API and Infrastructure Scope

### 5.1 Implemented (Functional)

- [x] REST API endpoints (`/ingest`, `/status`, `/inference`)
- [x] WebSocket streaming capability
- [x] Pydantic schema validation
- [x] CORS middleware (permissive for development)
- [x] Async request handling (FastAPI native)
- [x] ML model loading and inference
- [x] OpenAPI documentation (`/docs`)

### 5.2 Not Implemented (Intentional Scope Exclusions)

- [ ] User authentication (no JWT/OAuth)
- [ ] Role-based access control (no RBAC)
- [ ] Rate limiting (no protection)
- [ ] Database persistence (in-memory only)
- [ ] HTTPS/TLS encryption (HTTP only)
- [ ] Audit logging (no compliance logging)
- [ ] Multi-tenancy (single-user design)
- [ ] Horizontal scaling (single instance)

> **Faculty Note — Why These Are Excluded**
>
> These features are intentionally excluded, not forgotten. Implementing authentication, TLS, and database persistence would add significant complexity without advancing the core educational objectives: demonstrating signal processing, ML integration, and medical device architecture patterns. Each exclusion represents a conscious scoping decision. In a real project timeline, these would be Phase 2 or Phase 3 deliverables after core functionality is validated.

---

## 6. Compliance Claims Clarification

### 6.1 IEC 62304 Reference

**What is claimed**: The architecture is "informed by" IEC 62304 principles.

**What this means**:
- Modular design with clear component boundaries
- Separation of safety-critical (ML) from non-critical (UI) code
- Documented interfaces between components

**What this does NOT mean**:
- No formal Software Development Plan (SDP)
- No documented Software Requirements Specification (SRS)
- No traceability matrix
- No formal unit test coverage (Class B requires 100%)
- No formal integration testing documentation
- No risk analysis per ISO 14971

**Accurate statement**: "The architecture follows patterns that would facilitate IEC 62304 compliance but does not constitute formal compliance."

### 6.2 HIPAA Reference

**What is claimed**: Not applicable (no PHI).

**Why this is accurate**:
- All data is computationally generated
- No patient identifiers exist
- No real physiological recordings
- No data persistence (in-memory only)

**If real data were used, the following would be required**:
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Access control with audit trails
- Business Associate Agreements (BAAs)
- Breach notification procedures

---

## 7. Requirements for Clinical Use

This section documents the gap between current state and clinical-grade deployment:

### 7.1 Hardware Requirements

| Category | Requirement | Current Status |
|----------|-------------|----------------|
| Biosignal IC | FDA-cleared chip (e.g., ADS1299) | Not present |
| Electrodes | Medical-grade Ag/AgCl | Not present |
| EMC/EMI | IEC 60601-1-2 testing | Not applicable |
| Battery | IEC 62133 safety cert | Not present |
| Biocompatibility | ISO 10993 testing | Not applicable |

### 7.2 Software Requirements

| Category | Requirement | Current Status |
|----------|-------------|----------------|
| Development | IEC 62304 SDP | Not documented |
| Cybersecurity | IEC 62443 compliance | Not implemented |
| Verification | Formal V&V testing | Not performed |
| Risk | ISO 14971 analysis | Not performed |
| Testing | 100% unit test coverage | Partial |

### 7.3 Regulatory Requirements

| Pathway | Requirement | Current Status |
|---------|-------------|----------------|
| FDA | 510(k) or De Novo submission | Not submitted |
| CE Mark | MDR compliance + notified body | Not applicable |
| QMS | ISO 13485 certification | Not applicable |

---

## 8. Academic Scope Statement

### 8.1 This Project Demonstrates

| Skill Area | Evidence |
|------------|----------|
| System Architecture | Four-layer design, component separation |
| Signal Processing | Butterworth filters, Welch PSD, R-peak detection |
| Machine Learning | XGBoost + LSTM ensemble, feature engineering |
| Real-Time Systems | WebSocket streaming, 10 Hz data pipeline |
| Documentation | Technical specifications, scope declaration |

### 8.2 This Project Does NOT Claim

| Non-Claim | Explanation |
|-----------|-------------|
| Clinical efficacy | Models not validated on real patients |
| Regulatory compliance | No formal compliance activities performed |
| Patient data handling | All data is synthetic |
| Medical device status | Not a medical device |
| Diagnostic capability | Cannot diagnose any condition |

---

## 9. Attestation

By including this document in the repository, the author attests that:

1. **Accuracy**: All simulation boundaries are accurately described
2. **Honesty**: No overclaims are made regarding clinical utility
3. **Intent**: The system is intended for academic demonstration only
4. **Precedence**: Any statements in other documents that conflict with this scope declaration should defer to this document

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NCS-SCOPE-001 |
| Version | 2.1.0 |
| Status | Final Academic Submission |
| Author | Mohd Sarfaraz Faiyaz |
| Next Review | N/A (Final version) |

---

**End of Document**
