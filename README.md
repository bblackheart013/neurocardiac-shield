<p align="center">
  <img src="https://img.shields.io/badge/🧠_Brain-EEG_8_Channels-3b82f6?style=for-the-badge" alt="EEG"/>
  <img src="https://img.shields.io/badge/❤️_Heart-ECG_3_Lead-ef4444?style=for-the-badge" alt="ECG"/>
  <img src="https://img.shields.io/badge/🤖_ML-76_Features-10b981?style=for-the-badge" alt="ML"/>
</p>

<h1 align="center">
  <br>
  🛡️ NeuroCardiac Shield
  <br>
</h1>

<h3 align="center">
  <em>Integrated Brain-Heart Monitoring for Next-Generation Wearables</em>
</h3>

<p align="center">
  <strong>A complete multi-modal physiological monitoring platform integrating EEG and ECG analysis for real-time cardiovascular-neurological risk assessment</strong>
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-the-science">The Science</a> •
  <a href="#-documentation">Docs</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/Tests-67%2F67_Passing-success?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/TypeScript-5.0-3178c6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/License-Academic-orange?style=flat-square" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NYU_Tandon-ECE--GY_9953-57068c?style=flat-square" alt="NYU"/>
  <img src="https://img.shields.io/badge/Fall-2025-purple?style=flat-square" alt="Term"/>
  <img src="https://img.shields.io/badge/Advisor-Dr._Matthew_Campisi-blueviolet?style=flat-square" alt="Advisor"/>
</p>

---

<p align="center">
  <img width="1313" height="1289" alt="image" src="https://github.com/user-attachments/assets/b3c5e2b7-9bbb-4ace-ab5a-91c26dbdcb26" />
</p>

---

## 🌟 Why NeuroCardiac Shield?

<table>
<tr>
<td width="50%">

### The Problem

**805,000** heart attacks occur in the US every year. **1 in 26** people develop epilepsy. These aren't separate problems—the heart and brain are connected through the autonomic nervous system.

- Cardiac events cause neurological symptoms *before* the heart attack
- Neurological events trigger cardiac arrhythmias (SUDEP kills thousands yearly)
- **No consumer device monitors both simultaneously**

</td>
<td width="50%">

### Our Solution

We built a complete end-to-end system from **embedded firmware** to **machine learning** to demonstrate how next-generation wearable medical devices could detect cardiovascular-neurological risks *before* they become emergencies.

- 🧠 8-channel EEG with physiological state detection
- ❤️ 3-lead ECG with HRV analysis
- 🤖 76-feature ensemble ML for risk prediction
- 📱 Real-time web dashboard with device connectivity

</td>
</tr>
</table>

---

## 🎯 Key Features

<table>
<tr>
<td align="center" width="25%">
<h3>🔬</h3>
<h4>Clinically-Grounded Signals</h4>
<p>EEG generated using multi-band synthesis with 1/f pink noise. ECG follows the McSharry dynamical model from IEEE literature.</p>
</td>
<td align="center" width="25%">
<h3>🧪</h3>
<h4>Physiological States</h4>
<p>Real-time simulation of Alert, Relaxed, Drowsy, and Stressed states with accurate band power modulation.</p>
</td>
<td align="center" width="25%">
<h3>📊</h3>
<h4>Interactive Visualization</h4>
<p>Smooth, real-time signal visualization with state selection, component exploration, and educational overlays.</p>
</td>
<td align="center" width="25%">
<h3>📱</h3>
<h4>Device Connectivity</h4>
<p>Web Bluetooth API integration for real heart rate monitors. Connect your Polar H10, Garmin, or Wahoo device.</p>
</td>
</tr>
</table>

---

## 🚀 Live Demo

### [**→ Launch NeuroCardiac Shield**](https://neurocardiac-shield.netlify.app)

<p align="center">
  <a href="https://neurocardiac-shield.netlify.app">
    <img src="https://img.shields.io/badge/🌐_Live_Demo-neurocardiac--shield.netlify.app-00C7B7?style=for-the-badge&logo=netlify&logoColor=white" alt="Live Demo"/>
  </a>
</p>

<details>
<summary><strong>📸 Screenshots</strong></summary>
<br>

| Live Signals | Signal Science | Device Connection |
|:---:|:---:|:---:|
| Real-time 8-channel EEG and ECG visualization with state selection | Deep dive into data generation algorithms with interactive explorers | Connect real Bluetooth heart rate monitors |

</details>

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/bblackheart013/neurocardiac-shield.git
cd neurocardiac-shield

# Run setup (creates venvs, installs deps, trains models)
chmod +x setup.sh && ./setup.sh

# Verify system integrity (67/67 checks should pass)
python verify_system.py

# Launch the full demo
./run_complete_demo.sh
```

**Frontend Only:**
```bash
cd web
npm install
npm run dev
# Open http://localhost:3000
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐ │
│    │   FIRMWARE   │───▶│   GATEWAY    │───▶│     API      │───▶│DASHBOARD│ │
│    │     (C)      │    │   (Python)   │    │  (FastAPI)   │    │(Next.js)│ │
│    └──────────────┘    └──────────────┘    └──────────────┘    └─────────┘ │
│                                                                              │
│    8-ch EEG            Binary→JSON          DSP + ML           Real-time    │
│    3-lead ECG          BLE Bridge           76 Features        Visualize    │
│    250 Hz              10 pkt/s             XGB + LSTM         WebSocket    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
neurocardiac-shield/
├── 📁 firmware/                 # C - Simulated embedded acquisition
│   ├── main.c                   # Main loop, 569-byte packet assembly
│   ├── eeg/eeg_sim.c           # 8-channel EEG multi-band synthesis
│   └── ecg/ecg_sim.c           # PQRST morphology (McSharry model)
│
├── 📁 cloud/                    # Python - Backend services
│   ├── api/server.py           # FastAPI REST/WebSocket server
│   └── signal_processing/       # DSP, filtering, feature extraction
│
├── 📁 ml/                       # Machine learning pipeline
│   ├── model/                   # XGBoost + BiLSTM ensemble
│   └── checkpoints/             # Trained model weights
│
├── 📁 web/                      # Next.js 14 - Frontend
│   └── src/app/page.tsx        # Complete dashboard implementation
│
└── 📁 docs/                     # Technical documentation
    ├── ARCHITECTURE.md
    ├── DATA_FLOW.md
    └── ML_PIPELINE.md
```

---

## 🔬 The Science

### EEG Signal Generation

Our EEG synthesis uses **multi-band composition** with **1/f pink noise** based on peer-reviewed neuroscience literature:

| Band | Frequency | Physiological Meaning |
|:----:|:---------:|:---------------------|
| **δ Delta** | 0.5-4 Hz | Deep sleep, unconsciousness |
| **θ Theta** | 4-8 Hz | Drowsiness, meditation |
| **α Alpha** | 8-13 Hz | Relaxed wakefulness, eyes closed |
| **β Beta** | 13-30 Hz | Active thinking, concentration |
| **γ Gamma** | 30-50 Hz | High-level cognition, perception |

**Key Algorithm:** Voss-McCartney for 1/f noise generation

### ECG Signal Generation

ECG follows the **McSharry dynamical model** (IEEE Trans Biomed Eng, 2003):

- **PQRST morphology** using Gaussian functions
- **Heart Rate Variability** with respiratory sinus arrhythmia
- **Beat-to-beat variation** for realistic appearance
- **Optional pathology** simulation (PVC, ST elevation)

### Feature Engineering (76 Features)

```
┌─────────────────────────────────────────────────────┐
│  EEG Features (66)                                  │
│  ├── Band Powers: 5 bands × 8 channels = 40        │
│  ├── Spectral Ratios: alpha/beta, theta/alpha      │
│  └── Entropy & Coherence metrics                   │
├─────────────────────────────────────────────────────┤
│  HRV Features (7)                                   │
│  ├── Time Domain: SDNN, RMSSD, pNN50              │
│  └── Frequency Domain: LF, HF, LF/HF ratio        │
├─────────────────────────────────────────────────────┤
│  ECG Morphology (3)                                │
│  └── QRS duration, RR interval, amplitude          │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Documentation

| Document | Description |
|:---------|:------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, component interfaces |
| [DATA_FLOW.md](docs/DATA_FLOW.md) | Binary packet format, throughput analysis |
| [ML_PIPELINE.md](docs/ML_PIPELINE.md) | Feature engineering, model training |
| [SIMULATION_SCOPE.md](docs/SIMULATION_SCOPE.md) | What's simulated vs real |

---

## 🧪 Verification

Run the comprehensive verification suite:

```bash
python verify_system.py
```

**Expected:** `67/67 checks passed`

| Category | Checks |
|:---------|:-------|
| Directory Structure | ✅ All required directories exist |
| Core Source Files | ✅ All implementation files present |
| Python Dependencies | ✅ NumPy, SciPy, FastAPI importable |
| Signal Processing | ✅ Filters and features validated |
| ML Inference | ✅ Model checkpoints loadable |

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|:-------|:------|:------|
| **EEG Channels** | 8 | 10-20 system subset |
| **Sample Rate** | 250 Hz | Captures up to gamma band |
| **ML Features** | 76 | Hand-crafted, interpretable |
| **XGBoost Accuracy** | 81.1% | 3-class risk prediction |
| **BiLSTM Accuracy** | 99.75% | On synthetic data |
| **Ensemble** | 60/40 | XGB/LSTM weighting |
| **Packet Rate** | 10 Hz | Real-time streaming |

---

## 🔗 Connect Real Devices

NeuroCardiac Shield supports **Web Bluetooth API** for connecting real heart rate monitors:

### Compatible Devices

| Device | Type | Tested |
|:-------|:-----|:------:|
| Polar H10 | Chest Strap | ✅ |
| Polar OH1/Verity | Optical | ✅ |
| Garmin HRM-Pro | Chest Strap | ✅ |
| Wahoo TICKR | Chest Strap | ✅ |
| Any BLE HR Monitor | GATT 0x180D | ✅ |

**Note:** Web Bluetooth requires Chrome/Edge on desktop or Chrome on Android.

---

## 📖 Scientific References

1. **McSharry, P.E., et al.** (2003). A dynamical model for generating synthetic electrocardiogram signals. *IEEE Trans Biomed Eng.*

2. **Task Force of ESC and NASPE** (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *Circulation.*

3. **Nunez, P.L., & Srinivasan, R.** (2006). Electric Fields of the Brain: The Neurophysics of EEG. *Oxford University Press.*

4. **Voss, R.F., & Clarke, J.** (1975). 1/f noise in music and speech. *Nature.*

5. **Pan, J., & Tompkins, W.J.** (1985). A Real-Time QRS Detection Algorithm. *IEEE Trans Biomed Eng.*

---

## 👥 Team

<table>
<tr>
<td align="center">
<strong>Mohd Sarfaraz Faiyaz</strong>
<br>
<em>Systems & Machine Learning</em>
<br>
<a href="https://github.com/bblackheart013">@bblackheart013</a>
</td>
<td align="center">
<strong>Vaibhav D. Chandgir</strong>
<br>
<em>Signal Processing</em>
</td>
</tr>
</table>

**Advisor:** Dr. Matthew Campisi, NYU Tandon School of Engineering

---

## ⚠️ Disclaimer

> **This is an academic prototype.** All physiological signals are computationally generated using scientifically-grounded models. This system is **not FDA-cleared**, **not clinically validated**, and **not intended for medical use**. The ML models are trained exclusively on synthetic data and carry no clinical validity.

---

## 📄 License

This project is developed for academic purposes at **New York University Tandon School of Engineering**. All rights reserved.

---

<p align="center">
  <strong>NYU Tandon School of Engineering — Advanced Project (ECE-GY 9953) — Fall 2025</strong>
</p>

<p align="center">
  <a href="#-neurocardiac-shield">⬆️ Back to Top</a>
</p>
