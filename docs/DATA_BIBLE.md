# NeuroCardiac Shield — Data Bible

**Document Type**: Authoritative Data Specification
**Version**: 2.0.0
**Last Updated**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Timing & Sampling](#2-timing--sampling)
3. [EEG Specification](#3-eeg-specification)
4. [ECG Specification](#4-ecg-specification)
5. [Auxiliary Sensors](#5-auxiliary-sensors)
6. [Packet Structure](#6-packet-structure)
7. [Feature Extraction](#7-feature-extraction)
8. [Example Walkthrough](#8-example-walkthrough)
9. [Validation Rules](#9-validation-rules)

---

## 1. Introduction

This document is the **authoritative specification** for all data formats in the NeuroCardiac Shield system. Every byte, every unit, every scaling factor is defined here.

### 1.1 Design Principles

1. **Explicit over implicit**: All units are stated, no assumptions required
2. **Traceable**: Every value can be verified against this document
3. **Consistent**: Same format from firmware to visualization
4. **Validated**: Bounds checking at every layer boundary

### 1.2 Quick Reference

| Signal | Channels | Sample Rate | Packet Rate | Samples/Packet | Units |
|--------|----------|-------------|-------------|----------------|-------|
| EEG | 8 | 250 Hz | 10 Hz | 25 | µV |
| ECG | 3 | 250 Hz | 10 Hz | 25 | mV |
| SpO2 | 1 | N/A | 10 Hz | 1 | % |
| Temperature | 1 | N/A | 10 Hz | 1 | °C |
| Accelerometer | 3 | N/A | 10 Hz | 1 | g |

---

## 2. Timing & Sampling

### 2.1 Sampling Rate

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TIMING HIERARCHY                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SAMPLE RATE = 250 Hz                                                   │
│   ─────────────────────                                                  │
│   One sample every 4 milliseconds                                        │
│   Nyquist frequency: 125 Hz (sufficient for gamma band)                  │
│                                                                          │
│   PACKET RATE = 10 Hz                                                    │
│   ───────────────────                                                    │
│   One packet every 100 milliseconds                                      │
│   Contains 25 samples per channel                                        │
│                                                                          │
│   WINDOW SIZE (ML) = 10 seconds                                          │
│   ────────────────────────────                                           │
│   2500 samples per channel                                               │
│   100 packets accumulated                                                │
│                                                                          │
│   Timeline:                                                              │
│                                                                          │
│   0ms    4ms    8ms   12ms  ...  96ms  100ms  (1 packet = 25 samples)   │
│    │      │      │      │         │      │                               │
│    ▼      ▼      ▼      ▼         ▼      ▼                               │
│   [s0]   [s1]   [s2]   [s3] ... [s24] [packet sent]                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Latency Budget

| Stage | Typical Latency | Maximum Latency |
|-------|----------------|-----------------|
| Sensor ADC | < 1 ms | 2 ms |
| Packet assembly | < 1 ms | 2 ms |
| BLE transmission | 10-30 ms | 100 ms |
| API ingestion | < 5 ms | 20 ms |
| Signal processing | < 10 ms | 50 ms |
| ML inference | 50-100 ms | 500 ms |
| Dashboard render | < 16 ms | 50 ms |
| **Total (end-to-end)** | **~100 ms** | **~750 ms** |

### 2.3 Timestamp Specification

```python
timestamp_ms: int
    # Unix timestamp in milliseconds (64-bit safe)
    # Origin: 1970-01-01 00:00:00 UTC
    # Example: 1703001600000 = 2024-12-19 12:00:00 UTC
    # Resolution: 1 millisecond
    # Overflow: Year 292,277,026 (not a concern)
```

---

## 3. EEG Specification

### 3.1 Channel Configuration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      10-20 ELECTRODE PLACEMENT                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                              NASION                                      │
│                                 │                                        │
│                      ┌──────────┼──────────┐                            │
│                     Fp1        Fpz        Fp2                           │
│                      │          │          │                             │
│                      │    ┌─────┴─────┐    │                             │
│                      │    │           │    │                             │
│               T3 ────┼────C3    Cz    C4───┼──── T4                      │
│                      │    │           │    │                             │
│                      │    └─────┬─────┘    │                             │
│                      │          │          │                             │
│                     O1         Oz         O2                            │
│                      └──────────┼──────────┘                            │
│                                 │                                        │
│                              INION                                       │
│                                                                          │
│   CHANNELS IMPLEMENTED (subset of 10-20):                                │
│   ───────────────────────────────────────                                │
│   Index 0: Fp1 — Left frontal pole (eye blinks, frontal activity)       │
│   Index 1: Fp2 — Right frontal pole                                      │
│   Index 2: C3  — Left central (motor cortex)                             │
│   Index 3: C4  — Right central                                           │
│   Index 4: T3  — Left temporal (auditory processing)                     │
│   Index 5: T4  — Right temporal                                          │
│   Index 6: O1  — Left occipital (visual processing, alpha rhythm)        │
│   Index 7: O2  — Right occipital                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Amplitude Specification

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Unit** | Microvolts (µV) | Always floating-point |
| **Expected Range** | -500 to +500 µV | Warn if exceeded |
| **Absolute Maximum** | ±1000 µV | Reject if exceeded |
| **Baseline** | 0 µV | After high-pass filter |
| **Resolution** | 0.1 µV | Minimum detectable change |

### 3.3 Frequency Bands

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      EEG FREQUENCY BANDS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   BAND        RANGE (Hz)     AMPLITUDE (µV)    CLINICAL SIGNIFICANCE    │
│   ────        ──────────     ──────────────    ─────────────────────    │
│                                                                          │
│   Delta (δ)   0.5 – 4 Hz     20 – 200 µV       Deep sleep, pathology    │
│   Theta (θ)   4 – 8 Hz       10 – 100 µV       Drowsiness, meditation   │
│   Alpha (α)   8 – 13 Hz      15 – 50 µV        Relaxed wakefulness      │
│   Beta (β)    13 – 30 Hz     5 – 30 µV         Active thinking          │
│   Gamma (γ)   30 – 50 Hz     < 10 µV           High-level cognition     │
│                                                                          │
│   POWER SPECTRUM VISUALIZATION:                                          │
│                                                                          │
│   Power                                                                  │
│   (µV²/Hz)   ▲                                                          │
│              │    ╭───╮                                                  │
│         100  │   ╱     ╲         ╭─╮                                    │
│              │  ╱       ╲       ╱   ╲                                   │
│          50  │ ╱         ╲     ╱     ╲   ╭──╮                           │
│              │╱           ╲   ╱       ╲ ╱    ╲  ╭╮                      │
│          10  │─────────────╲─╱─────────╳──────╲╱──╲─────────────        │
│              └──────────────────────────────────────────▶ Freq (Hz)     │
│              0    4    8   13   20   30   40   50                       │
│                  δ    θ    α        β        γ                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Common Artifacts

| Artifact | Source | Frequency | Amplitude | Affected Channels |
|----------|--------|-----------|-----------|-------------------|
| Eye blink | EOG | 0.5-5 Hz | 50-200 µV | Fp1, Fp2 |
| Eye movement | EOG | 0.1-2 Hz | 20-100 µV | Fp1, Fp2 |
| Muscle (EMG) | Jaw/face | 20-300 Hz | 10-100 µV | T3, T4 |
| Cardiac | ECG coupling | 1-2 Hz | 5-20 µV | All |
| Line noise | Power grid | 50/60 Hz | Variable | All |

### 3.5 Filtering Applied

```python
# EEG Bandpass Filter Specification
filter_type: "Butterworth IIR"
order: 4
low_cutoff_hz: 0.5    # High-pass removes DC drift
high_cutoff_hz: 50.0  # Low-pass removes muscle/line noise
method: "filtfilt"    # Zero-phase (no delay)
```

---

## 4. ECG Specification

### 4.1 Lead Configuration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EINTHOVEN'S TRIANGLE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                              RA ●──────────────● LA                      │
│                                 ╲              ╱                         │
│                                  ╲   Lead I   ╱                          │
│                                   ╲          ╱                           │
│                                    ╲        ╱                            │
│                              Lead   ╲      ╱  Lead                       │
│                               II     ╲    ╱    III                       │
│                                       ╲  ╱                               │
│                                        ╲╱                                │
│                                        ●                                 │
│                                       LL                                 │
│                                                                          │
│   Lead Definitions:                                                      │
│   ─────────────────                                                      │
│   Lead I   = LA - RA    (horizontal, left-to-right)                     │
│   Lead II  = LL - RA    (diagonal, foot-to-right-arm)                   │
│   Lead III = LL - LA    (diagonal, foot-to-left-arm)                    │
│                                                                          │
│   Relationship: Lead I + Lead III = Lead II                              │
│                                                                          │
│   Index Mapping:                                                         │
│   ──────────────                                                         │
│   ecg[0] = Lead I                                                        │
│   ecg[1] = Lead II   (primary diagnostic lead)                          │
│   ecg[2] = Lead III                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Amplitude Specification

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Unit** | Millivolts (mV) | Always floating-point |
| **R-wave Range** | 0.5 to 2.5 mV | Normal adult |
| **P-wave Range** | 0.1 to 0.25 mV | Atrial depolarization |
| **T-wave Range** | 0.1 to 0.5 mV | Ventricular repolarization |
| **Expected Total** | -1.0 to +3.0 mV | Includes all waves |
| **Absolute Maximum** | ±5.0 mV | Reject if exceeded |

### 4.3 PQRST Morphology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     NORMAL ECG WAVEFORM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Amplitude                                                              │
│   (mV)                           R                                       │
│      1.5 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─/\─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │
│                                 /  \                                     │
│      1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─/    \─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │
│                               /      \                                   │
│      0.5 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─/        \─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │
│                             /          \                     ___         │
│              ___           /            \                   /   \        │
│      0.0 ─ ─/   \─ ─ ─ ─ ─/              \─ ─ ─ ─ ─ ─ ─ ─ ─/     \─ ─  │
│            P     \_____  /                \       ___      T      \_    │
│                   PR   \/                  \_____/   \____/          \  │
│     -0.5 ─ ─ ─ ─ ─ ─ ─ Q─ ─ ─ ─ ─ ─ ─ ─ ─ S ─ ─ ST ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                          │
│            └──┬──┘ └┬─┘ └──────┬──────┘   └───────┬───────┘             │
│            P-wave  PR    QRS Complex           ST-T                     │
│                  interval  (<120ms)           segment                   │
│                                                                          │
│   Wave Timing (normal):                                                  │
│   ─────────────────────                                                  │
│   P-wave duration:    80-120 ms                                          │
│   PR interval:        120-200 ms                                         │
│   QRS duration:       60-100 ms                                          │
│   QT interval:        350-440 ms (rate-dependent)                        │
│   RR interval:        600-1000 ms (60-100 BPM)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Heart Rate Variability (HRV) Metrics

The following metrics are computed from R-R intervals:

```python
# Time-domain metrics
def compute_hrv_time_domain(rr_intervals_ms: np.ndarray) -> dict:
    """
    rr_intervals_ms: Array of R-R intervals in milliseconds
    """
    return {
        'mean_rr_ms': np.mean(rr_intervals_ms),
        'sdnn_ms': np.std(rr_intervals_ms, ddof=1),
        'rmssd_ms': np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2)),
        'pnn50_percent': 100 * np.sum(np.abs(np.diff(rr_intervals_ms)) > 50) / len(rr_intervals_ms),
        'heart_rate_bpm': 60000 / np.mean(rr_intervals_ms)
    }
```

| Metric | Formula | Units | Normal Range | Interpretation |
|--------|---------|-------|--------------|----------------|
| **Mean RR** | mean(RR) | ms | 600-1000 | Average interval |
| **SDNN** | std(RR) | ms | 50-150 | Overall variability |
| **RMSSD** | √mean(ΔRR²) | ms | 20-50 | Parasympathetic activity |
| **pNN50** | %\|ΔRR\|>50ms | % | 10-40 | Parasympathetic activity |
| **LF/HF** | Power(0.04-0.15)/Power(0.15-0.4) | ratio | 0.5-2.0 | Autonomic balance |

### 4.5 Filtering Applied

```python
# ECG Bandpass Filter Specification
filter_type: "Butterworth IIR"
order: 3
low_cutoff_hz: 0.5    # High-pass removes baseline wander
high_cutoff_hz: 40.0  # Low-pass removes EMG noise
method: "filtfilt"    # Zero-phase (no delay)
```

---

## 5. Auxiliary Sensors

### 5.1 SpO2 (Pulse Oximetry)

```python
spo2_percent: float
    # Blood oxygen saturation percentage
    # Range: 0-100%
    # Normal: 95-100%
    # Warning: < 95% (hypoxemia)
    # Critical: < 90%
    # Resolution: 1%
```

### 5.2 Temperature

```python
temp_celsius: float
    # Body temperature
    # Range: 30.0-45.0 °C
    # Normal: 36.1-37.2 °C
    # Fever: > 38.0 °C
    # Resolution: 0.1 °C
    # Sensor: typically ear/forehead infrared
```

### 5.3 Accelerometer

```python
accel_xyz_g: List[float]  # [x, y, z]
    # 3-axis acceleration in g-force units
    # Range: ±16 g (device dependent)
    # At rest: [0, 0, ~1.0] (gravity)
    # Resolution: 0.001 g
    # Uses:
    #   - Motion artifact detection
    #   - Activity level estimation
    #   - Fall detection (future)
```

---

## 6. Packet Structure

### 6.1 Gold Schema (Canonical Format)

```python
@dataclass
class GoldPacket:
    # === TIMING ===
    timestamp_ms: int           # Unix timestamp (ms)
    device_id: str              # e.g., "ble-001", "sim-42"
    packet_seq: int             # Monotonic counter (0, 1, 2, ...)
    sample_rate_hz: float       # 250.0

    # === PHYSIOLOGICAL DATA ===
    eeg: List[List[float]]      # [8 channels][25 samples], µV
    ecg: List[List[float]]      # [3 leads][25 samples], mV

    # === AUXILIARY ===
    spo2_percent: float         # 0-100
    temp_celsius: float         # °C
    accel_xyz_g: List[float]    # [x, y, z] in g

    # === QUALITY ===
    quality_flags: int          # Bitmap (see below)

    # === EXTENSIBILITY ===
    extra: Dict[str, Any]       # Adapter-specific metadata
```

### 6.2 Quality Flags Bitmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      QUALITY FLAGS (16-bit)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Bit    Hex      Name                 Description                       │
│   ───    ───      ────                 ───────────                       │
│   0      0x0001   EEG_SATURATION       EEG ADC clipped                   │
│   1      0x0002   ECG_SATURATION       ECG ADC clipped                   │
│   2      0x0004   EEG_LEAD_OFF         EEG electrode disconnected        │
│   3      0x0008   ECG_LEAD_OFF         ECG electrode disconnected        │
│   4      0x0010   LOW_BATTERY          Battery < 20%                     │
│   5      0x0020   MOTION_ARTIFACT      High accelerometer activity       │
│   6      0x0040   POOR_SIGNAL          General quality warning           │
│   7      0x0080   CHECKSUM_INVALID     CRC check failed                  │
│   8      0x0100   SYNTHETIC_DATA       Data is simulated                 │
│   9-15   (reserved for future use)                                       │
│                                                                          │
│   Example: flags = 0x0105                                                │
│            = EEG_SATURATION | EEG_LEAD_OFF | SYNTHETIC_DATA              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Binary Wire Format (569 bytes)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BINARY PACKET LAYOUT                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Offset  Size   Type      Field           Description                   │
│   ──────  ────   ────      ─────           ───────────                   │
│                                                                          │
│   0       4      uint32    timestamp_ms    Milliseconds since boot       │
│   4       2      uint16    packet_id       Sequence counter              │
│   6       1      uint8     device_id       Device identifier             │
│   7       1      uint8     status_flags    Quality bitmap (low byte)     │
│                                                                          │
│   8       400    int16[]   eeg_data        8×25 samples, scale ÷10 = µV  │
│                                                                          │
│   408     150    int16[]   ecg_data        3×25 samples, scale ÷1000 = mV│
│                                                                          │
│   558     1      uint8     spo2            SpO2 percentage               │
│   559     2      int16     temp_x10        Temperature × 10 (°C)         │
│   561     6      int16[3]  accel_xyz       Acceleration × 1000 (g)       │
│                                                                          │
│   567     2      uint16    crc16           CRC16-CCITT checksum          │
│                                                                          │
│   TOTAL: 569 bytes                                                       │
│                                                                          │
│   Byte Order: Little-endian (x86/ARM native)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Example JSON Payload

```json
{
  "timestamp_ms": 1703001600000,
  "device_id": "simulated-001",
  "packet_seq": 42,
  "sample_rate_hz": 250.0,
  "eeg": [
    [12.5, 14.2, 11.8, 15.1, ...],  // Fp1: 25 samples in µV
    [10.3, 12.1, 9.8, 13.4, ...],   // Fp2
    [8.7, 9.2, 10.1, 8.5, ...],     // C3
    [7.9, 8.4, 9.0, 7.2, ...],      // C4
    [11.2, 10.8, 12.3, 11.0, ...],  // T3
    [9.5, 10.2, 9.8, 10.5, ...],    // T4
    [25.1, 28.3, 22.7, 30.2, ...],  // O1 (alpha dominant)
    [24.8, 27.9, 23.1, 29.5, ...]   // O2
  ],
  "ecg": [
    [0.05, 0.08, 0.12, 0.85, ...],  // Lead I: 25 samples in mV
    [0.06, 0.09, 0.14, 0.95, ...],  // Lead II
    [0.04, 0.07, 0.11, 0.78, ...]   // Lead III
  ],
  "spo2_percent": 98.5,
  "temp_celsius": 37.1,
  "accel_xyz_g": [0.02, -0.01, 0.98],
  "quality_flags": 256,  // SYNTHETIC_DATA
  "extra": {
    "hrv_state": "normal",
    "heart_rate_bpm": 72
  }
}
```

---

## 7. Feature Extraction

### 7.1 EEG Features (64 features)

```python
def extract_eeg_features(eeg: np.ndarray, fs: float = 250.0) -> dict:
    """
    Input:  eeg[8][N] — 8 channels, N samples
    Output: Dictionary with 64 features
    """
    features = {}

    # Band power per channel (5 bands × 8 channels = 40 features)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    for ch in range(8):
        for band_name, (f_low, f_high) in bands.items():
            power = compute_band_power(eeg[ch], fs, f_low, f_high)
            features[f'{CHANNEL_NAMES[ch]}_{band_name}_power'] = power

    # Global features (8 features)
    for ch in range(8):
        features[f'{CHANNEL_NAMES[ch]}_std'] = np.std(eeg[ch])

    # Asymmetry features (8 features)
    features['alpha_asymmetry_frontal'] = (
        features['Fp2_alpha_power'] - features['Fp1_alpha_power']
    )
    features['alpha_asymmetry_occipital'] = (
        features['O2_alpha_power'] - features['O1_alpha_power']
    )
    # ... (additional asymmetry features)

    # Coherence features (8 features)
    # Inter-hemispheric coherence in alpha band
    features['coherence_C3_C4_alpha'] = compute_coherence(
        eeg[2], eeg[3], fs, 8, 13
    )
    # ... (additional coherence features)

    return features  # 64 total features
```

### 7.2 HRV Features (12 features)

```python
def extract_hrv_features(rr_ms: np.ndarray) -> dict:
    """
    Input:  rr_ms — Array of R-R intervals in milliseconds
    Output: Dictionary with 12 features
    """
    # Time domain (5 features)
    features = {
        'mean_rr': np.mean(rr_ms),
        'sdnn': np.std(rr_ms, ddof=1),
        'rmssd': np.sqrt(np.mean(np.diff(rr_ms) ** 2)),
        'pnn50': 100 * np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms),
        'heart_rate': 60000 / np.mean(rr_ms)
    }

    # Frequency domain (5 features)
    # Compute power spectral density
    psd_freq, psd_power = compute_psd(rr_ms)

    features['vlf_power'] = integrate_band(psd_freq, psd_power, 0.003, 0.04)
    features['lf_power'] = integrate_band(psd_freq, psd_power, 0.04, 0.15)
    features['hf_power'] = integrate_band(psd_freq, psd_power, 0.15, 0.4)
    features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-10)
    features['total_power'] = features['vlf_power'] + features['lf_power'] + features['hf_power']

    # Nonlinear (2 features)
    features['sd1'] = np.std(np.diff(rr_ms)) / np.sqrt(2)
    features['sd2'] = np.sqrt(2 * np.var(rr_ms) - features['sd1'] ** 2)

    return features  # 12 total features
```

### 7.3 Combined Feature Vector (76 features)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE VECTOR COMPOSITION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Category            Count    Description                               │
│   ────────            ─────    ───────────                               │
│                                                                          │
│   EEG Band Power      40       5 bands × 8 channels                      │
│   EEG Statistics      8        Standard deviation per channel            │
│   EEG Asymmetry       8        Hemispheric differences                   │
│   EEG Coherence       8        Inter-channel relationships               │
│                       ──                                                 │
│   EEG Subtotal        64                                                 │
│                                                                          │
│   HRV Time Domain     5        SDNN, RMSSD, pNN50, etc.                 │
│   HRV Frequency       5        VLF, LF, HF power, LF/HF                 │
│   HRV Nonlinear       2        SD1, SD2 (Poincaré)                      │
│                       ──                                                 │
│   HRV Subtotal        12                                                 │
│                                                                          │
│   ═══════════════════════════════════════                               │
│   TOTAL               76       Features per 10-second window            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Example Walkthrough

### 8.1 From Raw Packet to Risk Score

```
┌─────────────────────────────────────────────────────────────────────────┐
│           COMPLETE DATA TRANSFORMATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STEP 1: Raw Packet Reception                                           │
│   ─────────────────────────────                                          │
│   569 bytes arrive from device                                           │
│   Binary: AA 00 00 01 A4 00 2A 01 00 4E 00 51 00 49 00 ...              │
│                                                                          │
│   STEP 2: Parse to GoldPacket                                            │
│   ──────────────────────────                                             │
│   timestamp_ms = 1703001600000                                           │
│   eeg[0] = [7.8, 8.1, 7.5, 7.9, ...] µV   (25 samples)                  │
│   ecg[0] = [0.05, 0.08, 0.12, ...] mV     (25 samples)                  │
│   ...                                                                    │
│                                                                          │
│   STEP 3: Accumulate Window (100 packets = 10 seconds)                   │
│   ─────────────────────────────────────────────────────                  │
│   eeg_window[8][2500] — 8 channels × 2500 samples                        │
│   ecg_window[3][2500] — 3 leads × 2500 samples                           │
│                                                                          │
│   STEP 4: Apply Filters                                                  │
│   ─────────────────────                                                  │
│   EEG: Butterworth 0.5-50 Hz, order 4, zero-phase                       │
│   ECG: Butterworth 0.5-40 Hz, order 3, zero-phase                       │
│                                                                          │
│   STEP 5: Detect R-peaks                                                 │
│   ──────────────────────                                                 │
│   Lead II signal → Pan-Tompkins algorithm                                │
│   r_peaks = [0.82s, 1.65s, 2.48s, 3.30s, ...]                           │
│   rr_intervals = [830ms, 830ms, 820ms, ...]                             │
│                                                                          │
│   STEP 6: Extract Features                                               │
│   ────────────────────────                                               │
│   eeg_features = extract_eeg_features(eeg_window)  # 64 features        │
│   hrv_features = extract_hrv_features(rr_intervals)  # 12 features      │
│   feature_vector = concat([eeg_features, hrv_features])  # 76 features  │
│                                                                          │
│   STEP 7: Scale Features                                                 │
│   ──────────────────────                                                 │
│   feature_vector_scaled = scaler.transform(feature_vector)               │
│   (StandardScaler: mean=0, std=1)                                        │
│                                                                          │
│   STEP 8: ML Inference                                                   │
│   ────────────────────                                                   │
│   xgb_proba = xgboost_model.predict_proba(feature_vector_scaled)        │
│             = [0.15, 0.55, 0.30]  (LOW, MEDIUM, HIGH)                   │
│                                                                          │
│   lstm_proba = lstm_model.predict(sequence)                              │
│              = [0.12, 0.60, 0.28]                                        │
│                                                                          │
│   STEP 9: Ensemble Combination                                           │
│   ────────────────────────────                                           │
│   ensemble_proba = 0.6 * xgb_proba + 0.4 * lstm_proba                   │
│                  = [0.138, 0.57, 0.292]                                  │
│                                                                          │
│   risk_score = argmax(ensemble_proba) = MEDIUM                           │
│   confidence = max(ensemble_proba) = 57%                                 │
│                                                                          │
│   STEP 10: Output                                                        │
│   ───────────────                                                        │
│   {                                                                      │
│     "risk_level": "MEDIUM",                                              │
│     "confidence": 0.57,                                                  │
│     "probabilities": {"low": 0.138, "medium": 0.57, "high": 0.292},     │
│     "top_features": ["rmssd", "O1_alpha_power", "lf_hf_ratio"],         │
│     "model_agreement": false  (XGB=MEDIUM, LSTM=MEDIUM ✓)               │
│   }                                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Validation Rules

### 9.1 Packet Validation

```python
def validate_packet(packet: GoldPacket) -> List[str]:
    """
    Returns list of error messages (empty if valid).
    """
    errors = []

    # Structure checks
    if len(packet.eeg) != 8:
        errors.append(f"EEG must have 8 channels, got {len(packet.eeg)}")
    if len(packet.ecg) != 3:
        errors.append(f"ECG must have 3 leads, got {len(packet.ecg)}")

    # Sample count consistency
    eeg_samples = len(packet.eeg[0]) if packet.eeg else 0
    ecg_samples = len(packet.ecg[0]) if packet.ecg else 0
    if eeg_samples != ecg_samples:
        errors.append(f"Sample mismatch: EEG={eeg_samples}, ECG={ecg_samples}")

    # EEG amplitude bounds
    for ch, data in enumerate(packet.eeg):
        for val in data:
            if abs(val) > 1000:
                errors.append(f"EEG ch{ch} value {val}µV exceeds ±1000µV")
                break

    # ECG amplitude bounds
    for lead, data in enumerate(packet.ecg):
        for val in data:
            if abs(val) > 10:
                errors.append(f"ECG lead{lead} value {val}mV exceeds ±10mV")
                break

    # Auxiliary bounds
    if not (0 <= packet.spo2_percent <= 100):
        errors.append(f"SpO2 {packet.spo2_percent}% out of range")
    if not (20 <= packet.temp_celsius <= 45):
        errors.append(f"Temperature {packet.temp_celsius}°C out of range")

    return errors
```

### 9.2 Feature Validation

| Feature | Valid Range | Warning Range | Reject Range |
|---------|-------------|---------------|--------------|
| sdnn | 0-300 ms | < 10 or > 200 ms | < 0 or > 500 ms |
| rmssd | 0-200 ms | < 5 or > 100 ms | < 0 or > 300 ms |
| lf_hf_ratio | 0-10 | < 0.1 or > 5 | < 0 or > 20 |
| alpha_power | 0-1000 µV² | > 500 µV² | > 2000 µV² |

---

## Authors

**Mohd Sarfaraz Faiyaz** and **Vaibhav D. Chandgir**
NYU Tandon School of Engineering
ECE-GY 9953 — Advanced Project
Fall 2025

*Advisor: Dr. Matthew Campisi*

---

*This is an academic demonstration system. Not intended for clinical use.*
