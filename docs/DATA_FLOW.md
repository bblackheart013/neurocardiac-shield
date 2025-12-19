# NeuroCardiac Shield — Data Flow Specification

**Document Type**: Technical Specification
**Version**: 2.1.0
**Last Updated**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav Devram Chandgir

---

## Table of Contents

1. [Overview](#1-overview)
2. [Binary Packet Format](#2-binary-packet-format)
3. [JSON Schemas](#3-json-schemas)
4. [Data Pipeline Stages](#4-data-pipeline-stages)
5. [Throughput Analysis](#5-throughput-analysis)
6. [Error Handling](#6-error-handling)
7. [Buffering Strategy](#7-buffering-strategy)

---

## 1. Overview

This document specifies the data formats and flow paths in the NeuroCardiac Shield system. Understanding these specifications is essential for evaluating the system's data integrity and for extending or debugging the pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW OVERVIEW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────┐                                                         │
│   │   FIRMWARE    │                                                         │
│   │   (C code)    │                                                         │
│   └───────┬───────┘                                                         │
│           │                                                                  │
│           │  Binary packet (569 bytes)                                      │
│           │  Format: struct PhysiologicalDataPacket                         │
│           │  Rate: 10 Hz                                                     │
│           ▼                                                                  │
│   ┌───────────────┐                                                         │
│   │     FILE      │  /tmp/neurocardiac_ble_data.bin                         │
│   │   (binary)    │                                                         │
│   └───────┬───────┘                                                         │
│           │                                                                  │
│           │  struct.unpack("<IHBBhhhh...")                                  │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                         │
│   │    GATEWAY    │                                                         │
│   │   (Python)    │                                                         │
│   └───────┬───────┘                                                         │
│           │                                                                  │
│           │  HTTP POST with JSON body                                       │
│           │  Content-Type: application/json                                 │
│           │  ~2 KB per request                                              │
│           ▼                                                                  │
│   ┌───────────────┐                                                         │
│   │     API       │  POST /api/v1/ingest                                    │
│   │   (FastAPI)   │                                                         │
│   └───────┬───────┘                                                         │
│           │                                                                  │
│           │  Pydantic validation                                            │
│           │  In-memory buffer (1000 packets)                                │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                         │
│   │   PROCESSING  │  Signal filtering + feature extraction                  │
│   │               │  76 features → ML inference                             │
│   └───────┬───────┘                                                         │
│           │                                                                  │
│           │  WebSocket push (JSON)                                          │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                         │
│   │   DASHBOARD   │  Plotly visualization                                   │
│   │  (Streamlit)  │  1 Hz refresh                                           │
│   └───────────────┘                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Faculty Note — Why Binary First, Then JSON?**
>
> The firmware uses binary encoding for efficiency (569 bytes vs ~2 KB JSON), simulating what a real embedded device would transmit over BLE. The gateway performs the binary-to-JSON conversion to enable standard HTTP/REST tooling and easier debugging. This architecture separates concerns: the firmware focuses on efficient data encoding, while the cloud uses human-readable formats for interoperability.

---

## 2. Binary Packet Format

### 2.1 Packet Structure (C Definition)

```c
/**
 * PhysiologicalDataPacket
 * -----------------------
 * Binary packet transmitted at 10 Hz from firmware to gateway.
 * Total size: 569 bytes (little-endian encoding)
 *
 * Alignment: __attribute__((packed)) to prevent compiler padding
 */
typedef struct __attribute__((packed)) {

    // ═══════════════════════════════════════════════════════════
    // HEADER SECTION (8 bytes)
    // ═══════════════════════════════════════════════════════════

    uint32_t timestamp_ms;      // Offset 0-3:   Milliseconds since boot
    uint16_t packet_id;         // Offset 4-5:   Sequence counter (0-65535)
    uint8_t  device_id;         // Offset 6:     Device identifier
    uint8_t  status_flags;      // Offset 7:     Status bitmap
                                //   Bit 0: Valid data
                                //   Bit 1: Buffer overflow
                                //   Bit 2: Electrode contact issue
                                //   Bit 3: Low battery
                                //   Bits 4-7: Reserved

    // ═══════════════════════════════════════════════════════════
    // EEG DATA SECTION (400 bytes)
    // ═══════════════════════════════════════════════════════════

    int16_t eeg_data[8][25];    // Offset 8-407: 8 channels × 25 samples
                                // Physical value: raw / 10.0 µV
                                // Channel order:
                                //   [0]=Fp1, [1]=Fp2, [2]=C3, [3]=C4
                                //   [4]=T3,  [5]=T4,  [6]=O1, [7]=O2

    // ═══════════════════════════════════════════════════════════
    // ECG DATA SECTION (150 bytes)
    // ═══════════════════════════════════════════════════════════

    int16_t ecg_data[3][25];    // Offset 408-557: 3 leads × 25 samples
                                // Physical value: raw / 1000.0 mV
                                // Lead order:
                                //   [0]=Lead I,  [1]=Lead II, [2]=Lead III

    // ═══════════════════════════════════════════════════════════
    // AUXILIARY SENSORS (10 bytes)
    // ═══════════════════════════════════════════════════════════

    uint8_t  spo2_percent;      // Offset 558:     SpO2 percentage (0-100)
    int16_t  temperature_x10;   // Offset 559-560: Temperature × 10 (°C)
                                // Physical value: raw / 10.0 °C
    int16_t  accel_x_mg;        // Offset 561-562: X acceleration (milli-g)
    int16_t  accel_y_mg;        // Offset 563-564: Y acceleration (milli-g)
    int16_t  accel_z_mg;        // Offset 565-566: Z acceleration (milli-g)

    // ═══════════════════════════════════════════════════════════
    // INTEGRITY CHECK (2 bytes)
    // ═══════════════════════════════════════════════════════════

    uint16_t checksum;          // Offset 567-568: CRC16-CCITT
                                // Polynomial: 0x1021
                                // Init: 0xFFFF

} PhysiologicalDataPacket;

// Size verification
_Static_assert(sizeof(PhysiologicalDataPacket) == 569,
               "Packet size must be exactly 569 bytes");
```

### 2.2 Byte Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PACKET BYTE MAP (569 BYTES)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Byte Offset    Field              Type       Size    Description          │
│   ───────────    ─────              ────       ────    ───────────          │
│                                                                              │
│   HEADER (8 bytes)                                                          │
│   0-3            timestamp_ms       uint32     4       ms since boot        │
│   4-5            packet_id          uint16     2       sequence number      │
│   6              device_id          uint8      1       device ID            │
│   7              status_flags       uint8      1       status bits          │
│                                                                              │
│   EEG (400 bytes)                                                           │
│   8-57           eeg_data[0]        int16[25]  50      Fp1 samples          │
│   58-107         eeg_data[1]        int16[25]  50      Fp2 samples          │
│   108-157        eeg_data[2]        int16[25]  50      C3 samples           │
│   158-207        eeg_data[3]        int16[25]  50      C4 samples           │
│   208-257        eeg_data[4]        int16[25]  50      T3 samples           │
│   258-307        eeg_data[5]        int16[25]  50      T4 samples           │
│   308-357        eeg_data[6]        int16[25]  50      O1 samples           │
│   358-407        eeg_data[7]        int16[25]  50      O2 samples           │
│                                                                              │
│   ECG (150 bytes)                                                           │
│   408-457        ecg_data[0]        int16[25]  50      Lead I samples       │
│   458-507        ecg_data[1]        int16[25]  50      Lead II samples      │
│   508-557        ecg_data[2]        int16[25]  50      Lead III samples     │
│                                                                              │
│   VITALS (10 bytes)                                                         │
│   558            spo2_percent       uint8      1       SpO2 %               │
│   559-560        temperature_x10    int16      2       Temp × 10            │
│   561-562        accel_x_mg         int16      2       X accel              │
│   563-564        accel_y_mg         int16      2       Y accel              │
│   565-566        accel_z_mg         int16      2       Z accel              │
│                                                                              │
│   CHECKSUM (2 bytes)                                                        │
│   567-568        checksum           uint16     2       CRC16-CCITT          │
│                                                                              │
│   TOTAL: 8 + 400 + 150 + 10 + 2 = 569 bytes per packet                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Python Unpacking

```python
import struct

PACKET_FORMAT = "<"      # Little-endian
PACKET_FORMAT += "I"     # timestamp_ms (uint32)
PACKET_FORMAT += "H"     # packet_id (uint16)
PACKET_FORMAT += "B"     # device_id (uint8)
PACKET_FORMAT += "B"     # status_flags (uint8)
PACKET_FORMAT += "200h"  # eeg_data (8×25 int16)
PACKET_FORMAT += "75h"   # ecg_data (3×25 int16)
PACKET_FORMAT += "B"     # spo2_percent (uint8)
PACKET_FORMAT += "h"     # temperature_x10 (int16)
PACKET_FORMAT += "hhh"   # accel_x/y/z (int16×3)
PACKET_FORMAT += "H"     # checksum (uint16)

PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # = 569
```

---

## 3. JSON Schemas

### 3.1 Ingest Request Schema

**Endpoint**: `POST /api/v1/ingest`

```json
{
  "timestamp_ms": 1234567890,
  "packet_id": 42,
  "device_id": 1,
  "status_flags": 0,
  "eeg_data": [
    [102, -45, 78, ...],
    [98, -32, 65, ...],
    ...
  ],
  "ecg_data": [
    [150, 890, 320, ...],
    [180, 920, 350, ...],
    [160, 900, 330, ...]
  ],
  "spo2_percent": 98,
  "temperature_x10": 368,
  "accel_x_mg": 12,
  "accel_y_mg": -5,
  "accel_z_mg": 985
}
```

### 3.2 Inference Response Schema

**Endpoint**: `POST /api/v1/inference`

```json
{
  "risk_score": 0.42,
  "risk_category": "MEDIUM",
  "confidence": 0.78,
  "probabilities": {
    "LOW": 0.25,
    "MEDIUM": 0.45,
    "HIGH": 0.30
  },
  "model_breakdown": {
    "xgboost": {
      "score": 0.38,
      "category": "LOW",
      "probabilities": [0.35, 0.42, 0.23]
    },
    "lstm": {
      "score": 0.48,
      "category": "MEDIUM",
      "probabilities": [0.15, 0.50, 0.35]
    }
  },
  "model_agreement": false,
  "explanations": {
    "top_contributors": [
      {"feature": "lf_hf_ratio", "importance": 0.042, "value": 2.1}
    ],
    "interpretation_notes": [
      "LF/HF ratio within normal range"
    ]
  }
}
```

---

## 4. Data Pipeline Stages

### 4.1 Stage-by-Stage Transformation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA TRANSFORMATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: Signal Generation (firmware/)                                     │
│  ─────────────────────────────────────                                      │
│  Input:  None (synthetic generation)                                        │
│  Output: Raw int16 arrays                                                   │
│  EEG: Sum of band oscillations + pink noise + artifacts                     │
│  ECG: PQRST morphology + HRV modulation + noise                            │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 2: Binary Encoding (firmware/main.c)                                 │
│  ──────────────────────────────────────────                                 │
│  Input:  Raw int16 arrays                                                   │
│  Output: 569-byte packed struct                                             │
│  Action: memcpy + CRC16 checksum                                            │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 3: File Transport (simulated BLE)                                    │
│  ─────────────────────────────────────────                                  │
│  Input:  569-byte struct                                                    │
│  Output: Same (file is transparent transport)                               │
│  Action: fwrite to /tmp/neurocardiac_ble_data.bin                          │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 4: Binary Parsing (cloud/ble_gateway.py)                             │
│  ───────────────────────────────────────────────                            │
│  Input:  569-byte binary                                                    │
│  Output: Python dict with raw values                                        │
│  Action: struct.unpack + checksum verification                              │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 5: JSON Encoding + HTTP POST                                         │
│  ─────────────────────────────────                                          │
│  Input:  Python dict                                                        │
│  Output: JSON over HTTP (~2 KB)                                             │
│  Action: requests.post(API_URL, json=packet_dict)                           │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 6: Validation (cloud/api/server.py)                                  │
│  ─────────────────────────────────────────                                  │
│  Input:  JSON request body                                                  │
│  Output: Validated Pydantic model                                           │
│  Action: PhysiologicalPacket(**request_json)                                │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 7: Physical Unit Conversion                                          │
│  ──────────────────────────────────                                         │
│  Input:  Raw int16 values                                                   │
│  Output: Physical float values                                              │
│  Action: eeg_µV = raw / 10.0, ecg_mV = raw / 1000.0                        │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 8: Signal Filtering                                                  │
│  ───────────────────────────                                                │
│  Input:  Physical float arrays                                              │
│  Output: Filtered float arrays                                              │
│  Action: Butterworth bandpass + notch (60 Hz)                               │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 9: Feature Extraction                                                │
│  ────────────────────────────                                               │
│  Input:  Filtered arrays (~10 sec buffer)                                   │
│  Output: 76-element feature vector                                          │
│  Action: Welch PSD, R-peak detection, HRV computation                       │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  STAGE 10: ML Inference                                                     │
│  ───────────────────────                                                    │
│  Input:  76 features + 250×11 sequence                                      │
│  Output: Risk prediction + explanations                                     │
│  Action: XGBoost + LSTM ensemble (60/40)                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Throughput Analysis

### 5.1 Bandwidth Calculations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THROUGHPUT CALCULATIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SIGNAL DATA RATE                                                           │
│  ─────────────────                                                          │
│  Sampling: 250 Hz                                                           │
│  EEG:  8 channels × 250 samples/sec × 2 bytes = 4,000 bytes/sec            │
│  ECG:  3 leads    × 250 samples/sec × 2 bytes = 1,500 bytes/sec            │
│  Total signal:                                   5,500 bytes/sec            │
│                                                                              │
│  PACKET RATE                                                                │
│  ───────────                                                                │
│  Samples per packet: 25                                                     │
│  Packet rate: 250 Hz / 25 samples = 10 Hz                                  │
│  Packet size: 569 bytes                                                     │
│  Bandwidth:   569 × 10 = 5,690 bytes/sec ≈ 5.6 KB/s                        │
│                                                                              │
│  JSON OVERHEAD                                                              │
│  ─────────────                                                              │
│  Binary packet:   569 bytes                                                 │
│  JSON equivalent: ~2,000 bytes (3.5× expansion)                             │
│  JSON rate:       ~20 KB/s                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Latency Budget

| Stage | Typical Latency | Notes |
|-------|----------------|-------|
| Firmware generation | <1 ms | CPU-bound |
| File write | <1 ms | Buffered I/O |
| Gateway poll | 100 ms | Poll interval |
| JSON encode | 1-2 ms | Python json.dumps |
| HTTP POST | 5-10 ms | Local loopback |
| Pydantic validation | 1-2 ms | Schema checking |
| Signal processing | 10-20 ms | Per inference |
| ML inference | 50-100 ms | XGBoost + LSTM |
| WebSocket push | 1-2 ms | Per message |
| **Total (worst case)** | **~200 ms** | End-to-end |

> **Faculty Note — Latency in Medical Devices**
>
> A 200 ms end-to-end latency is acceptable for this monitoring application where alerts are informational rather than safety-critical. Real-time defibrillators or pacemakers require sub-millisecond latency and would never use HTTP/JSON. This system prioritizes implementation simplicity over latency optimization—an appropriate trade-off for academic purposes.

---

## 6. Error Handling

### 6.1 Error Categories

| Error Type | Detection | Handling |
|------------|-----------|----------|
| Checksum mismatch | CRC16 validation | Packet discarded, counter incremented |
| Malformed binary | struct.unpack failure | Gateway logs error, continues |
| Invalid JSON | Pydantic ValidationError | API returns 422, logs details |
| Buffer overflow | Packet count > 1000 | Oldest packets evicted (ring buffer) |
| ML model load failure | Exception on init | Fallback to simulated response |

### 6.2 HTTP Status Codes

| Code | Meaning | Response |
|------|---------|----------|
| 200 | Success | Data accepted |
| 400 | Bad Request | Malformed JSON |
| 422 | Validation Error | Schema mismatch |
| 500 | Server Error | Internal failure |
| 503 | Service Unavailable | ML models not loaded |

---

## 7. Buffering Strategy

### 7.1 In-Memory Ring Buffer

```python
from collections import deque
from threading import Lock

class PacketBuffer:
    """Thread-safe ring buffer for physiological packets."""

    def __init__(self, max_size: int = 1000):
        self._buffer = deque(maxlen=max_size)
        self._lock = Lock()

    def append(self, packet: dict) -> None:
        with self._lock:
            self._buffer.append(packet)

    def get_last_n(self, n: int) -> list:
        with self._lock:
            return list(self._buffer)[-n:]
```

### 7.2 Buffer Sizing Rationale

| Use Case | Required Data | Buffer Needed |
|----------|--------------|---------------|
| EEG feature extraction | 2 seconds | 20 packets |
| HRV analysis (reliable) | 60 seconds | 600 packets |
| HRV analysis (minimum) | 10 seconds | 100 packets |
| Dashboard display | 5 seconds | 50 packets |
| ML inference window | 1 second | 10 packets |

Current buffer of 1000 packets (100 seconds) provides comfortable margin for all use cases with memory cost of ~1 MB.

---

**End of Document**
