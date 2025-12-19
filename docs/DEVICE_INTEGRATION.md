# NeuroCardiac Shield — Device Integration Guide

**Document Type**: Hardware Integration Specification
**Version**: 2.0.0
**Last Updated**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir

---

## Table of Contents

1. [Overview](#1-overview)
2. [Adapter Architecture](#2-adapter-architecture)
3. [Quick Start](#3-quick-start)
4. [Simulated Adapter](#4-simulated-adapter)
5. [BLE Adapter](#5-ble-adapter)
6. [Serial Adapter](#6-serial-adapter)
7. [Gold Schema Specification](#7-gold-schema-specification)
8. [Firmware Requirements](#8-firmware-requirements)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview

The NeuroCardiac Shield system is designed with a **pluggable device adapter layer** that allows seamless switching between:

- **Simulated devices** — For development and testing
- **BLE wearables** — For wireless production devices
- **Serial microcontrollers** — For wired prototyping

All adapters convert their native data format into a unified **Gold Schema**, ensuring the processing pipeline remains unchanged regardless of data source.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEVICE ADAPTER ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │
│   │  SIMULATED    │  │     BLE       │  │    SERIAL     │              │
│   │   ADAPTER     │  │   ADAPTER     │  │   ADAPTER     │              │
│   │               │  │               │  │               │              │
│   │ Synthetic     │  │ Bluetooth LE  │  │ UART/USB      │              │
│   │ Signal Gen    │  │ Connection    │  │ Connection    │              │
│   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘              │
│           │                  │                  │                       │
│           └──────────────────┼──────────────────┘                       │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │    GOLD SCHEMA      │                              │
│                    │   (GoldPacket)      │                              │
│                    │                     │                              │
│                    │ Unified data format │                              │
│                    │ for all adapters    │                              │
│                    └──────────┬──────────┘                              │
│                               │                                         │
│                               ▼                                         │
│                    ┌─────────────────────┐                              │
│                    │  PROCESSING PIPELINE │                              │
│                    │  (Signal → ML → UI)  │                              │
│                    └─────────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Adapter Architecture

### 2.1 Base Interface

All adapters implement the `DeviceAdapter` interface:

```python
from cloud.device_adapters import DeviceAdapter

class DeviceAdapter(ABC):
    def connect(self) -> bool:
        """Establish connection to device."""

    def start_stream(self) -> None:
        """Begin data streaming."""

    def read_packet(self, timeout_ms: int = 1000) -> Optional[GoldPacket]:
        """Read next data packet."""

    def stop(self) -> None:
        """Stop streaming and disconnect."""

    def metadata(self) -> AdapterMetadata:
        """Return device metadata."""
```

### 2.2 Lifecycle

```
┌───────────────────────────────────────────────────────────────────────┐
│                        ADAPTER LIFECYCLE                               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   1. CREATE          2. CONNECT         3. STREAM         4. STOP     │
│   ─────────          ─────────          ────────          ──────      │
│                                                                        │
│   adapter =          adapter.           adapter.          adapter.    │
│   MyAdapter()        connect()          start_stream()    stop()      │
│       │                  │                  │                │        │
│       ▼                  ▼                  ▼                ▼        │
│   ┌────────┐        ┌────────┐         ┌────────┐      ┌────────┐    │
│   │INITIAL │───────▶│CONNECTED│────────▶│STREAMING│─────▶│ STOPPED│   │
│   └────────┘        └────────┘         └────────┘      └────────┘    │
│                                             │                         │
│                                             ▼                         │
│                                     packet = adapter.                 │
│                                       read_packet()                   │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start

### 3.1 Using Simulated Adapter (Default)

```python
from cloud.device_adapters import get_adapter

# Create simulated adapter
adapter = get_adapter('simulated', seed=42, heart_rate_bpm=72)

# Connect and stream
adapter.connect()
adapter.start_stream()

# Read packets
for _ in range(100):
    packet = adapter.read_packet()
    if packet:
        print(f"EEG shape: {len(packet.eeg)}x{len(packet.eeg[0])}")
        print(f"ECG shape: {len(packet.ecg)}x{len(packet.ecg[0])}")

adapter.stop()
```

### 3.2 Using Context Manager

```python
from cloud.device_adapters import get_adapter

with get_adapter('simulated') as adapter:
    for _ in range(100):
        packet = adapter.read_packet()
        process(packet)
# Automatically stops on exit
```

---

## 4. Simulated Adapter

The `SimulatedAdapter` generates scientifically-grounded synthetic signals.

### 4.1 Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for reproducibility |
| `sample_rate_hz` | float | 250.0 | Sampling rate in Hz |
| `packet_rate_hz` | float | 10.0 | Packet output rate in Hz |
| `heart_rate_bpm` | float | 72.0 | Simulated heart rate |
| `hrv_state` | str | 'normal' | One of: 'normal', 'stressed', 'relaxed' |
| `device_id` | str | 'simulated-001' | Device identifier |

### 4.2 Example

```python
from cloud.device_adapters import SimulatedAdapter

adapter = SimulatedAdapter(
    seed=42,
    heart_rate_bpm=80,
    hrv_state='stressed'
)

adapter.connect()
adapter.start_stream()

packet = adapter.read_packet()
print(f"SpO2: {packet.spo2_percent}%")
print(f"Temp: {packet.temp_celsius}°C")
```

### 4.3 Scientific Basis

The simulated signals are grounded in peer-reviewed literature:

| Signal | Model | Reference |
|--------|-------|-----------|
| EEG | Multi-band oscillation with 1/f noise | Nunez & Srinivasan (2006), He et al. (2010) |
| ECG | Gaussian-modulated PQRST morphology | McSharry et al. (2003) |
| HRV | LF/HF power ratio modulation | Task Force (1996) |

---

## 5. BLE Adapter

**Status**: Interface defined, implementation requires hardware.

The `BLEAdapter` connects to Bluetooth Low Energy wearable devices.

### 5.1 Prerequisites

```bash
pip install bleak
```

### 5.2 Configuration

```python
from cloud.device_adapters import BLEAdapter, BLEDeviceConfig

config = BLEDeviceConfig(
    device_address="AA:BB:CC:DD:EE:FF",
    device_name="NeuroCardiac",
    service_uuid="12345678-1234-5678-1234-56789abcdef0",
    data_char_uuid="12345678-1234-5678-1234-56789abcdef1",
    mtu_size=512
)

adapter = BLEAdapter(config)
```

### 5.3 BLE Packet Format

The firmware sends 569-byte binary packets over BLE:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    BLE PACKET FORMAT (569 BYTES)                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   HEADER (8 bytes)                                                      │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ timestamp_ms  [uint32]  4 bytes   Milliseconds since boot      │   │
│   │ packet_id     [uint16]  2 bytes   Sequence counter             │   │
│   │ device_id     [uint8]   1 byte    Device identifier            │   │
│   │ status_flags  [uint8]   1 byte    Quality/status bitmap        │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   EEG DATA (400 bytes)                                                  │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ eeg[8][25]    [int16]   400 bytes  8 channels × 25 samples     │   │
│   │ Channels: Fp1, Fp2, C3, C4, T3, T4, O1, O2                     │   │
│   │ Scaling: raw_value / 10.0 = microvolts (µV)                    │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ECG DATA (150 bytes)                                                  │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ ecg[3][25]    [int16]   150 bytes  3 leads × 25 samples        │   │
│   │ Leads: I, II, III                                               │   │
│   │ Scaling: raw_value / 1000.0 = millivolts (mV)                  │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   VITALS (10 bytes)                                                     │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ spo2_percent    [uint8]   1 byte    SpO2 percentage (0-100)    │   │
│   │ temperature_x10 [int16]   2 bytes   Temp × 10 (°C)             │   │
│   │ accel_x_mg      [int16]   2 bytes   Acceleration X (milli-g)   │   │
│   │ accel_y_mg      [int16]   2 bytes   Acceleration Y (milli-g)   │   │
│   │ accel_z_mg      [int16]   2 bytes   Acceleration Z (milli-g)   │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   CHECKSUM (2 bytes)                                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ crc16           [uint16]  2 bytes   CRC16-CCITT                │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   TOTAL: 8 + 400 + 150 + 10 + 2 = 569 bytes                            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.4 MTU Considerations

BLE has a Maximum Transmission Unit (MTU) that limits packet size:

| MTU Size | Strategy |
|----------|----------|
| 512+ bytes | Send packet in single notification |
| 256 bytes | Split into 3 fragments with sequence numbers |
| 128 bytes | Split into 5 fragments |
| < 128 bytes | Not recommended (high overhead) |

Negotiate MTU during connection:
```python
# Bleak will negotiate automatically
# Request 512 bytes for optimal performance
```

---

## 6. Serial Adapter

The `SerialAdapter` connects to microcontrollers via UART/USB.

### 6.1 Prerequisites

```bash
pip install pyserial
```

### 6.2 Finding Your Port

```python
from cloud.device_adapters import SerialAdapter

# List available ports
ports = SerialAdapter.list_ports()
for p in ports:
    print(f"{p['port']}: {p['description']}")
```

Common port names:
- **Linux**: `/dev/ttyUSB0`, `/dev/ttyACM0`
- **macOS**: `/dev/cu.usbserial-*`, `/dev/cu.usbmodem*`
- **Windows**: `COM3`, `COM4`, etc.

### 6.3 Configuration

```python
from cloud.device_adapters import SerialAdapter, SerialConfig

config = SerialConfig(
    port="/dev/ttyUSB0",
    baudrate=115200,
    packet_format='binary'  # or 'json'
)

adapter = SerialAdapter(config)
adapter.connect()
adapter.start_stream()
```

### 6.4 Supported Formats

**Binary Format (569 bytes)**:
Same as BLE format, framed with sync bytes:
```
[0xAA][569 bytes payload][0x55]
```

**JSON Format (variable)**:
Newline-delimited JSON objects:
```json
{"ts":1234567890,"seq":42,"eeg":[[...],[...],...],"ecg":[[...],[...],[...]],"spo2":98.5,"temp":37.2,"accel":[0.01,-0.02,0.98]}
```

---

## 7. Gold Schema Specification

All adapters output data in the `GoldPacket` format:

```python
@dataclass
class GoldPacket:
    # Timing
    timestamp_ms: int          # Unix timestamp in milliseconds
    device_id: str             # Unique device identifier
    packet_seq: int            # Sequence number
    sample_rate_hz: float      # Sampling rate (Hz)

    # Physiological data
    eeg: List[List[float]]     # [8 channels][N samples] in µV
    ecg: List[List[float]]     # [3 leads][N samples] in mV

    # Auxiliary sensors
    spo2_percent: float        # 0-100%
    temp_celsius: float        # °C
    accel_xyz_g: List[float]   # [x, y, z] in g-force

    # Quality
    quality_flags: PacketQualityFlags
```

### 7.1 Quality Flags

```python
class PacketQualityFlags(IntFlag):
    NONE = 0
    EEG_SATURATION = 1 << 0      # EEG channel saturated
    ECG_SATURATION = 1 << 1      # ECG signal saturated
    EEG_LEAD_OFF = 1 << 2        # EEG electrode disconnected
    ECG_LEAD_OFF = 1 << 3        # ECG electrode disconnected
    LOW_BATTERY = 1 << 4         # Battery < 20%
    MOTION_ARTIFACT = 1 << 5     # High accelerometer activity
    POOR_SIGNAL = 1 << 6         # General quality warning
    CHECKSUM_INVALID = 1 << 7    # Checksum failed
    SYNTHETIC_DATA = 1 << 8      # Data is simulated
```

---

## 8. Firmware Requirements

If you are developing custom firmware, it must produce packets matching the binary specification.

### 8.1 Arduino/ESP32 Example

```cpp
struct __attribute__((packed)) PhysioPacket {
    // Header (8 bytes)
    uint32_t timestamp_ms;
    uint16_t packet_id;
    uint8_t device_id;
    uint8_t status_flags;

    // EEG (400 bytes)
    int16_t eeg[8][25];

    // ECG (150 bytes)
    int16_t ecg[3][25];

    // Vitals (10 bytes)
    uint8_t spo2;
    int16_t temp_x10;
    int16_t accel[3];

    // Checksum (2 bytes)
    uint16_t crc16;
};

void send_packet(PhysioPacket& pkt) {
    Serial.write(0xAA);  // Sync byte
    Serial.write((uint8_t*)&pkt, sizeof(pkt));
    Serial.write(0x55);  // End byte
}
```

### 8.2 Scaling Requirements

| Sensor | Raw Type | Scaling | Units |
|--------|----------|---------|-------|
| EEG | int16 | ÷ 10.0 | µV |
| ECG | int16 | ÷ 1000.0 | mV |
| Temperature | int16 | ÷ 10.0 | °C |
| Accelerometer | int16 | ÷ 1000.0 | g |
| SpO2 | uint8 | direct | % |

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No packets received | Wrong port/address | Verify connection parameters |
| Corrupted data | Baud rate mismatch | Match firmware baud rate |
| Packets timing out | Wrong packet format | Check binary vs JSON setting |
| Quality flags set | Electrode issues | Check sensor connections |
| BLE not connecting | Adapter not installed | Install `bleak` library |
| Serial port busy | Port in use | Close other applications |

### 9.2 Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

adapter = get_adapter('serial', port='/dev/ttyUSB0')
adapter.connect()
```

---

## Authors

**Mohd Sarfaraz Faiyaz** and **Vaibhav D. Chandgir**
NYU Tandon School of Engineering
ECE-GY 9953 — Advanced Project
Fall 2025

*Advisor: Dr. Matthew Campisi*

---

*This is an academic demonstration system. All physiological data in simulation mode is computationally generated. Not intended for clinical use.*
