# Firmware Documentation

**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Version:** 1.0.0

---

## Overview

The firmware layer simulates an embedded medical device for acquiring multi-modal physiological signals. Written in ISO C11-compliant code, the architecture mirrors production STM32/nRF52 implementations.

---

## Module Structure

```
firmware/
├── main.c                      Main acquisition loop
├── eeg/
│   └── eeg_sim.c              8-channel EEG simulator
├── ecg/
│   └── ecg_sim.c              3-lead ECG simulator
├── sensors/
│   ├── spo2_sim.c             Pulse oximetry
│   ├── temp_sim.c             Temperature sensor
│   └── accel_sim.c            3-axis accelerometer
└── communication/
    └── ble_stub.c             BLE transmission layer
```

---

## Signal Specifications

### EEG (Electroencephalography)

- **Channels**: 8 (10-20 system subset)
  - Fp1, Fp2: Frontal pole
  - C3, C4: Central
  - T3, T4: Temporal
  - O1, O2: Occipital
- **Sampling Rate**: 250 Hz
- **Amplitude Range**: ±200 µV typical
- **Frequency Content**: 0.5-100 Hz
- **Signal Model**: Composite of frequency bands with physiological noise

### ECG (Electrocardiography)

- **Leads**: 3 (I, II, III)
- **Sampling Rate**: 250 Hz
- **Amplitude Range**: ±3 mV typical
- **PQRST Morphology**: Synthetic waveform with:
  - P-wave: Atrial depolarization
  - QRS Complex: Ventricular depolarization
  - T-wave: Ventricular repolarization
- **Heart Rate Variability**: Random RR interval variation (±50 ms)

### Ancillary Sensors

| Sensor | Range | Resolution |
|--------|-------|------------|
| SpO2 | 90-100% | 1% |
| Temperature | 35.0-38.0°C | 0.1°C |
| Accelerometer | ±2g | 1 mg |

---

## Data Packet Structure

```c
typedef struct {
    uint32_t timestamp_ms;           // 4 bytes
    uint16_t packet_id;              // 2 bytes
    uint8_t device_id;               // 1 byte
    uint8_t status_flags;            // 1 byte
    int16_t eeg_data[8][25];         // 400 bytes (8 channels × 25 samples × 2)
    int16_t ecg_data[3][25];         // 150 bytes (3 leads × 25 samples × 2)
    uint8_t spo2_percent;            // 1 byte
    int16_t temperature_celsius_x10; // 2 bytes
    int16_t accel_x_mg;              // 2 bytes
    int16_t accel_y_mg;              // 2 bytes
    int16_t accel_z_mg;              // 2 bytes
    uint16_t checksum;               // 2 bytes
} __attribute__((packed)) DataPacket;
```

**Total Size**: 569 bytes

---

## Compilation

```bash
gcc -o firmware/neurocardiac_fw \
    firmware/main.c \
    firmware/eeg/eeg_sim.c \
    firmware/ecg/ecg_sim.c \
    firmware/sensors/spo2_sim.c \
    firmware/sensors/temp_sim.c \
    firmware/sensors/accel_sim.c \
    firmware/communication/ble_stub.c \
    -lm -O2
```

---

## Execution

```bash
./firmware/neurocardiac_fw
```

Output:
- Binary data stream to `/tmp/neurocardiac_ble_data.bin`
- Console diagnostics (packet ID, timestamp)

---

## Production Migration Path

For STM32F4 deployment:

1. Replace simulation functions with HAL drivers:
   ```c
   // Initialize ADC for EEG channels
   HAL_ADC_Start_DMA(&hadc1, eeg_buffer, EEG_BUFFER_SIZE);

   // Configure sampling timer at 250 Hz
   HAL_TIM_Base_Start_IT(&htim2);
   ```

2. Replace BLE stub with actual stack:
   ```c
   // nRF52840 BLE notification
   ble_gatts_hvx(conn_handle, &hvx_params);
   ```

3. Implement CRC16 checksum calculation:
   ```c
   uint16_t crc16_ccitt(const uint8_t *data, size_t len);
   ```

---

## Signal Quality Metrics

The `status_flags` byte encodes sensor validity:
- Bit 0: EEG channels valid
- Bit 1: ECG leads valid
- Bit 2: SpO2 sensor valid
- Bit 3: Temperature sensor valid

Value `0x0F` indicates all sensors operational.

---

## Memory Footprint

| Resource | Requirement |
|----------|-------------|
| RAM | 128 KB |
| Flash | 512 KB |
| Stack | 8 KB |

---

**New York University - Advanced Project**
