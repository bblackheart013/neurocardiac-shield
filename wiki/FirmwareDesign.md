# Firmware Design

## Overview

Embedded C code simulating medical device signal acquisition.

## Module Structure

```
firmware/
├── main.c              # Main loop
├── eeg/eeg_sim.c       # EEG simulator
├── ecg/ecg_sim.c       # ECG simulator
├── sensors/
│   ├── spo2_sim.c      # SpO2 sensor
│   ├── temp_sim.c      # Temperature
│   └── accel_sim.c     # Accelerometer
└── communication/
    └── ble_stub.c      # BLE output
```

## Signal Specifications

### EEG
- 8 channels (Fp1, Fp2, C3, C4, T3, T4, O1, O2)
- 250 Hz sampling
- ±200 µV range
- Frequency bands: Delta, Theta, Alpha, Beta, Gamma

### ECG
- 3 leads (I, II, III)
- 250 Hz sampling
- ±3 mV range
- PQRST morphology

### Ancillary
- SpO2: 90-100%
- Temperature: 35-38°C
- Accelerometer: ±2g

## Packet Format

```c
struct Packet {
    uint32_t timestamp_ms;      // 4 bytes
    uint16_t packet_id;         // 2 bytes
    uint8_t device_id;          // 1 byte
    uint8_t status_flags;       // 1 byte
    int16_t eeg_data[8][25];    // 400 bytes
    int16_t ecg_data[3][25];    // 150 bytes
    uint8_t spo2_percent;       // 1 byte
    int16_t temperature_x10;    // 2 bytes
    int16_t accel_xyz[3];       // 6 bytes
    uint16_t checksum;          // 2 bytes
};
// Total: 569 bytes
```

## Compilation

```bash
gcc -o firmware/neurocardiac_fw \
    firmware/main.c \
    firmware/eeg/*.c \
    firmware/ecg/*.c \
    firmware/sensors/*.c \
    firmware/communication/*.c \
    -lm -O2
```

## Production Migration

Replace simulation functions with:
- STM32 HAL drivers
- nRF52 BLE stack
- Hardware ADC/DMA
- Timer interrupts
