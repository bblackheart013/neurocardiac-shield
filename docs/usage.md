# NeuroCardiac Shield - Usage Guide

**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Version:** 1.0.0

---

## Quick Start

### Prerequisites

Ensure the following are installed:
- Python 3.9 or higher
- GCC compiler
- pip package manager

### Installation

```bash
chmod +x setup.sh
./setup.sh
```

This script:
1. Creates isolated Python virtual environments for each component
2. Installs all required dependencies
3. Trains the machine learning models
4. Compiles the firmware simulator

### Running the System

```bash
./run_complete_demo.sh
```

This launches all components:
- **API Server**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **Firmware Simulator**: Generates physiological data
- **BLE Gateway**: Bridges firmware to cloud

### Stopping the System

Press `CTRL+C` in the terminal to gracefully stop all components.

---

## Component-Specific Usage

### API Server

Access the interactive API documentation:
```
http://localhost:8000/docs
```

Example API calls:

```bash
# Health check
curl http://localhost:8000/health

# Device status
curl http://localhost:8000/api/v1/device/1/status

# ML inference
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"device_id": 1}'
```

### Dashboard

Navigate to http://localhost:8501 in your web browser.

Features:
- Real-time EEG waveform display (8 channels)
- ECG signal visualization with PQRST morphology
- Risk score gauge indicator
- HRV metrics panel
- EEG frequency band power distribution

Configuration options:
- Time window: 1-10 seconds
- Auto-refresh toggle
- Channel selection

### Firmware Simulator

The firmware runs automatically when executing `run_complete_demo.sh`.

Manual execution:
```bash
./firmware/neurocardiac_fw
```

Output is written to `/tmp/neurocardiac_ble_data.bin`.

### BLE Gateway

The gateway reads firmware output and forwards it to the API.

Manual execution:
```bash
cd cloud
source venv/bin/activate
python ble_gateway.py
```

---

## Configuration

Edit `config/config.yaml` to modify system parameters:

```yaml
sampling:
  rate_hz: 250
  packet_rate_hz: 10

api:
  host: 0.0.0.0
  port: 8000

dashboard:
  port: 8501
  auto_refresh: true
```

---

## Troubleshooting

### API Server Not Starting

1. Check if port 8000 is in use:
   ```bash
   lsof -i :8000
   ```

2. Kill existing processes:
   ```bash
   pkill -f uvicorn
   ```

### Dashboard Not Loading

1. Verify Streamlit is running:
   ```bash
   pgrep -f streamlit
   ```

2. Check port 8501 availability:
   ```bash
   lsof -i :8501
   ```

### No Data in Dashboard

1. Verify firmware is generating data:
   ```bash
   ls -la /tmp/neurocardiac_ble_data.bin
   ```

2. Check BLE Gateway is running:
   ```bash
   pgrep -f ble_gateway
   ```

3. Verify API is receiving packets:
   ```bash
   curl http://localhost:8000/api/v1/device/1/status
   ```

---

## Data Flow

1. Firmware generates physiological signals at 250 Hz
2. Data is packetized (25 samples per packet, 10 Hz)
3. BLE Gateway reads binary packets
4. Gateway POSTs JSON to API `/api/v1/ingest`
5. API stores data in memory buffer
6. Dashboard fetches via REST API
7. ML inference triggered on demand

---

## Performance Metrics

| Metric | Target | Typical |
|--------|--------|---------|
| Sampling Rate | 250 Hz | 250 Hz |
| Packet Latency | <200 ms | ~150 ms |
| Inference Time | <100 ms | ~80 ms |
| Dashboard Refresh | <500 ms | ~400 ms |

---

## Security Considerations

- API runs on localhost by default
- No authentication in development mode
- Production deployment requires:
  - TLS/SSL encryption
  - JWT authentication
  - Rate limiting
  - Input validation

---

**New York University - Advanced Project**
