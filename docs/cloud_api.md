# Cloud API Documentation

**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Version:** 1.0.0

---

## Overview

The cloud backend provides a FastAPI-based asynchronous server for real-time data ingestion, signal processing orchestration, and machine learning inference delivery.

---

## Technology Stack

- **Framework**: FastAPI 0.104+
- **Runtime**: Uvicorn ASGI server
- **Language**: Python 3.9+
- **Validation**: Pydantic models
- **Async Support**: Full asyncio integration

---

## API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00.000000",
  "version": "1.0.0",
  "active_devices": 1,
  "active_ws_clients": 0
}
```

---

### Data Ingestion

```http
POST /api/v1/ingest
```

**Request Body:**
```json
{
  "timestamp_ms": 1000,
  "packet_id": 42,
  "device_id": 1,
  "status_flags": 15,
  "eeg_data": [[...], ...],
  "ecg_data": [[...], ...],
  "spo2_percent": 98,
  "temperature_celsius_x10": 368,
  "accel_x_mg": 10,
  "accel_y_mg": -5,
  "accel_z_mg": 980,
  "checksum": 12345
}
```

**Response (201 Created):**
```json
{
  "status": "accepted",
  "device_id": 1,
  "packet_id": 42,
  "timestamp": "2025-01-15T10:30:01.000000"
}
```

---

### Device Status

```http
GET /api/v1/device/{device_id}/status
```

**Response:**
```json
{
  "device_id": 1,
  "status": "online",
  "last_packet_id": 100,
  "last_update": "2025-01-15T10:30:00.000000",
  "packet_rate_hz": 10,
  "signal_quality": 1.0,
  "vitals": {
    "spo2": 98,
    "temperature_c": 36.8,
    "heart_rate_bpm": null
  }
}
```

---

### ML Inference

```http
POST /api/v1/inference
```

**Request Body:**
```json
{
  "device_id": 1,
  "window_size_seconds": 10
}
```

**Response:**
```json
{
  "device_id": 1,
  "timestamp": "2025-01-15T10:30:00.000000",
  "risk_score": 0.35,
  "risk_category": "LOW",
  "hrv_metrics": {
    "sdnn_ms": 45.2,
    "rmssd_ms": 38.1,
    "lf_hf_ratio": 1.2
  },
  "eeg_features": {
    "alpha_power": 0.42,
    "beta_power": 0.28,
    "theta_power": 0.18,
    "entropy": 0.76
  },
  "model_confidence": 0.89
}
```

---

### WebSocket Streaming

```http
WS /ws/stream
```

Real-time data streaming to dashboard clients.

---

## Signal Processing Pipeline

### Preprocessing Module

Located in `cloud/signal_processing/preprocess.py`:

1. **Bandpass Filtering**
   - EEG: 0.5-50 Hz (Butterworth, order 4)
   - ECG: 0.5-40 Hz (diagnostic quality)

2. **Notch Filtering**
   - 60 Hz powerline noise removal

3. **R-Peak Detection**
   - Pan-Tompkins algorithm for QRS detection

4. **Signal Quality Assessment**
   - Variance analysis
   - Clipping detection
   - SNR estimation

### Feature Extraction Module

Located in `cloud/signal_processing/features.py`:

**EEG Features (64 dimensions):**
- Band powers: Delta, Theta, Alpha, Beta, Gamma
- Ratios: Alpha/Beta, Theta/Alpha
- Spectral entropy per channel
- Inter-channel coherence

**HRV Features (7 dimensions):**
- Time-domain: Mean HR, SDNN, RMSSD, pNN50
- Frequency-domain: LF power, HF power, LF/HF ratio

**ECG Morphology (3 dimensions):**
- Mean RR interval
- QRS duration
- QRS amplitude

**Total**: 74 features per analysis window

---

## Data Storage

In-memory buffer with circular storage:
- Maximum 1000 packets per device (~100 seconds at 10 Hz)
- Automatic pruning of oldest data

Production deployment should integrate:
- TimescaleDB for time-series persistence
- Redis for real-time caching
- PostgreSQL for device metadata

---

## CORS Configuration

Development mode allows all origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Production requires restricted origin policies.

---

## Running the Server

Development:
```bash
cd cloud
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Production:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker cloud.api.server:app
```

---

## Error Handling

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created (ingestion) |
| 404 | Device not found |
| 422 | Validation error |
| 500 | Internal server error |

---

## Security Considerations

- Input validation via Pydantic
- Rate limiting recommended for production
- Authentication layer required (JWT/OAuth)
- HTTPS mandatory for PHI transmission

---

**New York University - Advanced Project**
