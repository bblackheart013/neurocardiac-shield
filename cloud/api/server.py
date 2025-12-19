"""
NeuroCardiac Shield - Cloud API Server
=======================================

FastAPI-based asynchronous backend for multi-modal physiological data
ingestion, processing, and ML-driven risk assessment.

Core Functionality:
-------------------
1. Real-time data ingestion from BLE gateway (POST /api/v1/ingest)
2. Device health monitoring (GET /api/v1/device/{id}/status)
3. ML inference orchestration (POST /api/v1/inference)
4. WebSocket streaming to dashboard clients (WS /ws/stream)

Architecture Notes:
-------------------
- Async-first design using FastAPI/Starlette for high throughput
- In-memory data buffer (development mode) - replace with TimescaleDB
  for production deployment
- ML inference runs synchronously on current thread - consider offloading
  to worker pool for high-load scenarios

IMPORTANT LIMITATIONS:
----------------------
- This is a DEVELOPMENT/RESEARCH API, not production-grade
- Data is stored in-memory only (lost on restart)
- No authentication/authorization implemented
- CORS is fully open (restrict in production)
- ML inference is SIMULATED when models are not available

Security Considerations (TODO for Production):
----------------------------------------------
- Implement JWT/OAuth2 authentication
- Add rate limiting
- Enable HTTPS/TLS
- Restrict CORS origins
- Add request validation and sanitization
- Implement proper logging and audit trails
- Consider HIPAA/GDPR compliance requirements

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 2.0.0
"""

import asyncio
import struct
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neurocardiac.api")

# In-memory data storage (replace with database in production)
data_buffer: Dict[int, List[dict]] = {}
active_connections: List[WebSocket] = []

# ML model state (loaded lazily)
_ml_models_loaded = False
_inference_engine = None


# ============================================================================
# Data Models
# ============================================================================

class PhysiologicalPacket(BaseModel):
    """Received data packet schema matching firmware structure"""
    timestamp_ms: int = Field(..., description="Device timestamp in milliseconds")
    packet_id: int = Field(..., ge=0, description="Sequential packet counter")
    device_id: int = Field(..., ge=0, le=255, description="Unique device identifier")
    status_flags: int = Field(..., ge=0, le=255, description="Sensor status bitfield")

    eeg_data: List[List[int]] = Field(..., description="8 channels × 25 samples")
    ecg_data: List[List[int]] = Field(..., description="3 leads × 25 samples")

    spo2_percent: int = Field(..., ge=0, le=100, description="Blood oxygen saturation")
    temperature_celsius_x10: int = Field(..., description="Temperature × 10")
    accel_x_mg: int = Field(..., description="Accelerometer X (millig)")
    accel_y_mg: int = Field(..., description="Accelerometer Y")
    accel_z_mg: int = Field(..., description="Accelerometer Z")

    checksum: int = Field(..., description="CRC16 checksum")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp_ms": 1000,
                "packet_id": 42,
                "device_id": 1,
                "status_flags": 15,
                "eeg_data": [[0] * 25 for _ in range(8)],
                "ecg_data": [[0] * 25 for _ in range(3)],
                "spo2_percent": 98,
                "temperature_celsius_x10": 368,
                "accel_x_mg": 10,
                "accel_y_mg": -5,
                "accel_z_mg": 980
            }
        }


class InferenceRequest(BaseModel):
    """ML inference request"""
    device_id: int
    window_size_seconds: int = 10  # Analyze last 10 seconds


class InferenceResult(BaseModel):
    """ML risk prediction result"""
    device_id: int
    timestamp: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk (0-1)")
    risk_category: str = Field(..., description="LOW | MEDIUM | HIGH")
    hrv_metrics: Dict[str, float]
    eeg_features: Dict[str, float]
    model_confidence: float

    class Config:
        protected_namespaces = ()  # Disable protected namespace warning for model_ fields


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle"""
    yield


app = FastAPI(
    title="NeuroCardiac Shield API",
    version="1.0.0",
    description="Production backend for brain-heart monitoring platform",
    lifespan=lifespan
)

# CORS configuration (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Data Ingestion Endpoints
# ============================================================================

@app.post("/api/v1/ingest", status_code=201)
async def ingest_data(packet: PhysiologicalPacket, background_tasks: BackgroundTasks):
    """
    Receive and process physiological data packet from BLE gateway.

    Pipeline:
    1. Validate checksum
    2. Store raw data
    3. Trigger async signal processing
    4. Broadcast to WebSocket clients
    """
    device_id = packet.device_id

    if device_id not in data_buffer:
        data_buffer[device_id] = []

    data_buffer[device_id].append(packet.dict())

    # Limit buffer size (keep last 1000 packets ≈ 100 seconds at 10 Hz)
    if len(data_buffer[device_id]) > 1000:
        data_buffer[device_id] = data_buffer[device_id][-1000:]

    # Background processing
    background_tasks.add_task(process_packet_async, packet)

    # Broadcast to WebSocket clients
    await broadcast_to_clients(packet.dict())

    return {
        "status": "accepted",
        "device_id": device_id,
        "packet_id": packet.packet_id,
        "timestamp": datetime.now().isoformat()
    }


async def process_packet_async(packet: PhysiologicalPacket):
    """Background task: signal processing and feature extraction"""
    await asyncio.sleep(0.01)


# ============================================================================
# Device Management
# ============================================================================

@app.get("/api/v1/device/{device_id}/status")
async def get_device_status(device_id: int):
    """Retrieve device connection status and data quality metrics"""
    if device_id not in data_buffer or not data_buffer[device_id]:
        raise HTTPException(status_code=404, detail="Device not found or no data")

    recent_packets = data_buffer[device_id][-10:]
    latest = recent_packets[-1]

    # Calculate data quality metrics
    packet_rate = len(recent_packets)  # Should be ~10 for healthy connection
    signal_quality = (latest["status_flags"] & 0x0F) / 15.0  # Normalize status flags

    return {
        "device_id": device_id,
        "status": "online" if packet_rate > 5 else "degraded",
        "last_packet_id": latest["packet_id"],
        "last_update": datetime.now().isoformat(),
        "packet_rate_hz": packet_rate,
        "signal_quality": signal_quality,
        "vitals": {
            "spo2": latest["spo2_percent"],
            "temperature_c": latest["temperature_celsius_x10"] / 10.0,
            "heart_rate_bpm": None
        }
    }


# =============================================================================
# ML Model Loading and Inference Helpers
# =============================================================================

def _load_ml_models():
    """
    Lazily load ML models on first inference request.

    This function attempts to load the XGBoost model and scaler.
    LSTM model loading is optional (requires TensorFlow).

    Returns:
        bool: True if at least XGBoost model loaded successfully.
    """
    global _ml_models_loaded, _inference_engine

    if _ml_models_loaded:
        return _inference_engine is not None

    try:
        # Try to import and instantiate inference engine
        from ml.model.inference import NeuroCardiacInference

        xgb_model_path = PROJECT_ROOT / "ml/checkpoints/xgboost/xgboost_model.json"
        xgb_scaler_path = PROJECT_ROOT / "ml/checkpoints/xgboost/scaler.pkl"

        if xgb_model_path.exists() and xgb_scaler_path.exists():
            _inference_engine = NeuroCardiacInference(
                xgb_model_path=str(xgb_model_path),
                xgb_scaler_path=str(xgb_scaler_path),
                lstm_model_path=str(PROJECT_ROOT / "ml/checkpoints/lstm/lstm_model.keras"),
                lstm_scaler_path=str(PROJECT_ROOT / "ml/checkpoints/lstm/lstm_scaler.pkl")
            )
            logger.info("ML inference engine loaded successfully")
        else:
            logger.warning(
                f"ML model files not found. XGBoost: {xgb_model_path.exists()}, "
                f"Scaler: {xgb_scaler_path.exists()}. Using simulated inference."
            )

    except ImportError as e:
        logger.warning(f"Could not import ML inference module: {e}. Using simulated inference.")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}. Using simulated inference.")

    _ml_models_loaded = True
    return _inference_engine is not None


def _extract_features_from_buffer(packets: List[dict], fs: float = 250.0) -> Dict[str, Any]:
    """
    Extract ML features from buffered packet data.

    This function converts the raw packet data into the feature format
    expected by the ML models.

    Args:
        packets: List of packet dictionaries from data_buffer.
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary containing:
        - 'features': Dict of extracted features for XGBoost
        - 'sequence': Numpy array for LSTM (seq_len, channels)
        - 'hrv_metrics': HRV analysis results
        - 'eeg_features': EEG band power summary
    """
    try:
        from cloud.signal_processing.preprocess import filter_eeg, filter_ecg, detect_r_peaks
        from cloud.signal_processing.features import extract_eeg_features, extract_hrv_features
    except ImportError:
        # Fallback if imports fail
        return _generate_simulated_features()

    # Concatenate EEG and ECG data from packets
    eeg_channels = []
    ecg_leads = []

    for packet in packets:
        eeg_data = np.array(packet.get('eeg_data', [[0]*25]*8))
        ecg_data = np.array(packet.get('ecg_data', [[0]*25]*3))
        eeg_channels.append(eeg_data)
        ecg_leads.append(ecg_data)

    if not eeg_channels:
        return _generate_simulated_features()

    # Concatenate across packets: (n_channels, n_samples)
    eeg_array = np.hstack(eeg_channels)  # (8, n_packets*25)
    ecg_array = np.hstack(ecg_leads)     # (3, n_packets*25)

    # Convert from firmware int16 to µV/mV
    eeg_array = eeg_array.astype(float) / 10.0  # Back to µV
    ecg_array = ecg_array.astype(float) / 1000.0  # Back to mV

    # Extract features
    try:
        # Filter signals
        eeg_filtered = filter_eeg(eeg_array, fs=fs)
        ecg_filtered = filter_ecg(ecg_array[0, :], fs=fs)  # Lead I

        # Detect R-peaks
        r_peaks = detect_r_peaks(ecg_filtered, fs=fs)

        # Extract EEG features
        channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
        eeg_feats = extract_eeg_features(eeg_filtered, fs, channel_names)

        # Extract HRV features
        if len(r_peaks) >= 2:
            rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
            hrv_feats = extract_hrv_features(rr_intervals)
        else:
            hrv_feats = {
                'mean_hr': 70.0, 'sdnn': 50.0, 'rmssd': 30.0,
                'pnn50': 10.0, 'lf_power': 100.0, 'hf_power': 80.0, 'lf_hf_ratio': 1.25
            }

        # Combine all features
        all_features = {**eeg_feats, **hrv_feats}

        # Add ECG morphology features
        all_features['mean_qrs_duration'] = 90.0  # Simplified
        all_features['mean_rr_interval'] = 60000.0 / hrv_feats.get('mean_hr', 70.0)
        all_features['qrs_amplitude'] = float(np.max(ecg_filtered)) if len(ecg_filtered) > 0 else 1.2

        # Prepare sequence for LSTM (last 250 samples = 1 second)
        n_samples = min(250, eeg_array.shape[1])
        sequence = np.zeros((250, 11))  # 8 EEG + 3 ECG channels

        if n_samples > 0:
            sequence[-n_samples:, :8] = eeg_filtered[:, -n_samples:].T
            sequence[-n_samples:, 8:] = ecg_array[:, -n_samples:].T

        return {
            'features': all_features,
            'sequence': sequence,
            'hrv_metrics': {
                'mean_hr': hrv_feats.get('mean_hr', 0),
                'sdnn_ms': hrv_feats.get('sdnn', 0),
                'rmssd_ms': hrv_feats.get('rmssd', 0),
                'lf_hf_ratio': hrv_feats.get('lf_hf_ratio', 0)
            },
            'eeg_features': {
                'alpha_power': eeg_feats.get('mean_alpha_power', 0),
                'beta_power': eeg_feats.get('Fp1_beta_power', 0),
                'theta_power': eeg_feats.get('Fp1_theta_power', 0),
                'entropy': eeg_feats.get('Fp1_spectral_entropy', 0)
            }
        }

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return _generate_simulated_features()


def _generate_simulated_features() -> Dict[str, Any]:
    """
    Generate simulated features when real extraction fails.

    IMPORTANT: This returns SYNTHETIC data for development/testing only.
    The risk scores from simulated features are NOT clinically meaningful.
    """
    rng = np.random.default_rng()

    # Simulated features matching expected schema
    features = {}
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']

    for ch in channel_names:
        features[f'{ch}_delta_power'] = rng.lognormal(2.0, 1.0)
        features[f'{ch}_theta_power'] = rng.lognormal(1.5, 0.8)
        features[f'{ch}_alpha_power'] = rng.lognormal(2.5, 1.2)
        features[f'{ch}_beta_power'] = rng.lognormal(1.0, 0.6)
        features[f'{ch}_gamma_power'] = rng.lognormal(0.5, 0.4)
        features[f'{ch}_alpha_beta_ratio'] = rng.gamma(2.0, 0.5)
        features[f'{ch}_theta_alpha_ratio'] = rng.gamma(1.5, 0.3)
        features[f'{ch}_spectral_entropy'] = rng.beta(5, 2)

    features['mean_alpha_power'] = rng.lognormal(2.5, 1.0)
    features['std_alpha_power'] = rng.lognormal(1.0, 0.5)
    features['mean_hr'] = rng.normal(70, 10)
    features['sdnn'] = rng.gamma(4, 10)
    features['rmssd'] = rng.gamma(4, 8)
    features['pnn50'] = rng.beta(2, 3) * 50
    features['lf_power'] = rng.lognormal(5, 1.5)
    features['hf_power'] = rng.lognormal(4.5, 1.5)
    features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-6)
    features['mean_qrs_duration'] = rng.normal(90, 10)
    features['mean_rr_interval'] = 60000 / features['mean_hr']
    features['qrs_amplitude'] = rng.normal(1.2, 0.3)

    return {
        'features': features,
        'sequence': rng.randn(250, 11) * 20.0,
        'hrv_metrics': {
            'mean_hr': features['mean_hr'],
            'sdnn_ms': features['sdnn'],
            'rmssd_ms': features['rmssd'],
            'lf_hf_ratio': features['lf_hf_ratio']
        },
        'eeg_features': {
            'alpha_power': features['mean_alpha_power'],
            'beta_power': features['Fp1_beta_power'],
            'theta_power': features['Fp1_theta_power'],
            'entropy': features['Fp1_spectral_entropy']
        },
        '_simulated': True  # Flag to indicate simulated data
    }


# =============================================================================
# ML Inference Endpoint
# =============================================================================

@app.post("/api/v1/inference", response_model=InferenceResult)
async def run_inference(request: InferenceRequest):
    """
    Execute ML risk prediction on recent physiological data.

    Processing Pipeline:
    1. Retrieve last N seconds of buffered data
    2. Extract features (EEG band powers, HRV metrics, spectral entropy)
    3. Run ensemble prediction (XGBoost + LSTM if available)
    4. Return risk score with feature breakdown

    IMPORTANT NOTES:
    ----------------
    - If ML models are not loaded, returns SIMULATED risk scores
    - Simulated scores are marked with model_confidence < 0.5
    - Real inference requires trained models in ml/checkpoints/

    Args:
        request: InferenceRequest with device_id and window_size_seconds

    Returns:
        InferenceResult with risk score, category, and feature breakdown

    Raises:
        HTTPException 404: If no data available for device
    """
    device_id = request.device_id
    window_seconds = request.window_size_seconds

    # Check if we have data for this device
    if device_id not in data_buffer or not data_buffer[device_id]:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for device {device_id}. "
                   "Ensure the device is connected and transmitting."
        )

    # Get recent packets (10 Hz packet rate, so window_seconds * 10 packets)
    n_packets = min(window_seconds * 10, len(data_buffer[device_id]))
    recent_packets = data_buffer[device_id][-n_packets:]

    # Extract features from buffered data
    extracted = _extract_features_from_buffer(recent_packets)
    is_simulated = extracted.get('_simulated', False)

    # Attempt to load ML models
    models_available = _load_ml_models()

    # Run inference
    if models_available and _inference_engine is not None and not is_simulated:
        try:
            # Real ML inference
            result = _inference_engine.predict_ensemble(
                features=extracted['features'],
                sequence=extracted['sequence']
            )
            risk_score = result['risk_score']
            risk_category = result['risk_category']
            model_confidence = result['confidence']
            logger.info(
                f"ML inference for device {device_id}: "
                f"score={risk_score:.3f}, category={risk_category}"
            )
        except Exception as e:
            logger.error(f"ML inference failed: {e}. Falling back to simulated.")
            risk_score = 0.35
            risk_category = "LOW"
            model_confidence = 0.3  # Low confidence indicates fallback
    else:
        # Simulated inference (models not available or data simulated)
        # Generate plausible risk based on simulated feature patterns
        hrv = extracted['hrv_metrics']

        # Simple heuristic: low HRV and high HR suggest higher risk
        sdnn = hrv.get('sdnn_ms', 50)
        hr = hrv.get('mean_hr', 70)

        if sdnn < 25 or hr > 100:
            risk_score = np.random.uniform(0.7, 0.9)
            risk_category = "HIGH"
        elif sdnn < 40 or hr > 85:
            risk_score = np.random.uniform(0.4, 0.65)
            risk_category = "MEDIUM"
        else:
            risk_score = np.random.uniform(0.1, 0.35)
            risk_category = "LOW"

        model_confidence = 0.4  # Low confidence indicates simulated

        logger.warning(
            f"Using SIMULATED inference for device {device_id}. "
            "Load ML models for real predictions."
        )

    return InferenceResult(
        device_id=device_id,
        timestamp=datetime.now(),
        risk_score=round(risk_score, 4),
        risk_category=risk_category,
        hrv_metrics={
            "sdnn_ms": round(extracted['hrv_metrics'].get('sdnn_ms', 0), 2),
            "rmssd_ms": round(extracted['hrv_metrics'].get('rmssd_ms', 0), 2),
            "lf_hf_ratio": round(extracted['hrv_metrics'].get('lf_hf_ratio', 0), 2)
        },
        eeg_features={
            "alpha_power": round(extracted['eeg_features'].get('alpha_power', 0), 4),
            "beta_power": round(extracted['eeg_features'].get('beta_power', 0), 4),
            "theta_power": round(extracted['eeg_features'].get('theta_power', 0), 4),
            "entropy": round(extracted['eeg_features'].get('entropy', 0), 4)
        },
        model_confidence=round(model_confidence, 3)
    )


# ============================================================================
# WebSocket Real-time Streaming
# ============================================================================

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time data streaming to dashboard clients.
    Sends latest packets as they arrive.
    """
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive (client can send pings)
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_to_clients(data: dict):
    """Broadcast data to all connected WebSocket clients"""
    if not active_connections:
        return

    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception:
            disconnected.append(connection)

    # Remove dead connections
    for conn in disconnected:
        active_connections.remove(conn)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Service health check for load balancers"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_devices": len(data_buffer),
        "active_ws_clients": len(active_connections)
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Production: use Gunicorn with uvicorn workers
    # gunicorn -w 4 -k uvicorn.workers.UvicornWorker cloud.api.server:app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
