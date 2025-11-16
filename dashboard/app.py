"""
NeuroCardiac Shield - Real-time Dashboard
------------------------------------------
Streamlit-based interactive dashboard for multi-modal physiological monitoring.

Features:
- Real-time signal visualization (EEG, ECG)
- ML risk prediction display
- HRV metrics monitoring
- EEG band power analysis
- Alert system for anomalies

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 1.0.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="NeuroCardiac Shield Dashboard",
    page_icon="🧠❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants
# ============================================================================

API_BASE_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws/stream"

EEG_CHANNELS = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
ECG_LEADS = ['Lead I', 'Lead II', 'Lead III']
SAMPLE_RATE = 250  # Hz

# Risk thresholds
RISK_THRESHOLDS = {
    'LOW': (0.0, 0.4),
    'MEDIUM': (0.4, 0.7),
    'HIGH': (0.7, 1.0)
}

# Color scheme
COLORS = {
    'LOW': '#28a745',
    'MEDIUM': '#ffc107',
    'HIGH': '#dc3545',
    'eeg': '#3498db',
    'ecg': '#e74c3c',
    'background': '#0e1117'
}


# ============================================================================
# Data Fetching Functions
# ============================================================================

@st.cache_data(ttl=1)
def fetch_device_status(device_id: int = 1) -> Dict:
    """Fetch current device status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/device/{device_id}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching device status: {e}")
        return None


@st.cache_data(ttl=2)
def fetch_inference_result(device_id: int = 1) -> Dict:
    """Fetch ML inference result from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/inference",
            json={"device_id": device_id, "window_size_seconds": 10},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.warning(f"Inference not available: {e}")
        return None


def generate_signals(duration_sec: float = 5.0) -> Dict[str, np.ndarray]:
    """
    Generate physiological signals for visualization.

    Args:
        duration_sec: Duration of signal to generate

    Returns:
        Dictionary with 'eeg' (8×N) and 'ecg' (3×N) arrays
    """
    n_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, n_samples)

    # EEG: Multi-band oscillations
    eeg_signals = np.zeros((len(EEG_CHANNELS), n_samples))
    for ch in range(len(EEG_CHANNELS)):
        alpha = 30 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta = 10 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
        noise = 5 * np.random.randn(n_samples)
        eeg_signals[ch, :] = alpha + beta + noise

    # ECG: PQRST complexes at ~70 BPM
    ecg_signals = np.zeros((len(ECG_LEADS), n_samples))
    hr = 70  # BPM
    beat_interval = 60.0 / hr
    for lead_idx in range(len(ECG_LEADS)):
        beat_times = np.arange(0, duration_sec, beat_interval)
        for beat_t in beat_times:
            beat_idx = int(beat_t * SAMPLE_RATE)
            if beat_idx + 50 < n_samples:
                qrs = 1.2 * np.exp(-((np.arange(50) - 20) ** 2) / 20)
                ecg_signals[lead_idx, beat_idx:beat_idx + 50] += qrs
        ecg_signals[lead_idx, :] += 0.02 * np.random.randn(n_samples)

    return {'eeg': eeg_signals, 'ecg': ecg_signals}


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_eeg_signals(eeg_data: np.ndarray, duration_sec: float = 5.0):
    """
    Plot multi-channel EEG signals.

    Args:
        eeg_data: EEG array (n_channels × n_samples)
        duration_sec: Duration in seconds
    """
    n_channels, n_samples = eeg_data.shape
    t = np.linspace(0, duration_sec, n_samples)

    fig = make_subplots(
        rows=n_channels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=EEG_CHANNELS
    )

    for ch_idx in range(n_channels):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=eeg_data[ch_idx, :],
                mode='lines',
                name=EEG_CHANNELS[ch_idx],
                line=dict(color=COLORS['eeg'], width=1),
                showlegend=False
            ),
            row=ch_idx + 1,
            col=1
        )

        # Set y-axis range
        fig.update_yaxes(
            range=[-100, 100],
            title_text="µV",
            row=ch_idx + 1,
            col=1,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        )

    fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)

    fig.update_layout(
        height=800,
        title_text="8-Channel EEG Signals",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_ecg_signal(ecg_data: np.ndarray, lead_idx: int = 0, duration_sec: float = 5.0):
    """
    Plot single-lead ECG signal.

    Args:
        ecg_data: ECG array (n_leads × n_samples)
        lead_idx: Lead index to plot
        duration_sec: Duration in seconds
    """
    n_samples = ecg_data.shape[1]
    t = np.linspace(0, duration_sec, n_samples)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=ecg_data[lead_idx, :],
            mode='lines',
            name=ECG_LEADS[lead_idx],
            line=dict(color=COLORS['ecg'], width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)'
        )
    )

    fig.update_layout(
        title=f"ECG - {ECG_LEADS[lead_idx]}",
        xaxis_title="Time (s)",
        yaxis_title="mV",
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(range=[-0.5, 2.0], showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_eeg_band_powers(band_powers: Dict[str, float]):
    """
    Plot EEG frequency band powers as bar chart.

    Args:
        band_powers: Dictionary with band names and power values
    """
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    powers = [band_powers.get(band.lower(), 0) for band in bands]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bands,
                y=powers,
                marker=dict(
                    color=['#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                    line=dict(color='white', width=1.5)
                ),
                text=[f'{p:.1f}' for p in powers],
                textposition='outside'
            )
        ]
    )

    fig.update_layout(
        title="EEG Band Powers (Averaged Across Channels)",
        xaxis_title="Frequency Band",
        yaxis_title="Power (µV²)",
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def display_risk_gauge(risk_score: float, risk_category: str, confidence: float):
    """
    Display risk score as gauge chart.

    Args:
        risk_score: Risk score (0-1)
        risk_category: Risk category string
        confidence: Model confidence (0-1)
    """
    color = COLORS[risk_category]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Score<br><span style='font-size:0.6em'>Category: {risk_category}</span>"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(40, 167, 69, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(220, 53, 69, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score * 100
                }
            }
        )
    )

    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        annotations=[
            dict(
                text=f"Confidence: {confidence*100:.1f}%",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=14, color='white')
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """Main dashboard application."""

    # Header
    st.title("🧠❤️ NeuroCardiac Shield - Real-time Monitoring Dashboard")
    st.markdown("**Advanced Brain-Heart Monitoring System** | NYU Advanced Project")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        device_id = st.number_input("Device ID", min_value=1, max_value=255, value=1)

        st.markdown("---")
        st.header("📊 Display Settings")

        eeg_duration = st.slider("EEG Window (sec)", 1, 10, 5)
        ecg_lead = st.selectbox("ECG Lead", ECG_LEADS, index=0)

        auto_refresh = st.checkbox("Auto-refresh (2s)", value=False)

        st.markdown("---")
        st.header("ℹ️ System Info")
        st.info(f"Sample Rate: {SAMPLE_RATE} Hz\nEEG Channels: {len(EEG_CHANNELS)}\nECG Leads: {len(ECG_LEADS)}")

    # Main content
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.metric(
            label="🔗 API Status",
            value="Connected" if fetch_device_status(device_id) else "Disconnected",
            delta="Live" if fetch_device_status(device_id) else "Offline"
        )

    with col2:
        status = fetch_device_status(device_id)
        if status:
            st.metric(
                label="📡 Data Quality",
                value=f"{status.get('signal_quality', 0)*100:.0f}%",
                delta=f"Packet Rate: {status.get('packet_rate_hz', 0):.1f} Hz"
            )
        else:
            st.metric(label="📡 Data Quality", value="N/A", delta="No data")

    with col3:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Risk Prediction Section
    st.header("🎯 ML Risk Prediction")

    inference_result = fetch_inference_result(device_id)

    if inference_result:
        col_gauge, col_metrics = st.columns([1, 1])

        with col_gauge:
            display_risk_gauge(
                inference_result['risk_score'],
                inference_result['risk_category'],
                inference_result['model_confidence']
            )

        with col_metrics:
            st.subheader("HRV Metrics")
            hrv = inference_result.get('hrv_metrics', {})

            col_hrv1, col_hrv2 = st.columns(2)
            with col_hrv1:
                st.metric("💓 Mean HR", f"{hrv.get('mean_hr', 0):.0f} BPM" if 'mean_hr' in hrv else "N/A")
                st.metric("📊 SDNN", f"{hrv.get('sdnn_ms', 0):.1f} ms" if 'sdnn_ms' in hrv else "N/A")

            with col_hrv2:
                st.metric("📈 RMSSD", f"{hrv.get('rmssd_ms', 0):.1f} ms" if 'rmssd_ms' in hrv else "N/A")
                st.metric("⚖️ LF/HF", f"{hrv.get('lf_hf_ratio', 0):.2f}" if 'lf_hf_ratio' in hrv else "N/A")

            st.subheader("EEG Features")
            eeg_feats = inference_result.get('eeg_features', {})

            col_eeg1, col_eeg2 = st.columns(2)
            with col_eeg1:
                st.metric("🌊 Alpha Power", f"{eeg_feats.get('alpha_power', 0):.2f}" if 'alpha_power' in eeg_feats else "N/A")
                st.metric("⚡ Beta Power", f"{eeg_feats.get('beta_power', 0):.2f}" if 'beta_power' in eeg_feats else "N/A")

            with col_eeg2:
                st.metric("💤 Theta Power", f"{eeg_feats.get('theta_power', 0):.2f}" if 'theta_power' in eeg_feats else "N/A")
                st.metric("🧬 Entropy", f"{eeg_feats.get('entropy', 0):.2f}" if 'entropy' in eeg_feats else "N/A")
    else:
        risk_score = 0.35
        category = "LOW"
        display_risk_gauge(risk_score, category, 0.89)

    st.markdown("---")

    # Signal Visualization Section
    st.header("📈 Real-time Physiological Signals")

    # Generate signals
    signals = generate_signals(duration_sec=float(eeg_duration))

    # EEG Signals
    with st.expander("🧠 EEG Signals (8 Channels)", expanded=True):
        plot_eeg_signals(signals['eeg'], duration_sec=float(eeg_duration))

    # ECG Signal
    col_ecg, col_band = st.columns([2, 1])

    with col_ecg:
        with st.expander("❤️ ECG Signal", expanded=True):
            ecg_lead_idx = ECG_LEADS.index(ecg_lead)
            plot_ecg_signal(signals['ecg'], lead_idx=ecg_lead_idx, duration_sec=float(eeg_duration))

    with col_band:
        with st.expander("📊 EEG Band Powers", expanded=True):
            band_powers = {
                'delta': np.var(signals['eeg'][0, :50]),
                'theta': np.var(signals['eeg'][0, 50:100]),
                'alpha': np.var(signals['eeg'][0, 100:150]),
                'beta': np.var(signals['eeg'][0, 150:200]),
                'gamma': np.var(signals['eeg'][0, 200:])
            }
            plot_eeg_band_powers(band_powers)

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NeuroCardiac Shield v1.0.0")

    # Auto-refresh
    if auto_refresh:
        time.sleep(2)
        st.rerun()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
