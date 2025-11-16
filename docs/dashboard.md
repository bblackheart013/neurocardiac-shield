# Dashboard Documentation

**Author:** Mohd Sarfaraz Faiyaz
**Contributor:** Vaibhav Devram Chandgir
**Version:** 1.0.0

---

## Overview

The NeuroCardiac Shield dashboard provides real-time visualization of physiological signals, risk indicators, and clinical metrics through an interactive web interface built with Streamlit and Plotly.

---

## Technology Stack

- **Framework**: Streamlit 1.28+
- **Visualization**: Plotly 5.0+
- **Data Processing**: NumPy, Pandas
- **HTTP Client**: Requests library

---

## Access

URL: http://localhost:8501

---

## Features

### EEG Waveform Display

- 8 simultaneous channels (Fp1, Fp2, C3, C4, T3, T4, O1, O2)
- Time window: 1-10 seconds (configurable)
- Amplitude range: ±100 µV
- Color-coded channels for differentiation
- Interactive zoom and pan capabilities

### ECG Signal Visualization

- PQRST morphology display
- Amplitude range: -0.5 to 2.0 mV
- Real-time scrolling waveform
- Beat detection markers

### Risk Score Indicator

- Gauge-style meter (0-100%)
- Color-coded zones:
  - Green (0-40%): LOW risk
  - Yellow (40-70%): MEDIUM risk
  - Red (70-100%): HIGH risk
- Numerical confidence score
- Model breakdown (XGBoost vs LSTM contributions)

### HRV Metrics Panel

Displays key heart rate variability metrics:
- Mean Heart Rate (BPM)
- SDNN (ms) - Overall HRV
- RMSSD (ms) - Vagal tone
- pNN50 (%) - Parasympathetic activity
- LF/HF Ratio - Sympathovagal balance

### EEG Band Power Distribution

Bar chart showing relative power in:
- Delta (0.5-4 Hz) - Deep sleep
- Theta (4-8 Hz) - Drowsiness
- Alpha (8-13 Hz) - Relaxation
- Beta (13-30 Hz) - Active thinking
- Gamma (30-100 Hz) - Higher cognition

---

## User Interface Layout

```
┌─────────────────────────────────────┐
│         NeuroCardiac Shield         │
│     Real-time Monitoring System     │
├─────────────────┬───────────────────┤
│    EEG Display  │   Risk Gauge      │
│   (8 channels)  │   + Confidence    │
├─────────────────┼───────────────────┤
│   ECG Display   │   HRV Metrics     │
│   (PQRST wave)  │   (SDNN, RMSSD)   │
├─────────────────┴───────────────────┤
│        EEG Band Power Chart         │
└─────────────────────────────────────┘
```

---

## Configuration Options

### Sidebar Controls

- **Time Window**: Slider (1-10 seconds)
- **Auto-Refresh**: Toggle (default: ON)
- **Refresh Interval**: 2 seconds
- **Channel Selection**: Multi-select dropdown

### Settings

Edit in `dashboard/app.py`:
```python
DEFAULT_TIME_WINDOW = 5.0  # seconds
REFRESH_INTERVAL = 2.0     # seconds
API_BASE_URL = "http://localhost:8000"
```

---

## Data Flow

1. Dashboard polls API at configured interval
2. Fetches device status: `GET /api/v1/device/1/status`
3. Requests ML inference: `POST /api/v1/inference`
4. Generates signal visualizations
5. Updates all chart components
6. Repeats on auto-refresh cycle

---

## Running the Dashboard

```bash
cd dashboard
source venv/bin/activate
streamlit run app.py --server.port 8501
```

Command-line options:
```bash
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

---

## Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Render Time | <500 ms | ~400 ms |
| Memory Usage | <512 MB | ~200 MB |
| CPU Usage | <10% | ~5% |
| Network Requests | 2 per refresh | 2 per refresh |

---

## Fallback Mode

When API is unavailable, dashboard generates visualization signals locally:
- Synthetic EEG based on frequency band composition
- Synthetic ECG with PQRST morphology
- Static risk metrics for display purposes

This ensures dashboard remains functional for UI demonstration.

---

## Customization

### Adding New Visualizations

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=signal, name="Custom Signal"))
fig.update_layout(title="Custom Metric")
st.plotly_chart(fig, use_container_width=True)
```

### Modifying Color Scheme

```python
COLORS = {
    'low_risk': '#28a745',
    'medium_risk': '#ffc107',
    'high_risk': '#dc3545',
    'background': '#1e1e1e',
    'text': '#ffffff'
}
```

---

## Security Considerations

- Runs on localhost by default
- No user authentication
- No data persistence in dashboard
- Production deployment requires:
  - HTTPS proxy (NGINX)
  - User authentication
  - Session management
  - Audit logging

---

## Troubleshooting

### Dashboard Not Loading

1. Check Streamlit process:
   ```bash
   pgrep -f streamlit
   ```

2. Verify port availability:
   ```bash
   lsof -i :8501
   ```

3. Check console for errors:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

### Blank Charts

1. Verify API connectivity:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check browser console for JavaScript errors

3. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```

---

## Browser Compatibility

Tested browsers:
- Chrome 100+
- Firefox 100+
- Safari 15+
- Edge 100+

Requires JavaScript enabled for Plotly interactivity.

---

**New York University - Advanced Project**
