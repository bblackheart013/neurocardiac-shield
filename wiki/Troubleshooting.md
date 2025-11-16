# Troubleshooting Guide

## Common Issues

### API Server Not Starting

**Symptom**: Port 8000 not responding

**Solution**:
```bash
# Check if port is in use
lsof -i :8000

# Kill existing processes
pkill -f uvicorn

# Restart
cd cloud && source venv/bin/activate
uvicorn api.server:app --port 8000
```

### Dashboard Not Loading

**Symptom**: Port 8501 timeout

**Solution**:
```bash
# Check Streamlit process
pgrep -f streamlit

# Restart dashboard
cd dashboard && source venv/bin/activate
streamlit run app.py --server.port 8501
```

### No Data in Dashboard

**Symptom**: Charts empty or static

**Solutions**:

1. Check firmware is running:
```bash
pgrep -f neurocardiac_fw
```

2. Check gateway is active:
```bash
pgrep -f ble_gateway
```

3. Verify API receiving data:
```bash
curl http://localhost:8000/api/v1/device/1/status
```

### ML Models Not Loading

**Symptom**: Inference returns error

**Solution**:
```bash
# Retrain models
cd ml && source venv/bin/activate
python model/train_xgboost.py
python model/train_lstm.py
```

### Firmware Compilation Error

**Symptom**: gcc fails

**Solution**:
```bash
# Ensure gcc is installed
which gcc

# On macOS
xcode-select --install
```

### Dependencies Missing

**Symptom**: Import errors

**Solution**:
```bash
# Reinstall dependencies
cd cloud && source venv/bin/activate
pip install -r requirements.txt

cd ../dashboard && source venv/bin/activate
pip install -r requirements.txt

cd ../ml && source venv/bin/activate
pip install -r requirements.txt
```

### High CPU Usage

**Symptom**: System slow

**Solution**:
- Reduce dashboard refresh rate
- Decrease time window
- Close unnecessary applications

## Logging

Check logs:
```bash
ls -la logs/
tail -f logs/*.log
```

## Clean Restart

```bash
# Kill all processes
pkill -f "uvicorn|streamlit|neurocardiac_fw|ble_gateway"

# Remove temp files
rm -f /tmp/neurocardiac_ble_data.bin

# Restart
./run_complete_demo.sh
```

## Support

For additional help, consult the documentation in `/docs` directory.
