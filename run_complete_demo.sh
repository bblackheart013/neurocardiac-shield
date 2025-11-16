#!/bin/bash

# NeuroCardiac Shield - System Runner
# Author: Mohd Sarfaraz Faiyaz
# Contributor: Vaibhav Devram Chandgir

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

API_PID=""
DASHBOARD_PID=""
FIRMWARE_PID=""
GATEWAY_PID=""

cleanup() {
    echo "Stopping all components..."
    [ ! -z "$GATEWAY_PID" ] && kill $GATEWAY_PID 2>/dev/null || true
    [ ! -z "$API_PID" ] && kill $API_PID 2>/dev/null || true
    [ ! -z "$DASHBOARD_PID" ] && kill $DASHBOARD_PID 2>/dev/null || true
    [ ! -z "$FIRMWARE_PID" ] && kill $FIRMWARE_PID 2>/dev/null || true
    rm -f /tmp/neurocardiac_ble_data.bin 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup INT TERM

if [ ! -d "cloud/venv" ]; then
    echo "Error: Run './setup.sh' first"
    exit 1
fi

echo "Starting NeuroCardiac Shield..."

# API Server
cd cloud && source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 > ../logs/api.log 2>&1 &
API_PID=$!
deactivate && cd ..
echo "API Server: http://localhost:8000"
sleep 2

# Dashboard
cd dashboard && source venv/bin/activate
streamlit run app.py --server.port 8501 --server.headless true > ../logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
deactivate && cd ..
echo "Dashboard: http://localhost:8501"
sleep 2

# Firmware
if [ ! -f "firmware/neurocardiac_fw" ]; then
    cd firmware
    gcc -o neurocardiac_fw main.c eeg/eeg_sim.c ecg/ecg_sim.c \
        sensors/spo2_sim.c sensors/temp_sim.c sensors/accel_sim.c \
        communication/ble_stub.c -lm
    cd ..
fi
rm -f /tmp/neurocardiac_ble_data.bin 2>/dev/null || true
./firmware/neurocardiac_fw > logs/firmware.log 2>&1 &
FIRMWARE_PID=$!
echo "Firmware: Running"
sleep 1

# BLE Gateway
cd cloud && source venv/bin/activate
python3 ble_gateway.py > ../logs/gateway.log 2>&1 &
GATEWAY_PID=$!
deactivate && cd ..
echo "BLE Gateway: Running"
sleep 2

echo ""
echo "All systems running. Press CTRL+C to stop."
echo ""

while true; do
    sleep 5
    kill -0 $API_PID 2>/dev/null || cleanup
    kill -0 $DASHBOARD_PID 2>/dev/null || cleanup
    kill -0 $FIRMWARE_PID 2>/dev/null || cleanup
    kill -0 $GATEWAY_PID 2>/dev/null || cleanup
done
