#!/bin/bash

# NeuroCardiac Shield - Setup Script
# Author: Mohd Sarfaraz Faiyaz
# Contributor: Vaibhav Devram Chandgir

set -e

echo "NeuroCardiac Shield - Setup"
echo ""

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

create_venv() {
    local dir=$1
    local name=$2
    echo "Setting up $name..."
    cd "$dir"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    deactivate
    cd - > /dev/null
}

echo "Installing Cloud Backend dependencies..."
create_venv "cloud" "Cloud Backend"

echo "Installing ML Pipeline dependencies..."
create_venv "ml" "ML Pipeline"

echo "Installing Dashboard dependencies..."
create_venv "dashboard" "Dashboard"

echo "Training ML models..."
cd ml
source venv/bin/activate
python model/train_xgboost.py
python model/train_lstm.py
deactivate
cd - > /dev/null

if command -v gcc &> /dev/null; then
    echo "Compiling firmware..."
    gcc -o firmware/neurocardiac_fw \
        firmware/main.c \
        firmware/eeg/eeg_sim.c \
        firmware/ecg/ecg_sim.c \
        firmware/sensors/spo2_sim.c \
        firmware/sensors/temp_sim.c \
        firmware/sensors/accel_sim.c \
        firmware/communication/ble_stub.c \
        -lm -O2
fi

echo ""
echo "Setup complete."
echo "Run ./run_complete_demo.sh to start all components."
