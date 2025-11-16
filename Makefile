# NeuroCardiac Shield - Makefile
# Convenience commands for development and deployment

.PHONY: help install-all train-models run-firmware run-api run-dashboard test-api clean

help:
	@echo "NeuroCardiac Shield - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install-all     - Install all Python dependencies"
	@echo "  make train-models    - Train XGBoost and LSTM models"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run-firmware    - Compile and run firmware simulator"
	@echo "  make run-api         - Start FastAPI cloud backend"
	@echo "  make run-dashboard   - Start Streamlit dashboard"
	@echo ""
	@echo "Testing:"
	@echo "  make test-api        - Test API endpoints"
	@echo "  make test-signals    - Test signal processing"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           - Remove compiled files and cache"
	@echo ""

# Installation
install-all:
	@echo "Installing cloud dependencies..."
	cd cloud && pip install -r requirements.txt
	@echo "Installing ML dependencies..."
	cd ml && pip install -r requirements.txt
	@echo "Installing dashboard dependencies..."
	cd dashboard && pip install -r requirements.txt
	@echo "✓ All dependencies installed"

# Model Training
train-models:
	@echo "Training XGBoost model..."
	cd ml/model && python train_xgboost.py
	@echo ""
	@echo "Training LSTM model..."
	cd ml/model && python train_lstm.py
	@echo "✓ Models trained and saved to ml/checkpoints/"

# Firmware
compile-firmware:
	@echo "Compiling firmware..."
	gcc -o firmware/neurocardiac_fw \
		firmware/main.c \
		firmware/eeg/eeg_sim.c \
		firmware/ecg/ecg_sim.c \
		firmware/sensors/spo2_sim.c \
		firmware/sensors/temp_sim.c \
		firmware/sensors/accel_sim.c \
		firmware/communication/ble_stub.c \
		-lm -O2
	@echo "✓ Firmware compiled: firmware/neurocardiac_fw"

run-firmware: compile-firmware
	@echo "Running firmware simulator..."
	./firmware/neurocardiac_fw

# Cloud Backend
run-api:
	@echo "Starting FastAPI server on http://localhost:8000"
	@echo "API docs: http://localhost:8000/docs"
	cd cloud && uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Dashboard
run-dashboard:
	@echo "Starting Streamlit dashboard on http://localhost:8501"
	cd dashboard && streamlit run app.py --server.port 8501

# Testing
test-api:
	@echo "Testing API health endpoint..."
	curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@echo "Testing device status endpoint..."
	curl -s http://localhost:8000/api/v1/device/1/status | python -m json.tool || echo "No device data yet"

test-signals:
	@echo "Testing signal processing..."
	cd cloud/signal_processing && python preprocess.py
	@echo ""
	cd cloud/signal_processing && python features.py

test-inference:
	@echo "Testing ML inference engine..."
	cd ml/model && python inference.py

# Cleanup
clean:
	@echo "Cleaning compiled files and cache..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type f -name '.DS_Store' -delete
	rm -f firmware/neurocardiac_fw
	rm -f /tmp/neurocardiac_ble_data.bin
	@echo "✓ Cleanup complete"

# All-in-one setup
setup: install-all train-models
	@echo ""
	@echo "✓ NeuroCardiac Shield setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  Terminal 1: make run-api"
	@echo "  Terminal 2: make run-dashboard"
	@echo "  Terminal 3: make run-firmware (optional)"
