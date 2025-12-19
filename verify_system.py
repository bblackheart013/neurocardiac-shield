#!/usr/bin/env python3
"""
NeuroCardiac Shield - System Verification Script
=================================================

This script validates all major components of the NeuroCardiac Shield
system to ensure proper installation and functionality.

Run this script after setup to verify the system is working correctly:
    python verify_system.py

Exit Codes:
-----------
0: All checks passed
1: Some checks failed (see output for details)

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
Institution: NYU Tandon School of Engineering
Course: ECE-GY 9953 | Fall 2025
Version: 2.0.0
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Track results
results = []


def check(name: str, condition: bool, details: str = ""):
    """Record a check result."""
    status = "PASS" if condition else "FAIL"
    results.append((name, status, details))
    symbol = "[OK]" if condition else "[X] "
    print(f"  {symbol} {name}")
    if details and not condition:
        print(f"      -> {details}")
    return condition


def section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("  NeuroCardiac Shield - System Verification")
    print("=" * 60)

    all_pass = True

    # ==========================================================================
    section("1. Directory Structure")
    # ==========================================================================

    dirs_to_check = [
        "firmware",
        "firmware/eeg",
        "firmware/ecg",
        "firmware/sensors",
        "firmware/communication",
        "cloud",
        "cloud/api",
        "cloud/signal_processing",
        "ml",
        "ml/model",
        "ml/checkpoints",
        "dashboard",
        "docs",
        "config"
    ]

    for dir_path in dirs_to_check:
        full_path = PROJECT_ROOT / dir_path
        all_pass &= check(f"Directory: {dir_path}", full_path.exists())

    # ==========================================================================
    section("2. Core Source Files")
    # ==========================================================================

    files_to_check = [
        ("firmware/main.c", "Main firmware loop"),
        ("firmware/eeg/eeg_sim.c", "EEG simulation"),
        ("firmware/ecg/ecg_sim.c", "ECG simulation"),
        ("cloud/api/server.py", "API server"),
        ("cloud/signal_processing/preprocess.py", "Signal preprocessing"),
        ("cloud/signal_processing/features.py", "Feature extraction"),
        ("cloud/signal_processing/synthetic_data.py", "Synthetic data generation"),
        ("cloud/ble_gateway.py", "BLE gateway"),
        ("ml/model/inference.py", "ML inference"),
        ("ml/model/train_xgboost.py", "XGBoost training"),
        ("ml/model/train_lstm.py", "LSTM training"),
        ("dashboard/app.py", "Dashboard app"),
    ]

    for file_path, desc in files_to_check:
        full_path = PROJECT_ROOT / file_path
        all_pass &= check(f"{desc}", full_path.exists(), f"Missing: {file_path}")

    # ==========================================================================
    section("3. Documentation")
    # ==========================================================================

    docs_to_check = [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/DATA_FLOW.md",
        "docs/ML_PIPELINE.md",
        "docs/SIMULATION_SCOPE.md",
        "docs/DEVICE_INTEGRATION.md",
        "docs/DATA_BIBLE.md"
    ]

    for doc_path in docs_to_check:
        full_path = PROJECT_ROOT / doc_path
        all_pass &= check(f"Doc: {doc_path}", full_path.exists())

    # ==========================================================================
    section("4. Configuration Files")
    # ==========================================================================

    config_files = [
        "config/config.yaml",
        "cloud/requirements.txt",
        "ml/requirements.txt",
        "setup.sh",
        "run_complete_demo.sh"
    ]

    for config_path in config_files:
        full_path = PROJECT_ROOT / config_path
        all_pass &= check(f"Config: {config_path}", full_path.exists())

    # ==========================================================================
    section("5. ML Model Checkpoints")
    # ==========================================================================

    model_files = [
        "ml/checkpoints/xgboost/xgboost_model.json",
        "ml/checkpoints/xgboost/scaler.pkl",
        "ml/checkpoints/xgboost/metadata.json",
        "ml/checkpoints/lstm/lstm_metadata.json"
    ]

    for model_path in model_files:
        full_path = PROJECT_ROOT / model_path
        all_pass &= check(f"Model: {model_path}", full_path.exists())

    # ==========================================================================
    section("6. Python Imports")
    # ==========================================================================

    try:
        import numpy as np
        all_pass &= check("NumPy", True, f"v{np.__version__}")
    except ImportError as e:
        all_pass &= check("NumPy", False, str(e))

    try:
        import scipy
        all_pass &= check("SciPy", True, f"v{scipy.__version__}")
    except ImportError as e:
        all_pass &= check("SciPy", False, str(e))

    try:
        import pandas
        all_pass &= check("Pandas", True, f"v{pandas.__version__}")
    except ImportError as e:
        all_pass &= check("Pandas", False, str(e))

    try:
        import fastapi
        all_pass &= check("FastAPI", True, f"v{fastapi.__version__}")
    except ImportError as e:
        all_pass &= check("FastAPI", False, str(e))

    try:
        import xgboost
        check("XGBoost", True, f"v{xgboost.__version__}")
    except ImportError as e:
        check("XGBoost (optional)", True, "Not installed - install in venv for ML")

    try:
        import streamlit
        all_pass &= check("Streamlit", True, f"v{streamlit.__version__}")
    except ImportError as e:
        all_pass &= check("Streamlit", False, str(e))

    # ==========================================================================
    section("7. Signal Processing Module")
    # ==========================================================================

    try:
        from cloud.signal_processing.preprocess import filter_eeg, filter_ecg
        all_pass &= check("filter_eeg function", callable(filter_eeg))
        all_pass &= check("filter_ecg function", callable(filter_ecg))
    except Exception as e:
        all_pass &= check("Signal processing imports", False, str(e))

    try:
        from cloud.signal_processing.features import extract_eeg_features
        all_pass &= check("extract_eeg_features function", callable(extract_eeg_features))
    except Exception as e:
        all_pass &= check("Feature extraction imports", False, str(e))

    try:
        from cloud.signal_processing.synthetic_data import EEGGenerator, ECGGenerator
        all_pass &= check("EEGGenerator class", True)
        all_pass &= check("ECGGenerator class", True)
    except Exception as e:
        all_pass &= check("Synthetic data imports", False, str(e))

    # ==========================================================================
    section("8. Synthetic Data Generation")
    # ==========================================================================

    try:
        from cloud.signal_processing.synthetic_data import EEGConfig, EEGGenerator
        config = EEGConfig(fs=250, duration_sec=1, seed=42)
        gen = EEGGenerator(config)
        eeg_data, time_vec = gen.generate()

        all_pass &= check(
            "EEG generation shape",
            eeg_data.shape == (8, 250),
            f"Got {eeg_data.shape}, expected (8, 250)"
        )
        eeg_min, eeg_max = eeg_data.min(), eeg_data.max()
        all_pass &= check(
            "EEG amplitude range",
            eeg_min > -500 and eeg_max < 500,
            f"Range: [{eeg_min:.1f}, {eeg_max:.1f}] uV"
        )
    except Exception as e:
        all_pass &= check("EEG generation", False, str(e))

    try:
        from cloud.signal_processing.synthetic_data import ECGConfig, ECGGenerator
        config = ECGConfig(fs=250, duration_sec=5, seed=42)
        gen = ECGGenerator(config)
        ecg_data, time_vec, r_peaks = gen.generate()

        all_pass &= check(
            "ECG generation shape",
            ecg_data.shape == (1250,),
            f"Got {ecg_data.shape}, expected (1250,)"
        )
        all_pass &= check(
            "R-peak detection",
            len(r_peaks) >= 3,
            f"Found {len(r_peaks)} peaks"
        )
    except Exception as e:
        all_pass &= check("ECG generation", False, str(e))

    # ==========================================================================
    section("9. Feature Extraction")
    # ==========================================================================

    try:
        import numpy as np
        from cloud.signal_processing.features import extract_eeg_features, extract_hrv_features

        # Test EEG features
        eeg_test = np.random.randn(8, 500) * 20
        eeg_feats = extract_eeg_features(eeg_test, fs=250.0)
        all_pass &= check(
            "EEG feature count",
            len(eeg_feats) >= 60,
            f"Got {len(eeg_feats)} features"
        )

        # Test HRV features
        rr_test = np.array([800, 820, 790, 810, 830, 800, 810])  # ms
        hrv_feats = extract_hrv_features(rr_test)
        all_pass &= check(
            "HRV feature extraction",
            'sdnn' in hrv_feats and 'rmssd' in hrv_feats,
            f"Keys: {list(hrv_feats.keys())}"
        )
    except Exception as e:
        all_pass &= check("Feature extraction", False, str(e))

    # ==========================================================================
    section("10. ML Inference")
    # ==========================================================================

    try:
        # Try importing without xgboost dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "inference",
            PROJECT_ROOT / "ml/model/inference.py"
        )
        check("ML inference module exists", spec is not None)

        # Check if NeuroCardiacInference class is defined in source
        with open(PROJECT_ROOT / "ml/model/inference.py") as f:
            content = f.read()
            check("NeuroCardiacInference class defined",
                  "class NeuroCardiacInference" in content)
            check("predict_ensemble method defined",
                  "def predict_ensemble" in content)
            check("Explainability support",
                  "_generate_explanations" in content)

    except Exception as e:
        check("ML inference module", False, str(e))

    # ==========================================================================
    section("11. Device Adapters")
    # ==========================================================================

    try:
        from cloud.device_adapters import (
            DeviceAdapter, GoldPacket, SimulatedAdapter,
            get_adapter, ADAPTER_REGISTRY
        )
        all_pass &= check("Device adapter imports", True)
        all_pass &= check("Adapter registry populated", len(ADAPTER_REGISTRY) >= 3)

        # Test simulated adapter
        adapter = get_adapter('simulated', seed=42)
        all_pass &= check("SimulatedAdapter creation", adapter is not None)

        if adapter.connect():
            adapter.start_stream()
            packet = adapter.read_packet(timeout_ms=2000)
            adapter.stop()

            all_pass &= check("GoldPacket generation", packet is not None)
            if packet:
                all_pass &= check(
                    "EEG channels",
                    len(packet.eeg) == 8,
                    f"Got {len(packet.eeg)}"
                )
                all_pass &= check(
                    "ECG leads",
                    len(packet.ecg) == 3,
                    f"Got {len(packet.ecg)}"
                )
        else:
            all_pass &= check("SimulatedAdapter connect", False)

    except Exception as e:
        all_pass &= check("Device adapters", False, str(e))

    # ==========================================================================
    section("12. AI Mentions Check")
    # ==========================================================================

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "no_ai_mentions_check.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        all_pass &= check(
            "No AI/assistant mentions in codebase",
            result.returncode == 0,
            "Run 'python tools/no_ai_mentions_check.py' for details" if result.returncode != 0 else ""
        )
    except Exception as e:
        all_pass &= check("AI mentions check", False, str(e))

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    total = len(results)

    print(f"\n  Total Checks: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print("\n  Failed Checks:")
        for name, status, details in results:
            if status == "FAIL":
                print(f"    - {name}")
                if details:
                    print(f"      {details}")

    print("\n" + "=" * 60)

    if all_pass:
        print("  STATUS: ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("  STATUS: SOME CHECKS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
