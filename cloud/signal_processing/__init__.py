"""
Signal Processing Module
========================

This module provides validated signal processing functions for
physiological data (EEG and ECG) in the NeuroCardiac Shield system.

Submodules:
-----------
- preprocess: Filtering, artifact removal, R-peak detection
- features: Feature extraction for ML models
- synthetic_data: Scientifically-grounded synthetic data generation

Quick Start:
------------
    from cloud.signal_processing import filter_eeg, filter_ecg, detect_r_peaks
    from cloud.signal_processing import extract_eeg_features, extract_hrv_features
    from cloud.signal_processing import EEGGenerator, ECGGenerator

Constants:
----------
- FS_EEG, FS_ECG: Default sampling frequencies (250 Hz)
- EEG_BANDS: Frequency band definitions (delta, theta, alpha, beta, gamma)

IMPORTANT:
----------
This module is for RESEARCH/DEVELOPMENT purposes only.
Signal processing algorithms are validated against scipy/MNE-Python
standards but have NOT been clinically validated for patient care.

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 2.0.0
"""

# Import main preprocessing functions
from .preprocess import (
    filter_eeg,
    filter_ecg,
    extract_eeg_bands,
    detect_r_peaks,
    FS_EEG,
    FS_ECG,
    EEG_BANDS,
    EEG_DELTA,
    EEG_THETA,
    EEG_ALPHA,
    EEG_BETA,
    EEG_GAMMA,
    FILTER_ORDER
)

# Import feature extraction functions
from .features import (
    compute_band_power,
    extract_eeg_features,
    compute_spectral_entropy,
    compute_channel_coherence,
    extract_hrv_features,
    extract_ecg_morphology_features,
    extract_all_features
)

# Import synthetic data generators
from .synthetic_data import (
    EEGConfig,
    ECGConfig,
    EEGGenerator,
    ECGGenerator,
    generate_training_dataset
)

__all__ = [
    # Preprocessing
    'filter_eeg',
    'filter_ecg',
    'extract_eeg_bands',
    'detect_r_peaks',
    # Constants
    'FS_EEG',
    'FS_ECG',
    'EEG_BANDS',
    'EEG_DELTA',
    'EEG_THETA',
    'EEG_ALPHA',
    'EEG_BETA',
    'EEG_GAMMA',
    'FILTER_ORDER',
    # Feature extraction
    'compute_band_power',
    'extract_eeg_features',
    'compute_spectral_entropy',
    'compute_channel_coherence',
    'extract_hrv_features',
    'extract_ecg_morphology_features',
    'extract_all_features',
    # Synthetic data
    'EEGConfig',
    'ECGConfig',
    'EEGGenerator',
    'ECGGenerator',
    'generate_training_dataset'
]

__version__ = '2.0.0'
