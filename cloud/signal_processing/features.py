"""
Feature Extraction Module
--------------------------
Extract clinically-relevant features from EEG and ECG for ML models.

EEG Features:
- Band power (delta, theta, alpha, beta, gamma)
- Spectral entropy
- Coherence between channels
- Alpha/beta ratio (cognitive load indicator)

ECG Features:
- Heart Rate Variability (HRV) metrics: SDNN, RMSSD, pNN50
- Frequency domain: LF, HF, LF/HF ratio
- Time-domain statistics

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 1.0.0
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy
from scipy.signal import welch
from typing import Dict, Tuple, List
import warnings


# ============================================================================
# EEG Feature Extraction
# ============================================================================

def compute_band_power(
    eeg_data: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    method: str = 'welch'
) -> float:
    """
    Compute power in specific frequency band using Welch's method.

    Args:
        eeg_data: Single-channel EEG signal (1D array)
        fs: Sampling frequency
        band: Frequency band as (low, high) tuple
        method: PSD estimation method ('welch' or 'multitaper')

    Returns:
        Band power in µV²

    Clinical significance:
        - High alpha: relaxed state
        - High beta: cognitive engagement
        - High theta: drowsiness
    """
    # Compute power spectral density
    if method == 'welch':
        nperseg = min(int(fs * 2), len(eeg_data))  # 2-second windows
        freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, scaling='density')
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    # Find indices for frequency band
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])

    # Integrate power over band (trapezoidal rule)
    band_power = np.trapz(psd[idx_band], freqs[idx_band])

    return band_power


def extract_eeg_features(
    eeg_data: np.ndarray,
    fs: float,
    channel_names: List[str] = None
) -> Dict[str, float]:
    """
    Extract comprehensive EEG feature set for ML models.

    Args:
        eeg_data: Multi-channel EEG (channels × samples)
        fs: Sampling frequency
        channel_names: Optional channel labels

    Returns:
        Dictionary of features:
        - Band powers (delta, theta, alpha, beta, gamma) per channel
        - Ratios: alpha/beta, theta/alpha
        - Spectral entropy
        - Mean power across channels

    Feature dimensionality:
        - 8 channels × 5 bands = 40 features
        - 8 channels × 2 ratios = 16 features
        - 8 channels × 1 entropy = 8 features
        - Total: ~64 features
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)

    num_channels = eeg_data.shape[0]
    features = {}

    # Frequency bands
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 100.0)
    }

    # Extract per-channel features
    for ch_idx in range(num_channels):
        ch_label = channel_names[ch_idx] if channel_names else f"ch{ch_idx}"
        channel_data = eeg_data[ch_idx, :]

        # Band powers
        band_powers = {}
        for band_name, band_range in bands.items():
            power = compute_band_power(channel_data, fs, band_range)
            features[f"{ch_label}_{band_name}_power"] = power
            band_powers[band_name] = power

        # Ratios (clinical markers)
        if band_powers['beta'] > 1e-6:
            features[f"{ch_label}_alpha_beta_ratio"] = band_powers['alpha'] / band_powers['beta']
        else:
            features[f"{ch_label}_alpha_beta_ratio"] = 0.0

        if band_powers['alpha'] > 1e-6:
            features[f"{ch_label}_theta_alpha_ratio"] = band_powers['theta'] / band_powers['alpha']
        else:
            features[f"{ch_label}_theta_alpha_ratio"] = 0.0

        # Spectral entropy (measure of signal complexity)
        features[f"{ch_label}_spectral_entropy"] = compute_spectral_entropy(channel_data, fs)

    # Global features (across all channels)
    all_alpha = [features[f"{channel_names[i] if channel_names else f'ch{i}'}_alpha_power"]
                 for i in range(num_channels)]
    features['mean_alpha_power'] = np.mean(all_alpha)
    features['std_alpha_power'] = np.std(all_alpha)

    return features


def compute_spectral_entropy(
    signal_data: np.ndarray,
    fs: float
) -> float:
    """
    Compute spectral entropy as a measure of signal complexity.

    Low entropy: Regular, predictable signal (e.g., deep sleep)
    High entropy: Irregular, complex signal (e.g., active cognition)

    Args:
        signal_data: 1D signal
        fs: Sampling frequency

    Returns:
        Spectral entropy (normalized 0-1)
    """
    # Compute power spectral density
    freqs, psd = welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))

    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)

    # Compute Shannon entropy
    spectral_ent = entropy(psd_norm, base=2)

    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(psd_norm))
    normalized_entropy = spectral_ent / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy


def compute_channel_coherence(
    eeg_data: np.ndarray,
    fs: float,
    ch1_idx: int,
    ch2_idx: int
) -> float:
    """
    Compute coherence between two EEG channels.

    Coherence measures functional connectivity between brain regions.

    Args:
        eeg_data: Multi-channel EEG (channels × samples)
        fs: Sampling frequency
        ch1_idx: First channel index
        ch2_idx: Second channel index

    Returns:
        Mean coherence in alpha band (8-13 Hz)

    Clinical relevance:
        - High coherence: Synchronized activity (e.g., attention)
        - Low coherence: Independent processing
    """
    if eeg_data.shape[0] <= max(ch1_idx, ch2_idx):
        return 0.0

    freqs, coh = sp_signal.coherence(
        eeg_data[ch1_idx, :],
        eeg_data[ch2_idx, :],
        fs=fs,
        nperseg=min(256, eeg_data.shape[1])
    )

    # Focus on alpha band coherence
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
    mean_alpha_coherence = np.mean(coh[alpha_idx])

    return mean_alpha_coherence


# ============================================================================
# ECG / HRV Feature Extraction
# ============================================================================

def extract_hrv_features(
    rr_intervals: np.ndarray,
    fs: float = 4.0  # Typical HRV resampling rate
) -> Dict[str, float]:
    """
    Extract Heart Rate Variability (HRV) features from RR intervals.

    Args:
        rr_intervals: RR intervals in milliseconds (peak-to-peak times)
        fs: Resampling frequency for frequency-domain analysis

    Returns:
        Dictionary of HRV metrics:
        Time-domain:
            - mean_hr: Mean heart rate (BPM)
            - sdnn: Standard deviation of NN intervals (ms)
            - rmssd: Root mean square of successive differences (ms)
            - pnn50: Percentage of intervals > 50 ms different (%)
        Frequency-domain:
            - lf_power: Low-frequency power (0.04-0.15 Hz)
            - hf_power: High-frequency power (0.15-0.4 Hz)
            - lf_hf_ratio: LF/HF ratio (sympathovagal balance)

    Clinical interpretation:
        - High SDNN: Good cardiac autonomic regulation
        - High RMSSD: Strong parasympathetic (vagal) activity
        - High LF/HF: Sympathetic dominance (stress)
        - Low HRV: Associated with cardiovascular risk
    """
    if len(rr_intervals) < 5:
        # Insufficient data
        return {
            'mean_hr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'lf_power': 0.0,
            'hf_power': 0.0,
            'lf_hf_ratio': 0.0
        }

    features = {}

    # Time-domain features
    mean_rr = np.mean(rr_intervals)
    features['mean_hr'] = 60000.0 / mean_rr if mean_rr > 0 else 0.0  # BPM

    features['sdnn'] = np.std(rr_intervals, ddof=1)  # Standard deviation

    # RMSSD: Square root of mean squared differences
    successive_diffs = np.diff(rr_intervals)
    features['rmssd'] = np.sqrt(np.mean(successive_diffs ** 2))

    # pNN50: Percentage of intervals differing by > 50 ms
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    features['pnn50'] = (nn50 / len(successive_diffs)) * 100.0 if len(successive_diffs) > 0 else 0.0

    # Frequency-domain features (requires resampling to uniform time grid)
    try:
        # Resample RR intervals to uniform time series
        rr_times = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
        rr_times = np.insert(rr_times, 0, 0)
        rr_values = np.append(rr_intervals, rr_intervals[-1])

        # Create uniform time grid
        time_uniform = np.arange(0, rr_times[-1], 1.0 / fs)
        rr_uniform = np.interp(time_uniform, rr_times, rr_values)

        # Compute PSD using Welch's method
        freqs, psd = welch(
            rr_uniform,
            fs=fs,
            nperseg=min(256, len(rr_uniform)),
            scaling='density'
        )

        # Define frequency bands
        lf_band = np.logical_and(freqs >= 0.04, freqs <= 0.15)  # Low frequency
        hf_band = np.logical_and(freqs >= 0.15, freqs <= 0.4)   # High frequency

        features['lf_power'] = np.trapz(psd[lf_band], freqs[lf_band])
        features['hf_power'] = np.trapz(psd[hf_band], freqs[hf_band])

        if features['hf_power'] > 1e-6:
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']
        else:
            features['lf_hf_ratio'] = 0.0

    except Exception as e:
        warnings.warn(f"Frequency-domain HRV calculation failed: {e}")
        features['lf_power'] = 0.0
        features['hf_power'] = 0.0
        features['lf_hf_ratio'] = 0.0

    return features


def extract_ecg_morphology_features(
    ecg_data: np.ndarray,
    r_peaks: np.ndarray,
    fs: float
) -> Dict[str, float]:
    """
    Extract morphological features from ECG waveform.

    Args:
        ecg_data: Single-lead ECG signal
        r_peaks: Indices of detected R-peaks
        fs: Sampling frequency

    Returns:
        Dictionary of morphology features:
        - mean_qrs_duration: Average QRS complex width (ms)
        - mean_rr_interval: Average RR interval (ms)
        - qrs_amplitude: Mean R-peak amplitude

    Notes:
        More advanced features (P-wave, T-wave analysis) require
        template matching and are computationally intensive.
    """
    features = {}

    if len(r_peaks) < 2:
        return {
            'mean_qrs_duration': 0.0,
            'mean_rr_interval': 0.0,
            'qrs_amplitude': 0.0
        }

    # RR intervals
    rr_intervals = np.diff(r_peaks) / fs * 1000.0  # Convert to milliseconds
    features['mean_rr_interval'] = np.mean(rr_intervals)

    # QRS amplitude (mean height of R-peaks)
    r_amplitudes = ecg_data[r_peaks]
    features['qrs_amplitude'] = np.mean(np.abs(r_amplitudes))

    # QRS duration (simplified: measure width at 50% amplitude)
    qrs_durations = []
    for peak_idx in r_peaks:
        # Define window around peak (±100 ms)
        window_samples = int(0.1 * fs)
        start = max(0, peak_idx - window_samples)
        end = min(len(ecg_data), peak_idx + window_samples)

        qrs_segment = ecg_data[start:end]
        peak_amplitude = ecg_data[peak_idx]

        # Find width at 50% amplitude
        half_amplitude = peak_amplitude * 0.5
        above_half = qrs_segment > half_amplitude
        if np.any(above_half):
            duration_samples = np.sum(above_half)
            duration_ms = (duration_samples / fs) * 1000.0
            qrs_durations.append(duration_ms)

    features['mean_qrs_duration'] = np.mean(qrs_durations) if qrs_durations else 0.0

    return features


# ============================================================================
# Combined Feature Extraction
# ============================================================================

def extract_all_features(
    eeg_data: np.ndarray,
    ecg_data: np.ndarray,
    r_peaks: np.ndarray,
    fs_eeg: float = 250.0,
    fs_ecg: float = 250.0,
    channel_names: List[str] = None
) -> Dict[str, float]:
    """
    Extract complete feature set from multi-modal physiological data.

    Args:
        eeg_data: Multi-channel EEG (channels × samples)
        ecg_data: Single-lead ECG
        r_peaks: R-peak indices for ECG
        fs_eeg: EEG sampling frequency
        fs_ecg: ECG sampling frequency
        channel_names: EEG channel labels

    Returns:
        Dictionary with ~80+ features for ML model input

    Pipeline:
    1. EEG band powers, ratios, entropy → ~64 features
    2. HRV time/frequency domain → 7 features
    3. ECG morphology → 3 features
    4. Cross-modal features (future: EEG-ECG coupling)
    """
    features = {}

    # EEG features
    eeg_feats = extract_eeg_features(eeg_data, fs_eeg, channel_names)
    features.update(eeg_feats)

    # HRV features
    if len(r_peaks) >= 2:
        rr_intervals = np.diff(r_peaks) / fs_ecg * 1000.0  # Convert to ms
        hrv_feats = extract_hrv_features(rr_intervals)
        features.update(hrv_feats)
    else:
        # Default values if insufficient peaks
        features.update({
            'mean_hr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'lf_power': 0.0,
            'hf_power': 0.0,
            'lf_hf_ratio': 0.0
        })

    # ECG morphology features
    ecg_morph = extract_ecg_morphology_features(ecg_data, r_peaks, fs_ecg)
    features.update(ecg_morph)

    return features


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Simulate test data
    fs = 250.0
    duration = 60.0  # 60 seconds
    t = np.arange(0, duration, 1/fs)

    # Simulate 8-channel EEG
    eeg_sim = np.random.randn(8, len(t)) * 20  # 8 channels, 20 µV noise
    for ch in range(8):
        eeg_sim[ch, :] += 30 * np.sin(2 * np.pi * 10 * t)  # Add 10 Hz alpha

    # Simulate ECG with R-peaks every ~0.85s (70 BPM)
    ecg_sim = np.random.randn(len(t)) * 0.05
    r_peaks_sim = np.arange(100, len(t), int(0.85 * fs))

    # Extract features
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
    all_features = extract_all_features(
        eeg_sim,
        ecg_sim,
        r_peaks_sim,
        fs_eeg=fs,
        fs_ecg=fs,
        channel_names=channel_names
    )

    print(f"Extracted {len(all_features)} features:")
    for key, value in list(all_features.items())[:10]:
        print(f"  {key}: {value:.4f}")
    print("  ...")
    print(f"\nKey HRV metrics:")
    print(f"  Mean HR: {all_features['mean_hr']:.1f} BPM")
    print(f"  SDNN: {all_features['sdnn']:.2f} ms")
    print(f"  RMSSD: {all_features['rmssd']:.2f} ms")
    print(f"  LF/HF Ratio: {all_features['lf_hf_ratio']:.2f}")
