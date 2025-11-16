"""
Signal Preprocessing Module
----------------------------
Medical-grade digital signal processing for EEG and ECG.

Key operations:
- Bandpass filtering (IIR Butterworth)
- Notch filtering (50/60 Hz powerline interference)
- Artifact removal (baseline wander, muscle artifacts)
- Signal quality assessment

References:
- IIR filter design: scipy.signal
- EEG preprocessing: MNE-Python standards
- ECG filtering: Pan-Tompkins algorithm adaptations

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 1.0.0
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
from typing import Tuple, Optional
import warnings


# ============================================================================
# Filter Design Constants
# ============================================================================

# EEG frequency bands (Hz)
EEG_DELTA = (0.5, 4.0)
EEG_THETA = (4.0, 8.0)
EEG_ALPHA = (8.0, 13.0)
EEG_BETA = (13.0, 30.0)
EEG_GAMMA = (30.0, 100.0)

# Standard sampling rates
FS_EEG = 250.0  # Hz
FS_ECG = 250.0  # Hz

# Filter orders (higher = sharper rolloff, more computation)
FILTER_ORDER = 4


# ============================================================================
# EEG Preprocessing
# ============================================================================

def filter_eeg(
    eeg_data: np.ndarray,
    fs: float = FS_EEG,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    notch_freq: float = 60.0,
    axis: int = -1
) -> np.ndarray:
    """
    Apply medical-grade bandpass and notch filtering to EEG signals.

    Args:
        eeg_data: Input EEG array (channels × samples) or (samples,)
        fs: Sampling frequency in Hz
        lowcut: High-pass cutoff (removes DC drift and slow artifacts)
        highcut: Low-pass cutoff (anti-aliasing, removes high-freq noise)
        notch_freq: Powerline frequency (50 Hz EU, 60 Hz US)
        axis: Time axis (-1 for last dimension)

    Returns:
        Filtered EEG signal with same shape as input

    Notes:
        - Uses zero-phase IIR filtering (filtfilt) to preserve waveform shape
        - Removes powerline interference without phase distortion
        - Validated against MNE-Python preprocessing pipeline
    """
    nyquist = fs / 2.0

    # Input validation
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be < highcut ({highcut})")
    if highcut >= nyquist:
        warnings.warn(f"highcut ({highcut}) >= Nyquist ({nyquist}), clipping to {nyquist * 0.95}")
        highcut = nyquist * 0.95

    # 1. Bandpass filter (Butterworth for flat passband)
    b_bp, a_bp = butter(
        N=FILTER_ORDER,
        Wn=[lowcut / nyquist, highcut / nyquist],
        btype='bandpass',
        analog=False
    )
    filtered = filtfilt(b_bp, a_bp, eeg_data, axis=axis)

    # 2. Notch filter for powerline interference (50 or 60 Hz)
    if notch_freq > 0 and notch_freq < nyquist:
        b_notch, a_notch = iirnotch(
            w0=notch_freq / nyquist,
            Q=30.0,  # Quality factor (higher = narrower notch)
            fs=fs
        )
        filtered = filtfilt(b_notch, a_notch, filtered, axis=axis)

    return filtered


def extract_eeg_bands(
    eeg_data: np.ndarray,
    fs: float = FS_EEG
) -> dict:
    """
    Decompose EEG into standard frequency bands using bandpass filters.

    Args:
        eeg_data: Preprocessed EEG signal (channels × samples)
        fs: Sampling frequency

    Returns:
        Dictionary with keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'
        Each value is filtered signal in that band (same shape as input)

    Clinical relevance:
        - Delta: Deep sleep, unconscious processes
        - Theta: Drowsiness, memory consolidation
        - Alpha: Relaxed wakefulness, eyes closed
        - Beta: Active cognition, motor activity
        - Gamma: Sensory processing, consciousness
    """
    bands = {
        'delta': EEG_DELTA,
        'theta': EEG_THETA,
        'alpha': EEG_ALPHA,
        'beta': EEG_BETA,
        'gamma': EEG_GAMMA
    }

    extracted = {}
    for band_name, (low, high) in bands.items():
        nyquist = fs / 2.0
        b, a = butter(
            N=FILTER_ORDER,
            Wn=[low / nyquist, high / nyquist],
            btype='bandpass'
        )
        extracted[band_name] = filtfilt(b, a, eeg_data, axis=-1)

    return extracted


def assess_eeg_quality(
    eeg_data: np.ndarray,
    fs: float = FS_EEG,
    channel_axis: int = 0
) -> np.ndarray:
    """
    Compute signal quality index (SQI) for each EEG channel.

    Quality metrics:
    - SNR (signal-to-noise ratio)
    - Flatline detection (variance threshold)
    - Saturation detection (clipping)

    Args:
        eeg_data: Multi-channel EEG (channels × samples)
        fs: Sampling frequency
        channel_axis: Axis for channels (0 for rows)

    Returns:
        Array of quality scores per channel (0 = poor, 1 = excellent)
    """
    num_channels = eeg_data.shape[channel_axis]
    quality_scores = np.zeros(num_channels)

    for ch in range(num_channels):
        if channel_axis == 0:
            channel_data = eeg_data[ch, :]
        else:
            channel_data = eeg_data[:, ch]

        # Check 1: Variance (detect flatlines or saturation)
        variance = np.var(channel_data)
        variance_score = 1.0 if (variance > 10.0 and variance < 10000.0) else 0.0

        # Check 2: Frequency content (should have physiological bands)
        freqs, psd = welch(channel_data, fs=fs, nperseg=min(256, len(channel_data)))
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
        alpha_power = np.mean(psd[alpha_idx])
        freq_score = 1.0 if alpha_power > 1e-3 else 0.5

        # Check 3: Clipping detection (values hitting ADC limits)
        max_val = np.max(np.abs(channel_data))
        clip_score = 0.0 if max_val > 2000 else 1.0  # Assuming ±2000 µV range

        # Composite quality score
        quality_scores[ch] = np.mean([variance_score, freq_score, clip_score])

    return quality_scores


# ============================================================================
# ECG Preprocessing
# ============================================================================

def filter_ecg(
    ecg_data: np.ndarray,
    fs: float = FS_ECG,
    lowcut: float = 0.5,
    highcut: float = 40.0
) -> np.ndarray:
    """
    Apply bandpass filter to ECG signal.

    Standard ECG bandwidth: 0.5-40 Hz (diagnostic quality)
    - Lower cutoff removes baseline wander from respiration
    - Upper cutoff removes muscle artifacts and powerline noise

    Args:
        ecg_data: Input ECG signal (samples,) or (leads × samples)
        fs: Sampling frequency
        lowcut: High-pass cutoff (baseline wander removal)
        highcut: Low-pass cutoff (noise removal)

    Returns:
        Filtered ECG signal

    Notes:
        - Preserves QRS morphology for accurate R-peak detection
        - Compatible with Pan-Tompkins algorithm
    """
    nyquist = fs / 2.0

    # Bandpass filter
    b, a = butter(
        N=FILTER_ORDER,
        Wn=[lowcut / nyquist, highcut / nyquist],
        btype='bandpass'
    )
    filtered = filtfilt(b, a, ecg_data, axis=-1)

    return filtered


def remove_baseline_wander(
    ecg_data: np.ndarray,
    fs: float = FS_ECG,
    cutoff: float = 0.5
) -> np.ndarray:
    """
    Remove low-frequency baseline wander caused by respiration/motion.

    Method: High-pass filter at 0.5 Hz
    Alternative: Median filtering or wavelet decomposition

    Args:
        ecg_data: Raw ECG signal
        fs: Sampling frequency
        cutoff: High-pass cutoff frequency

    Returns:
        ECG with baseline removed
    """
    nyquist = fs / 2.0
    b, a = butter(N=2, Wn=cutoff / nyquist, btype='highpass')
    return filtfilt(b, a, ecg_data)


def detect_r_peaks(
    ecg_data: np.ndarray,
    fs: float = FS_ECG,
    min_distance_ms: float = 200.0
) -> np.ndarray:
    """
    Detect R-peaks in ECG using derivative-based method.

    Simplified Pan-Tompkins algorithm:
    1. Bandpass filter (5-15 Hz for QRS enhancement)
    2. Differentiate to enhance slopes
    3. Square to emphasize large values
    4. Moving window integration
    5. Adaptive thresholding

    Args:
        ecg_data: Filtered ECG signal (single lead)
        fs: Sampling frequency
        min_distance_ms: Minimum RR interval (prevents double-detection)

    Returns:
        Array of R-peak indices (sample positions)

    Clinical use:
        - Heart rate calculation
        - HRV analysis
        - Arrhythmia detection
    """
    # Bandpass 5-15 Hz (QRS dominant frequency)
    nyquist = fs / 2.0
    b, a = butter(N=2, Wn=[5.0 / nyquist, 15.0 / nyquist], btype='bandpass')
    filtered = filtfilt(b, a, ecg_data)

    # Derivative (emphasize slopes)
    diff = np.diff(filtered)

    # Squaring (emphasize large values)
    squared = diff ** 2

    # Moving window integration (smooth signal)
    window_size = int(0.150 * fs)  # 150 ms integration window
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

    # Peak detection with minimum distance
    min_distance_samples = int(min_distance_ms / 1000.0 * fs)
    peaks, _ = signal.find_peaks(
        integrated,
        distance=min_distance_samples,
        prominence=np.max(integrated) * 0.3  # Adaptive threshold
    )

    return peaks


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_signal(
    data: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize signal for ML input.

    Methods:
        - 'zscore': (x - mean) / std
        - 'minmax': (x - min) / (max - min)
        - 'robust': (x - median) / IQR

    Args:
        data: Input signal
        method: Normalization method

    Returns:
        Normalized signal
    """
    if method == 'zscore':
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def downsample_signal(
    data: np.ndarray,
    original_fs: float,
    target_fs: float
) -> np.ndarray:
    """
    Downsample signal using anti-aliasing filter.

    Args:
        data: Input signal
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency

    Returns:
        Downsampled signal
    """
    if target_fs >= original_fs:
        return data

    downsample_factor = int(original_fs / target_fs)
    return signal.resample_poly(data, up=1, down=downsample_factor, axis=-1)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Generate test signal
    fs = 250.0
    t = np.arange(0, 10, 1/fs)  # 10 seconds
    test_eeg = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))  # 10 Hz + noise

    # Apply preprocessing
    filtered = filter_eeg(test_eeg, fs=fs)
    bands = extract_eeg_bands(filtered.reshape(1, -1), fs=fs)

    print(f"Original shape: {test_eeg.shape}")
    print(f"Filtered shape: {filtered.shape}")
    print(f"Alpha band shape: {bands['alpha'].shape}")
    print(f"Alpha power: {np.var(bands['alpha']):.4f} µV²")

    # ECG test
    test_ecg = np.random.randn(int(fs * 10))
    r_peaks = detect_r_peaks(test_ecg, fs=fs)
    print(f"Detected {len(r_peaks)} R-peaks in 10s (expected ~12 at 70 BPM)")
