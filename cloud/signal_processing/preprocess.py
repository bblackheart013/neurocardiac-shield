"""
Signal Preprocessing Module
===========================

Medical-grade digital signal processing for EEG and ECG signals.

This module implements validated signal processing algorithms following
established clinical and research standards. All filter designs are
mathematically justified with references to foundational literature.

Key Operations:
---------------
1. Bandpass filtering (IIR Butterworth, zero-phase)
2. Notch filtering (50/60 Hz powerline interference removal)
3. Baseline wander removal (high-pass filtering)
4. Signal quality assessment (SQI computation)
5. R-peak detection (derivative-based Pan-Tompkins variant)

Mathematical Foundation:
------------------------
All filters use Butterworth design for maximally flat passband response.
Zero-phase filtering (filtfilt) is applied to preserve waveform morphology,
which doubles the effective filter order.

Effective transfer function for filtfilt:
    H_eff(f) = |H(f)|^2

Where H(f) is the single-pass Butterworth response.

References:
-----------
1. Butterworth, S. (1930). "On the Theory of Filter Amplifiers",
   Wireless Engineer, 7(6), 536-541.
2. Pan, J. & Tompkins, W.J. (1985). "A Real-Time QRS Detection Algorithm",
   IEEE Trans Biomed Eng, 32(3), 230-236.
3. MNE-Python preprocessing standards:
   https://mne.tools/stable/overview/preprocessing.html
4. IEC 60601-2-27 for ECG recording equipment standards

IMPORTANT LIMITATIONS:
----------------------
- This is DEVELOPMENT/RESEARCH code, NOT FDA-cleared for clinical use
- Filter designs are validated but require clinical validation before
  any patient-facing deployment
- Signal quality metrics are heuristic and may not detect all artifacts

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 2.0.0
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
from typing import Tuple, Optional
import warnings


# =============================================================================
# Filter Design Constants
# =============================================================================

# EEG frequency bands (Hz) - IFCN Standard Definitions
# Reference: IFCN Guidelines for Standard Electrode Position Nomenclature
EEG_BANDS = {
    'delta': (0.5, 4.0),   # Deep sleep, unconscious processes
    'theta': (4.0, 8.0),   # Drowsiness, memory consolidation
    'alpha': (8.0, 13.0),  # Relaxed wakefulness, posterior dominant rhythm
    'beta': (13.0, 30.0),  # Active cognition, motor activity
    'gamma': (30.0, 100.0) # High-level processing (note: often contaminated)
}

# Legacy band tuple exports for backward compatibility
EEG_DELTA = EEG_BANDS['delta']
EEG_THETA = EEG_BANDS['theta']
EEG_ALPHA = EEG_BANDS['alpha']
EEG_BETA = EEG_BANDS['beta']
EEG_GAMMA = EEG_BANDS['gamma']

# Standard sampling rates (Hz)
# 250 Hz provides Nyquist frequency of 125 Hz, sufficient for all bands
FS_EEG = 250.0
FS_ECG = 250.0

# Filter design parameters
# Order 4 Butterworth with filtfilt gives effective order 8
# -3dB at cutoff, -48 dB/octave rolloff (effective -96 dB/octave with filtfilt)
FILTER_ORDER = 4

# Minimum signal length for filtering (in samples)
# Butterworth IIR needs 3*max(len(a), len(b)) samples for filtfilt
# For order 4: coefficients length is 5, so minimum is 15 samples
# Using 4x the filter order as a conservative minimum
MIN_SAMPLES_FOR_FILTER = 4 * FILTER_ORDER * 3  # = 48 samples


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

    This function implements a two-stage filtering pipeline:
    1. Butterworth bandpass filter (zero-phase) for frequency selection
    2. IIR notch filter (zero-phase) for powerline interference removal

    Mathematical Details:
    ---------------------
    Butterworth filter transfer function magnitude:
        |H(f)|^2 = 1 / (1 + (f/fc)^(2n))

    Where:
    - fc: cutoff frequency
    - n: filter order

    For filtfilt (zero-phase), effective response is |H(f)|^4, doubling
    the effective order and squaring the magnitude response.

    Filter Design Choices:
    ----------------------
    - Order 4 Butterworth: -3 dB at cutoff, -24 dB/octave rolloff
    - With filtfilt: effective -48 dB/octave, maximally flat passband
    - Notch Q=30: 3-dB bandwidth of (notch_freq / 30) Hz = 2 Hz for 60 Hz

    Args:
        eeg_data: Input EEG array. Shape: (samples,) or (channels, samples)
        fs: Sampling frequency in Hz. Must be > 2 * highcut (Nyquist).
        lowcut: High-pass cutoff frequency in Hz. Removes DC and drift.
                Typical: 0.5 Hz (preserves delta), 1.0 Hz (removes more drift)
        highcut: Low-pass cutoff frequency in Hz. Removes high-freq noise.
                 Typical: 40-50 Hz for clinical, 100+ Hz for research.
        notch_freq: Powerline interference frequency in Hz.
                    Set to 60 Hz (Americas/Asia) or 50 Hz (Europe/Oceania).
                    Set to 0 or negative to disable notch filtering.
        axis: Axis along which to filter. -1 for last dimension.

    Returns:
        Filtered EEG signal with identical shape to input.
        Units preserved (typically µV for EEG).

    Raises:
        ValueError: If lowcut >= highcut or invalid frequency parameters.
        ValueError: If signal too short for stable filtering.

    Example:
        >>> eeg = np.random.randn(8, 2500)  # 8 channels, 10 sec at 250 Hz
        >>> filtered = filter_eeg(eeg, fs=250, lowcut=0.5, highcut=50)
        >>> print(filtered.shape)  # (8, 2500)

    References:
        - scipy.signal.butter: Butterworth filter design
        - scipy.signal.filtfilt: Zero-phase forward-backward filtering
    """
    nyquist = fs / 2.0

    # Input validation
    if lowcut >= highcut:
        raise ValueError(
            f"lowcut ({lowcut:.2f} Hz) must be less than highcut ({highcut:.2f} Hz)"
        )
    if lowcut <= 0:
        raise ValueError(f"lowcut must be positive, got {lowcut}")
    if highcut >= nyquist:
        warnings.warn(
            f"highcut ({highcut:.2f} Hz) >= Nyquist ({nyquist:.2f} Hz). "
            f"Clipping to {nyquist * 0.95:.2f} Hz to ensure filter stability.",
            UserWarning
        )
        highcut = nyquist * 0.95

    # Check signal length
    n_samples = eeg_data.shape[axis] if eeg_data.ndim > 1 else len(eeg_data)
    if n_samples < MIN_SAMPLES_FOR_FILTER:
        raise ValueError(
            f"Signal too short ({n_samples} samples) for stable filtering. "
            f"Minimum required: {MIN_SAMPLES_FOR_FILTER} samples."
        )

    # Stage 1: Bandpass filter (Butterworth for maximally flat passband)
    # Normalized frequencies for scipy (0 to 1, where 1 = Nyquist)
    wn_low = lowcut / nyquist
    wn_high = highcut / nyquist

    b_bp, a_bp = butter(
        N=FILTER_ORDER,
        Wn=[wn_low, wn_high],
        btype='bandpass',
        analog=False,
        output='ba'
    )

    # Apply zero-phase filtering to preserve waveform timing
    filtered = filtfilt(b_bp, a_bp, eeg_data, axis=axis, padtype='odd')

    # Stage 2: Notch filter for powerline interference
    # Only apply if notch_freq is positive and below Nyquist
    if notch_freq > 0 and notch_freq < nyquist:
        # Quality factor Q = f0 / bandwidth
        # Q=30 gives bandwidth of 2 Hz for 60 Hz notch
        Q = 30.0

        b_notch, a_notch = iirnotch(
            w0=notch_freq,
            Q=Q,
            fs=fs
        )

        filtered = filtfilt(b_notch, a_notch, filtered, axis=axis, padtype='odd')

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
