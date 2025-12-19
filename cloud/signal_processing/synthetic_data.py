"""
Synthetic Physiological Data Generation Module
===============================================

Generates scientifically accurate synthetic EEG and ECG signals for development,
testing, and ML training in the absence of real clinical data.

IMPORTANT DISCLAIMER:
---------------------
This module generates SIMULATED physiological signals based on published
literature parameters. These signals are for DEVELOPMENT AND TESTING ONLY
and do not represent real patient data. Clinical validation requires
real physiological recordings from IRB-approved studies.

Scientific Basis:
-----------------
EEG Generation:
- Frequency bands follow IFCN (International Federation of Clinical
  Neurophysiology) definitions: Delta (0.5-4 Hz), Theta (4-8 Hz),
  Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
- Amplitude ranges based on Nunez & Srinivasan (2006), "Electric Fields
  of the Brain": scalp EEG typically 10-100 µV
- 1/f spectral characteristics (pink noise) based on He et al. (2010),
  "The Temporal Structures and Functional Significance of Scale-free
  Brain Activity", Neuron 66(3):353-369
- Spatial correlations based on 10-20 electrode distance relationships

ECG Generation:
- PQRST morphology based on McSharry et al. (2003), "A Dynamical Model
  for Generating Synthetic Electrocardiogram Signals", IEEE Trans Biomed
  Eng 50(3):289-294
- Normal sinus rhythm: 60-100 BPM
- HRV parameters based on Task Force (1996), "Heart rate variability:
  standards of measurement, physiological interpretation and clinical
  use", Circulation 93(5):1043-1065

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 2.0.0
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
import warnings


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class EEGConfig:
    """Configuration for EEG signal generation.

    Attributes:
        fs: Sampling frequency in Hz. Standard clinical EEG uses 250-512 Hz.
        n_channels: Number of EEG channels (1-256 typical).
        duration_sec: Signal duration in seconds.
        seed: Random seed for reproducibility. None for random.

    Frequency Band Amplitudes (RMS µV):
        Based on normative values from Klimesch (1999) and Barry et al. (2007).
        Individual variation can be significant (±50%).
    """
    fs: float = 250.0
    n_channels: int = 8
    duration_sec: float = 60.0
    seed: Optional[int] = None

    # RMS amplitudes per band (µV) - empirically derived from literature
    delta_amplitude: float = 20.0   # 0.5-4 Hz, highest during sleep
    theta_amplitude: float = 15.0   # 4-8 Hz, drowsiness/memory
    alpha_amplitude: float = 25.0   # 8-13 Hz, relaxed wakefulness
    beta_amplitude: float = 8.0     # 13-30 Hz, active cognition
    gamma_amplitude: float = 3.0    # 30-100 Hz, perception/attention
    noise_amplitude: float = 5.0    # Background noise floor

    # Spatial configuration
    channel_names: Tuple[str, ...] = (
        'Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2'
    )

    # Alpha dominance by region (occipital highest)
    # Based on posterior alpha rhythm (PAR) distribution
    alpha_spatial_weights: Tuple[float, ...] = (
        0.5, 0.5,   # Frontal (Fp1, Fp2) - low alpha
        0.8, 0.8,   # Central (C3, C4) - moderate
        0.7, 0.7,   # Temporal (T3, T4) - moderate
        1.5, 1.5    # Occipital (O1, O2) - highest (alpha generators)
    )


@dataclass
class ECGConfig:
    """Configuration for ECG signal generation.

    Attributes:
        fs: Sampling frequency in Hz. Diagnostic ECG: 250-500 Hz.
        duration_sec: Signal duration in seconds.
        seed: Random seed for reproducibility.

    PQRST Parameters:
        Based on normal adult ECG morphology from Goldberger et al. (2017),
        "Goldberger's Clinical Electrocardiography", 9th Edition.

    HRV Parameters:
        Based on Task Force (1996) normal values for healthy adults at rest:
        - SDNN: 141 ± 39 ms (24h), ~50-100 ms (short-term)
        - RMSSD: 27 ± 12 ms
    """
    fs: float = 250.0
    duration_sec: float = 60.0
    seed: Optional[int] = None

    # Heart rate parameters
    mean_hr_bpm: float = 70.0       # Mean heart rate (BPM)
    hr_std_bpm: float = 5.0         # Heart rate variability (standard deviation)

    # HRV parameters (milliseconds)
    sdnn_target_ms: float = 50.0    # Target SDNN for short-term recording
    rmssd_target_ms: float = 30.0   # Target RMSSD (parasympathetic marker)

    # PQRST morphology parameters (mV and ms)
    p_amplitude_mv: float = 0.15     # P wave amplitude
    p_duration_ms: float = 80.0      # P wave duration
    pr_interval_ms: float = 160.0    # PR interval
    qrs_duration_ms: float = 90.0    # QRS complex duration
    q_amplitude_mv: float = 0.1      # Q wave amplitude (negative)
    r_amplitude_mv: float = 1.2      # R wave amplitude (main peak)
    s_amplitude_mv: float = 0.15     # S wave amplitude (negative)
    t_amplitude_mv: float = 0.25     # T wave amplitude
    t_duration_ms: float = 160.0     # T wave duration
    qt_interval_ms: float = 400.0    # QT interval

    # Noise and artifacts
    baseline_wander_hz: float = 0.15  # Respiratory modulation frequency
    baseline_wander_mv: float = 0.05  # Baseline wander amplitude
    noise_amplitude_mv: float = 0.02  # High-frequency noise


# =============================================================================
# EEG Signal Generation
# =============================================================================

class EEGGenerator:
    """
    Generates realistic synthetic multi-channel EEG signals.

    The generator creates EEG-like signals with:
    1. Proper 1/f spectral characteristics (pink noise background)
    2. Superimposed oscillatory activity in canonical frequency bands
    3. Realistic inter-channel correlations
    4. Optional eye blink artifacts

    Mathematical Model:
    -------------------
    For each channel c at time t:

    x_c(t) = sum_b [ A_b * w_c^b * sin(2π * f_b(t) * t + φ_c^b) ] + n_c(t)

    Where:
    - A_b: Amplitude for frequency band b
    - w_c^b: Spatial weight for channel c in band b
    - f_b(t): Time-varying frequency within band b (with jitter)
    - φ_c^b: Phase offset for channel c in band b
    - n_c(t): 1/f background noise

    Usage:
        >>> config = EEGConfig(fs=250, duration_sec=60, seed=42)
        >>> generator = EEGGenerator(config)
        >>> eeg_data, time_vector = generator.generate()
        >>> print(f"Shape: {eeg_data.shape}")  # (8, 15000)
    """

    # Frequency band definitions (Hz) - IFCN standard
    BANDS = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 100.0)
    }

    def __init__(self, config: EEGConfig):
        """
        Initialize EEG generator with configuration.

        Args:
            config: EEGConfig dataclass with generation parameters.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Validate configuration
        if config.n_channels != len(config.channel_names):
            raise ValueError(
                f"n_channels ({config.n_channels}) must match "
                f"channel_names length ({len(config.channel_names)})"
            )
        if config.fs < 100:
            warnings.warn(
                f"Sampling frequency {config.fs} Hz is below recommended "
                "minimum of 100 Hz for EEG. Nyquist frequency too low for "
                "beta/gamma activity."
            )

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic multi-channel EEG signal.

        Returns:
            eeg_data: Array of shape (n_channels, n_samples) in µV.
            time_vector: Time vector in seconds.

        Notes:
            - Output units are microvolts (µV)
            - Time vector starts at 0.0
            - Channels are ordered as specified in config.channel_names
        """
        n_samples = int(self.config.duration_sec * self.config.fs)
        time_vec = np.arange(n_samples) / self.config.fs

        eeg_data = np.zeros((self.config.n_channels, n_samples))

        # Generate each channel
        for ch_idx in range(self.config.n_channels):
            channel_signal = self._generate_channel(ch_idx, time_vec)
            eeg_data[ch_idx, :] = channel_signal

        # Add inter-channel correlations (spatial coherence)
        eeg_data = self._add_spatial_correlations(eeg_data)

        # Add eye blink artifacts to frontal channels
        eeg_data = self._add_eye_blinks(eeg_data, time_vec)

        return eeg_data, time_vec

    def _generate_channel(
        self,
        channel_idx: int,
        time_vec: np.ndarray
    ) -> np.ndarray:
        """Generate single EEG channel signal."""
        n_samples = len(time_vec)
        signal = np.zeros(n_samples)

        # Get spatial weight for alpha (channel-specific)
        alpha_weight = self.config.alpha_spatial_weights[channel_idx]

        # Generate each frequency band
        band_amplitudes = {
            'delta': self.config.delta_amplitude,
            'theta': self.config.theta_amplitude,
            'alpha': self.config.alpha_amplitude * alpha_weight,
            'beta': self.config.beta_amplitude,
            'gamma': self.config.gamma_amplitude
        }

        for band_name, (f_low, f_high) in self.BANDS.items():
            # Skip if band exceeds Nyquist
            nyquist = self.config.fs / 2.0
            if f_low >= nyquist:
                continue
            f_high = min(f_high, nyquist * 0.9)

            amplitude = band_amplitudes[band_name]
            band_signal = self._generate_band_oscillation(
                time_vec, f_low, f_high, amplitude
            )
            signal += band_signal

        # Add 1/f background noise
        noise = self._generate_pink_noise(n_samples)
        signal += noise * self.config.noise_amplitude

        return signal

    def _generate_band_oscillation(
        self,
        time_vec: np.ndarray,
        f_low: float,
        f_high: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Generate oscillatory activity within a frequency band.

        Uses multiple sinusoids with slightly varying frequencies
        to create naturalistic, non-stationary oscillations.
        """
        n_samples = len(time_vec)
        signal = np.zeros(n_samples)

        # Number of component oscillators (more for wider bands)
        band_width = f_high - f_low
        n_oscillators = max(3, int(band_width / 2))

        for _ in range(n_oscillators):
            # Random frequency within band
            freq = self.rng.uniform(f_low, f_high)

            # Random phase
            phase = self.rng.uniform(0, 2 * np.pi)

            # Slowly varying amplitude (amplitude modulation)
            am_freq = self.rng.uniform(0.05, 0.3)  # 0.05-0.3 Hz modulation
            am_phase = self.rng.uniform(0, 2 * np.pi)
            am_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * am_freq * time_vec + am_phase)

            # Generate oscillation
            oscillation = np.sin(2 * np.pi * freq * time_vec + phase)
            signal += oscillation * am_envelope

        # Normalize and scale
        signal = signal / n_oscillators * amplitude

        return signal

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """
        Generate 1/f (pink) noise using the Voss-McCartney algorithm.

        Pink noise has equal power per octave, matching the background
        spectral characteristics of real EEG.

        Reference:
            Voss, R. F. & Clarke, J. (1978). "1/f noise in music: Music from
            1/f noise". J Acoust Soc Am, 63(1), 258-263.
        """
        # Number of octaves
        n_octaves = int(np.log2(n_samples)) + 1

        # Initialize
        pink = np.zeros(n_samples)

        # Generate using octave band summation
        for octave in range(n_octaves):
            period = 2 ** octave
            n_values = n_samples // period + 1
            values = self.rng.standard_normal(n_values)

            # Repeat each value 'period' times
            repeated = np.repeat(values, period)[:n_samples]
            pink += repeated

        # Normalize
        pink = pink / np.std(pink)

        return pink

    def _add_spatial_correlations(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Add realistic inter-channel correlations based on electrode distance.

        Adjacent electrodes show higher correlation than distant ones.
        Based on volume conduction and source distribution.
        """
        # Simple correlation model: blend each channel with neighbors
        # Correlation decreases with electrode distance

        # Adjacency matrix for 10-20 subset
        # (Fp1-Fp2, C3-C4, T3-T4, O1-O2 are symmetric pairs)
        correlation_pairs = [
            (0, 1, 0.5),   # Fp1-Fp2 (contralateral, moderate correlation)
            (2, 3, 0.4),   # C3-C4
            (4, 5, 0.3),   # T3-T4
            (6, 7, 0.6),   # O1-O2 (high alpha coherence)
            (0, 2, 0.3),   # Fp1-C3 (ipsilateral)
            (1, 3, 0.3),   # Fp2-C4
            (2, 6, 0.4),   # C3-O1
            (3, 7, 0.4),   # C4-O2
        ]

        # Apply correlations by mixing channels
        for i, j, corr in correlation_pairs:
            if i < eeg_data.shape[0] and j < eeg_data.shape[0]:
                # Create correlated component
                shared = (eeg_data[i, :] + eeg_data[j, :]) / 2
                eeg_data[i, :] = (1 - corr) * eeg_data[i, :] + corr * shared
                eeg_data[j, :] = (1 - corr) * eeg_data[j, :] + corr * shared

        return eeg_data

    def _add_eye_blinks(
        self,
        eeg_data: np.ndarray,
        time_vec: np.ndarray
    ) -> np.ndarray:
        """
        Add realistic eye blink artifacts to frontal channels.

        Eye blinks produce large (~100-200 µV) deflections primarily
        visible in frontal electrodes (Fp1, Fp2), with decreasing
        amplitude toward posterior sites.

        Reference:
            Berg, P. & Scherg, M. (1994). "A fast method for forward
            computation of multiple-shell spherical head models",
            Electroencephalogr Clin Neurophysiol, 90(1), 58-64.
        """
        duration_sec = time_vec[-1] - time_vec[0]

        # Skip blinks for very short durations (< 2 seconds)
        if duration_sec < 2.0:
            return eeg_data

        # Typical blink rate: 15-20 per minute
        n_blinks = int(self.rng.poisson(duration_sec * 17 / 60))

        if n_blinks == 0:
            return eeg_data

        # Blink timing (uniform with minimum interval)
        blink_times = np.sort(self.rng.uniform(0.5, duration_sec - 0.5, n_blinks))

        # Enforce minimum inter-blink interval (~150 ms)
        min_interval = 0.15
        filtered_times = [blink_times[0]] if len(blink_times) > 0 else []
        for t in blink_times[1:]:
            if t - filtered_times[-1] > min_interval:
                filtered_times.append(t)
        blink_times = np.array(filtered_times)

        # Blink artifact shape (Gaussian envelope)
        blink_duration_samples = int(0.15 * self.config.fs)  # ~150 ms
        blink_template = np.exp(-((np.arange(blink_duration_samples) -
                                   blink_duration_samples/2)**2) /
                                 (2 * (blink_duration_samples/6)**2))
        blink_template = blink_template / np.max(blink_template)

        # Spatial distribution: strongest at Fp1/Fp2, decreasing posteriorly
        spatial_weights = np.array([1.0, 1.0, 0.4, 0.4, 0.2, 0.2, 0.05, 0.05])

        for blink_time in blink_times:
            blink_idx = int(blink_time * self.config.fs)
            blink_amplitude = self.rng.uniform(80, 200)  # 80-200 µV

            start_idx = blink_idx - blink_duration_samples // 2
            end_idx = start_idx + blink_duration_samples

            if start_idx >= 0 and end_idx < eeg_data.shape[1]:
                for ch in range(min(len(spatial_weights), eeg_data.shape[0])):
                    eeg_data[ch, start_idx:end_idx] += (
                        blink_template * blink_amplitude * spatial_weights[ch]
                    )

        return eeg_data


# =============================================================================
# ECG Signal Generation
# =============================================================================

class ECGGenerator:
    """
    Generates realistic synthetic ECG signals with proper PQRST morphology.

    Based on the dynamical model from McSharry et al. (2003) with
    simplifications for computational efficiency. Generates single-lead
    ECG that can be scaled for 3-lead simulation.

    The generator creates:
    1. Realistic PQRST complexes with appropriate timing
    2. Heart rate variability following physiological patterns
    3. Baseline wander from simulated respiration
    4. High-frequency noise (muscle artifact, powerline)

    Mathematical Model:
    -------------------
    The ECG waveform is constructed using Gaussian pulses:

    z(t) = sum_i [ a_i * exp(-(θ_i - θ(t))² / (2*b_i²)) ]

    Where:
    - a_i: Amplitude of component i (P, Q, R, S, T)
    - b_i: Width parameter of component i
    - θ_i: Angular position of component i in the cardiac cycle
    - θ(t): Current phase in cardiac cycle

    Usage:
        >>> config = ECGConfig(fs=250, duration_sec=60, seed=42)
        >>> generator = ECGGenerator(config)
        >>> ecg_data, time_vector, r_peaks = generator.generate()
        >>> print(f"Detected {len(r_peaks)} R-peaks")
    """

    def __init__(self, config: ECGConfig):
        """
        Initialize ECG generator with configuration.

        Args:
            config: ECGConfig dataclass with generation parameters.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Validate configuration
        if config.mean_hr_bpm < 30 or config.mean_hr_bpm > 200:
            warnings.warn(
                f"Mean heart rate {config.mean_hr_bpm} BPM is outside "
                "normal physiological range (30-200 BPM)."
            )

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic single-lead ECG signal.

        Returns:
            ecg_data: ECG signal array (n_samples,) in mV.
            time_vector: Time vector in seconds.
            r_peak_indices: Indices of R-peak locations.

        Notes:
            - Output units are millivolts (mV)
            - R-peaks are guaranteed to be accurate (they're generated,
              not detected)
        """
        n_samples = int(self.config.duration_sec * self.config.fs)
        time_vec = np.arange(n_samples) / self.config.fs

        # Generate RR intervals with HRV
        rr_intervals_ms, r_peak_times = self._generate_rr_intervals()

        # Convert R-peak times to sample indices
        r_peak_indices = (r_peak_times * self.config.fs).astype(int)
        r_peak_indices = r_peak_indices[r_peak_indices < n_samples]

        # Generate PQRST morphology for each beat
        ecg_data = self._generate_pqrst_sequence(time_vec, r_peak_times)

        # Add baseline wander (respiratory modulation)
        baseline = self._generate_baseline_wander(time_vec)
        ecg_data += baseline

        # Add high-frequency noise
        noise = self.rng.normal(0, self.config.noise_amplitude_mv, n_samples)
        ecg_data += noise

        return ecg_data, time_vec, r_peak_indices

    def generate_3lead(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3-lead ECG (Lead I, II, III) from single-lead template.

        Uses Einthoven's law for lead relationships:
        - Lead II = Lead I + Lead III
        - Lead I: Left arm - Right arm
        - Lead II: Left leg - Right arm (largest R-wave typically)
        - Lead III: Left leg - Left arm

        Returns:
            ecg_3lead: ECG array (3, n_samples) in mV.
            time_vector: Time vector in seconds.
            r_peak_indices: Indices of R-peak locations.
        """
        # Generate base ECG (similar to Lead II)
        ecg_lead2, time_vec, r_peaks = self.generate()

        n_samples = len(ecg_lead2)
        ecg_3lead = np.zeros((3, n_samples))

        # Lead II is the template
        ecg_3lead[1, :] = ecg_lead2

        # Lead I: smaller amplitude, slightly different morphology
        # Approximate ratio from typical ECG: Lead I ~ 0.6 * Lead II
        ecg_3lead[0, :] = ecg_lead2 * 0.65 + self.rng.normal(0, 0.01, n_samples)

        # Lead III: Einthoven's law - Lead III = Lead II - Lead I
        ecg_3lead[2, :] = ecg_3lead[1, :] - ecg_3lead[0, :]

        return ecg_3lead, time_vec, r_peaks

    def _generate_rr_intervals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate RR intervals with realistic heart rate variability.

        Uses a multi-scale approach combining:
        1. Low-frequency (LF) oscillation: 0.04-0.15 Hz (sympathetic)
        2. High-frequency (HF) oscillation: 0.15-0.4 Hz (parasympathetic/respiratory)
        3. Very low frequency (VLF) trend: < 0.04 Hz

        Returns:
            rr_intervals_ms: Array of RR intervals in milliseconds.
            r_peak_times: Array of R-peak times in seconds.
        """
        # Mean RR interval
        mean_rr_ms = 60000.0 / self.config.mean_hr_bpm

        # Estimate number of beats
        approx_n_beats = int(self.config.duration_sec * self.config.mean_hr_bpm / 60) + 10

        # Generate time series for modulation
        t_mod = np.linspace(0, self.config.duration_sec, approx_n_beats)

        # LF component (0.04-0.15 Hz) - baroreflex, sympathetic
        lf_freq = self.rng.uniform(0.08, 0.12)
        lf_amplitude = self.config.sdnn_target_ms * 0.5  # ~50% of total HRV
        lf_component = lf_amplitude * np.sin(2 * np.pi * lf_freq * t_mod +
                                              self.rng.uniform(0, 2*np.pi))

        # HF component (0.15-0.4 Hz) - respiratory sinus arrhythmia
        hf_freq = self.rng.uniform(0.2, 0.3)  # ~12-18 breaths/min
        hf_amplitude = self.config.rmssd_target_ms * 0.7  # Related to RMSSD
        hf_component = hf_amplitude * np.sin(2 * np.pi * hf_freq * t_mod +
                                              self.rng.uniform(0, 2*np.pi))

        # VLF component (very slow drift)
        vlf_freq = self.rng.uniform(0.01, 0.03)
        vlf_amplitude = self.config.sdnn_target_ms * 0.3
        vlf_component = vlf_amplitude * np.sin(2 * np.pi * vlf_freq * t_mod +
                                                self.rng.uniform(0, 2*np.pi))

        # Random component (white noise)
        random_component = self.rng.normal(0, self.config.sdnn_target_ms * 0.2,
                                           approx_n_beats)

        # Combine all components
        rr_modulation = lf_component + hf_component + vlf_component + random_component
        rr_intervals_ms = mean_rr_ms + rr_modulation

        # Ensure physiological limits (300-2000 ms, i.e., 30-200 BPM)
        rr_intervals_ms = np.clip(rr_intervals_ms, 300, 2000)

        # Convert to cumulative times for R-peak positions
        r_peak_times = np.cumsum(rr_intervals_ms) / 1000.0  # Convert to seconds

        # Filter to only include beats within duration
        valid_mask = r_peak_times <= self.config.duration_sec
        rr_intervals_ms = rr_intervals_ms[valid_mask]
        r_peak_times = r_peak_times[valid_mask]

        return rr_intervals_ms, r_peak_times

    def _generate_pqrst_sequence(
        self,
        time_vec: np.ndarray,
        r_peak_times: np.ndarray
    ) -> np.ndarray:
        """
        Generate full ECG signal with PQRST complexes at each beat.

        Uses Gaussian pulses to model each wave component.
        """
        n_samples = len(time_vec)
        ecg = np.zeros(n_samples)

        # Convert timing parameters from ms to seconds
        p_duration_s = self.config.p_duration_ms / 1000.0
        pr_interval_s = self.config.pr_interval_ms / 1000.0
        qrs_duration_s = self.config.qrs_duration_ms / 1000.0
        t_duration_s = self.config.t_duration_ms / 1000.0
        qt_interval_s = self.config.qt_interval_ms / 1000.0

        for r_time in r_peak_times:
            if r_time < 0.2 or r_time > self.config.duration_sec - 0.3:
                continue  # Skip edge beats

            # Calculate positions relative to R-peak
            # P wave: PR interval before R
            p_center = r_time - pr_interval_s + p_duration_s/2

            # Q wave: Just before R
            q_center = r_time - qrs_duration_s * 0.3

            # R wave: At r_time
            r_center = r_time

            # S wave: Just after R
            s_center = r_time + qrs_duration_s * 0.3

            # T wave: QT interval from Q, centered
            t_center = r_time + qt_interval_s * 0.7

            # Add each wave component as Gaussian
            ecg += self._gaussian_pulse(
                time_vec, p_center, p_duration_s/4, self.config.p_amplitude_mv
            )
            ecg += self._gaussian_pulse(
                time_vec, q_center, qrs_duration_s/8, -self.config.q_amplitude_mv
            )
            ecg += self._gaussian_pulse(
                time_vec, r_center, qrs_duration_s/6, self.config.r_amplitude_mv
            )
            ecg += self._gaussian_pulse(
                time_vec, s_center, qrs_duration_s/8, -self.config.s_amplitude_mv
            )
            ecg += self._gaussian_pulse(
                time_vec, t_center, t_duration_s/3, self.config.t_amplitude_mv
            )

        return ecg

    def _gaussian_pulse(
        self,
        time_vec: np.ndarray,
        center: float,
        width: float,
        amplitude: float
    ) -> np.ndarray:
        """Generate Gaussian pulse centered at given time."""
        return amplitude * np.exp(-((time_vec - center) ** 2) / (2 * width ** 2))

    def _generate_baseline_wander(self, time_vec: np.ndarray) -> np.ndarray:
        """
        Generate baseline wander from respiratory modulation.

        Respiratory sinus arrhythmia causes baseline to shift with breathing.
        Typical respiratory rate: 12-20 breaths/min (0.2-0.33 Hz).
        """
        resp_freq = self.config.baseline_wander_hz
        baseline = self.config.baseline_wander_mv * np.sin(
            2 * np.pi * resp_freq * time_vec + self.rng.uniform(0, 2*np.pi)
        )

        # Add very slow drift (electrode shift, skin impedance changes)
        drift_freq = self.rng.uniform(0.01, 0.03)
        drift = self.config.baseline_wander_mv * 0.5 * np.sin(
            2 * np.pi * drift_freq * time_vec + self.rng.uniform(0, 2*np.pi)
        )

        return baseline + drift


# =============================================================================
# Data Generation Utilities
# =============================================================================

def generate_training_dataset(
    n_samples: int = 1000,
    duration_per_sample_sec: float = 10.0,
    fs: float = 250.0,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate complete training dataset with EEG, ECG, and labels.

    Generates samples with varying characteristics to create a diverse
    training set. Labels are generated based on simulated physiological
    state (normal, stressed, fatigued).

    Args:
        n_samples: Number of training samples to generate.
        duration_per_sample_sec: Duration of each sample in seconds.
        fs: Sampling frequency in Hz.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
        - 'eeg': Array (n_samples, n_channels, n_timepoints)
        - 'ecg': Array (n_samples, n_leads, n_timepoints)
        - 'labels': Risk labels (0=LOW, 1=MEDIUM, 2=HIGH)
        - 'features': Dict of extracted feature arrays
        - 'metadata': Generation metadata

    IMPORTANT:
        Labels are based on SIMULATED physiological state, not clinical
        ground truth. These labels are for DEVELOPMENT ONLY and do not
        represent validated clinical risk categories.
    """
    rng = np.random.default_rng(seed)

    n_timepoints = int(duration_per_sample_sec * fs)
    n_eeg_channels = 8
    n_ecg_leads = 3

    eeg_data = np.zeros((n_samples, n_eeg_channels, n_timepoints))
    ecg_data = np.zeros((n_samples, n_ecg_leads, n_timepoints))
    labels = np.zeros(n_samples, dtype=int)

    # Simulated physiological states with associated parameters
    states = {
        0: {  # LOW risk - relaxed, healthy
            'alpha_amplitude': 35.0,
            'beta_amplitude': 5.0,
            'mean_hr': 65.0,
            'sdnn': 60.0,
            'rmssd': 40.0
        },
        1: {  # MEDIUM risk - mild stress/fatigue
            'alpha_amplitude': 20.0,
            'beta_amplitude': 12.0,
            'mean_hr': 80.0,
            'sdnn': 35.0,
            'rmssd': 25.0
        },
        2: {  # HIGH risk - significant stress/distress
            'alpha_amplitude': 12.0,
            'beta_amplitude': 18.0,
            'mean_hr': 95.0,
            'sdnn': 20.0,
            'rmssd': 15.0
        }
    }

    # Class distribution: 60% LOW, 30% MEDIUM, 10% HIGH
    # (imbalanced, as expected in real monitoring scenario)
    class_probs = [0.6, 0.3, 0.1]

    for i in range(n_samples):
        # Assign label based on distribution
        label = rng.choice([0, 1, 2], p=class_probs)
        labels[i] = label

        # Get state parameters with some variation
        params = states[label]

        # EEG configuration
        eeg_config = EEGConfig(
            fs=fs,
            n_channels=n_eeg_channels,
            duration_sec=duration_per_sample_sec,
            seed=seed + i,
            alpha_amplitude=params['alpha_amplitude'] * rng.uniform(0.8, 1.2),
            beta_amplitude=params['beta_amplitude'] * rng.uniform(0.8, 1.2)
        )

        # ECG configuration
        ecg_config = ECGConfig(
            fs=fs,
            duration_sec=duration_per_sample_sec,
            seed=seed + i + n_samples,
            mean_hr_bpm=params['mean_hr'] * rng.uniform(0.9, 1.1),
            sdnn_target_ms=params['sdnn'] * rng.uniform(0.8, 1.2),
            rmssd_target_ms=params['rmssd'] * rng.uniform(0.8, 1.2)
        )

        # Generate signals
        eeg_gen = EEGGenerator(eeg_config)
        eeg_signal, _ = eeg_gen.generate()
        eeg_data[i, :, :] = eeg_signal[:, :n_timepoints]

        ecg_gen = ECGGenerator(ecg_config)
        ecg_signal, _, _ = ecg_gen.generate_3lead()
        ecg_data[i, :, :] = ecg_signal[:, :n_timepoints]

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")

    return {
        'eeg': eeg_data,
        'ecg': ecg_data,
        'labels': labels,
        'metadata': {
            'n_samples': n_samples,
            'duration_sec': duration_per_sample_sec,
            'fs': fs,
            'seed': seed,
            'n_eeg_channels': n_eeg_channels,
            'n_ecg_leads': n_ecg_leads,
            'class_distribution': np.bincount(labels).tolist(),
            'disclaimer': (
                'SIMULATED DATA - Not derived from real patients. '
                'Labels based on artificial physiological state modeling. '
                'For development and testing only.'
            )
        }
    }


# =============================================================================
# Example Usage and Validation
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Physiological Data Generator - Validation")
    print("=" * 70)

    # Test EEG generation
    print("\n[1] Testing EEG Generation...")
    eeg_config = EEGConfig(fs=250, duration_sec=10, seed=42)
    eeg_gen = EEGGenerator(eeg_config)
    eeg_data, eeg_time = eeg_gen.generate()

    print(f"    EEG shape: {eeg_data.shape}")
    print(f"    EEG range: [{eeg_data.min():.1f}, {eeg_data.max():.1f}] µV")
    print(f"    EEG std per channel: {eeg_data.std(axis=1).round(1)}")

    # Verify spectral content
    from scipy.signal import welch
    freqs, psd = welch(eeg_data[6, :], fs=250, nperseg=512)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])
    print(f"    O1 Alpha power (8-13 Hz): {alpha_power:.1f} µV²")

    # Test ECG generation
    print("\n[2] Testing ECG Generation...")
    ecg_config = ECGConfig(fs=250, duration_sec=60, seed=42)
    ecg_gen = ECGGenerator(ecg_config)
    ecg_data, ecg_time, r_peaks = ecg_gen.generate()

    print(f"    ECG shape: {ecg_data.shape}")
    print(f"    ECG range: [{ecg_data.min():.3f}, {ecg_data.max():.3f}] mV")
    print(f"    R-peaks detected: {len(r_peaks)}")

    # Calculate actual HRV
    rr_intervals_ms = np.diff(r_peaks) / 250 * 1000
    actual_sdnn = np.std(rr_intervals_ms)
    actual_rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))
    print(f"    Actual SDNN: {actual_sdnn:.1f} ms (target: {ecg_config.sdnn_target_ms})")
    print(f"    Actual RMSSD: {actual_rmssd:.1f} ms (target: {ecg_config.rmssd_target_ms})")

    # Test 3-lead generation
    print("\n[3] Testing 3-Lead ECG Generation...")
    ecg_3lead, _, _ = ecg_gen.generate_3lead()
    print(f"    3-Lead ECG shape: {ecg_3lead.shape}")

    # Test dataset generation
    print("\n[4] Testing Dataset Generation (small batch)...")
    dataset = generate_training_dataset(n_samples=10, duration_per_sample_sec=5, seed=42)
    print(f"    EEG batch shape: {dataset['eeg'].shape}")
    print(f"    ECG batch shape: {dataset['ecg'].shape}")
    print(f"    Labels distribution: {np.bincount(dataset['labels'])}")

    print("\n" + "=" * 70)
    print("Validation COMPLETE - All generators functional")
    print("=" * 70)
