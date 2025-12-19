"""
NeuroCardiac Shield — Simulated Device Adapter
================================================
Adapter that generates synthetic physiological data for development and testing.

This adapter uses the scientifically-grounded signal generators from
the signal_processing module to produce realistic EEG and ECG signals.

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
NYU Tandon School of Engineering | ECE-GY 9953 | Fall 2025
"""

import time
import numpy as np
from typing import Optional
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import DeviceAdapter, AdapterMetadata, AdapterState
from .gold_schema import GoldPacket, PacketQualityFlags


class SimulatedAdapter(DeviceAdapter):
    """
    Device adapter that generates synthetic physiological signals.

    This adapter wraps the EEGGenerator and ECGGenerator classes to
    produce scientifically-grounded synthetic data for development,
    testing, and demonstration purposes.

    Configuration:
        seed: Random seed for reproducibility (default: 42)
        sample_rate_hz: Sampling rate (default: 250 Hz)
        packet_rate_hz: Packet output rate (default: 10 Hz)
        heart_rate_bpm: Simulated heart rate (default: 72 BPM)
        hrv_state: One of 'normal', 'stressed', 'relaxed' (default: 'normal')

    Example:
        >>> adapter = SimulatedAdapter(seed=42)
        >>> adapter.connect()
        True
        >>> adapter.start_stream()
        >>> packet = adapter.read_packet()
        >>> print(f"Got {len(packet.eeg[0])} EEG samples")
        Got 25 EEG samples
    """

    def __init__(
        self,
        seed: int = 42,
        sample_rate_hz: float = 250.0,
        packet_rate_hz: float = 10.0,
        heart_rate_bpm: float = 72.0,
        hrv_state: str = 'normal',
        device_id: str = 'simulated-001',
    ):
        super().__init__()
        self.seed = seed
        self.sample_rate_hz = sample_rate_hz
        self.packet_rate_hz = packet_rate_hz
        self.samples_per_packet = int(sample_rate_hz / packet_rate_hz)
        self.heart_rate_bpm = heart_rate_bpm
        self.hrv_state = hrv_state
        self.device_id = device_id

        # Will be initialized on connect
        self._rng = None
        self._eeg_gen = None
        self._ecg_gen = None
        self._last_packet_time = 0
        self._time_offset = 0

    def connect(self) -> bool:
        """Initialize synthetic signal generators."""
        try:
            self._rng = np.random.RandomState(self.seed)
            self._state = AdapterState.CONNECTED
            self._time_offset = 0
            return True
        except Exception as e:
            self._state = AdapterState.ERROR
            raise ConnectionError(f"Failed to initialize simulator: {e}")

    def start_stream(self) -> None:
        """Begin generating synthetic data."""
        if self._state != AdapterState.CONNECTED:
            raise RuntimeError("Must connect before starting stream")

        self._last_packet_time = time.time()
        self._state = AdapterState.STREAMING

    def read_packet(self, timeout_ms: int = 1000) -> Optional[GoldPacket]:
        """
        Generate and return a synthetic data packet.

        This method simulates real-time data by enforcing the
        packet rate timing.
        """
        if self._state != AdapterState.STREAMING:
            raise RuntimeError("Must start stream before reading packets")

        # Enforce packet timing
        packet_interval = 1.0 / self.packet_rate_hz
        elapsed = time.time() - self._last_packet_time

        if elapsed < packet_interval:
            sleep_time = packet_interval - elapsed
            if sleep_time * 1000 > timeout_ms:
                return None
            time.sleep(sleep_time)

        self._last_packet_time = time.time()

        # Generate synthetic signals
        n_samples = self.samples_per_packet
        t = np.arange(n_samples) / self.sample_rate_hz + self._time_offset
        self._time_offset += n_samples / self.sample_rate_hz

        # Generate EEG (8 channels)
        eeg_data = self._generate_eeg(t)

        # Generate ECG (3 leads)
        ecg_data = self._generate_ecg(t)

        # Generate auxiliary sensors
        spo2 = 95 + self._rng.random() * 4  # 95-99%
        temp = 36.5 + self._rng.random() * 1.0  # 36.5-37.5°C
        accel = [
            (self._rng.random() - 0.5) * 0.1,  # Small motion
            (self._rng.random() - 0.5) * 0.1,
            0.98 + (self._rng.random() - 0.5) * 0.02,  # ~1g vertical
        ]

        # Build packet
        packet = GoldPacket(
            timestamp_ms=int(time.time() * 1000),
            device_id=self.device_id,
            packet_seq=self._next_seq(),
            sample_rate_hz=self.sample_rate_hz,
            eeg=eeg_data,
            ecg=ecg_data,
            spo2_percent=spo2,
            temp_celsius=temp,
            accel_xyz_g=accel,
            quality_flags=PacketQualityFlags.SYNTHETIC_DATA,
            extra={'hrv_state': self.hrv_state, 'heart_rate_bpm': self.heart_rate_bpm}
        )

        return packet

    def stop(self) -> None:
        """Stop the simulator."""
        self._state = AdapterState.DISCONNECTED
        self._eeg_gen = None
        self._ecg_gen = None

    def metadata(self) -> AdapterMetadata:
        """Return simulator metadata."""
        return AdapterMetadata(
            device_id=self.device_id,
            device_name="NeuroCardiac Simulator",
            adapter_type="simulated",
            sample_rate_hz=self.sample_rate_hz,
            samples_per_packet=self.samples_per_packet,
            packet_rate_hz=self.packet_rate_hz,
            eeg_channels=['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2'],
            ecg_leads=['I', 'II', 'III'],
            firmware_version="SIM-2.0.0",
        )

    def _generate_eeg(self, t: np.ndarray) -> list:
        """
        Generate synthetic 8-channel EEG data.

        Uses multi-band oscillation model with spatial weighting.
        References: Nunez & Srinivasan (2006), He et al. (2010)
        """
        n_samples = len(t)
        n_channels = 8
        eeg = np.zeros((n_channels, n_samples))

        # Band definitions (Hz): delta, theta, alpha, beta, gamma
        bands = {
            'delta': (0.5, 4, 20),    # freq range, amplitude µV
            'theta': (4, 8, 15),
            'alpha': (8, 13, 25),
            'beta': (13, 30, 8),
            'gamma': (30, 50, 3),
        }

        # Spatial weights for each channel (alpha stronger in occipital)
        spatial_weights = [0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5]

        for ch in range(n_channels):
            for band_name, (f_low, f_high, amp) in bands.items():
                # Random frequency within band
                freq = self._rng.uniform(f_low, f_high)
                phase = self._rng.uniform(0, 2 * np.pi)
                weight = spatial_weights[ch]

                # Alpha boost for occipital
                if band_name == 'alpha' and ch >= 6:
                    weight *= 1.5

                eeg[ch] += weight * amp * np.sin(2 * np.pi * freq * t + phase)

            # Add pink noise (1/f)
            white = self._rng.randn(n_samples)
            pink = np.cumsum(white) / 10
            pink = pink - np.mean(pink)
            eeg[ch] += pink * 5

        return eeg.tolist()

    def _generate_ecg(self, t: np.ndarray) -> list:
        """
        Generate synthetic 3-lead ECG data.

        Uses Gaussian-modulated PQRST morphology model.
        Reference: McSharry et al. (2003)
        """
        n_samples = len(t)
        n_leads = 3

        # Beat duration based on heart rate
        beat_duration = 60.0 / self.heart_rate_bpm

        # HRV variation based on state
        hrv_map = {'normal': 0.05, 'stressed': 0.02, 'relaxed': 0.08}
        hrv_var = hrv_map.get(self.hrv_state, 0.05)

        ecg = np.zeros((n_leads, n_samples))

        # PQRST component parameters (normalized to beat duration)
        # (center_fraction, width_fraction, amplitude_mV)
        components = {
            'P': (0.15, 0.04, 0.15),
            'Q': (0.28, 0.012, -0.10),
            'R': (0.32, 0.018, 1.30),
            'S': (0.36, 0.015, -0.20),
            'T': (0.55, 0.07, 0.25),
        }

        # Lead amplitude scaling (Einthoven's triangle approximation)
        lead_scale = [1.0, 1.1, 0.9]

        for lead in range(n_leads):
            beat_start = 0
            beat_idx = 0

            while beat_start < t[-1]:
                # Add HRV
                current_beat = beat_duration * (1 + self._rng.uniform(-hrv_var, hrv_var))

                for comp_name, (center_frac, width_frac, amp) in components.items():
                    center_time = beat_start + center_frac * current_beat
                    width = width_frac * current_beat

                    # Gaussian for each component
                    gaussian = amp * np.exp(-((t - center_time) ** 2) / (2 * width ** 2))
                    ecg[lead] += gaussian * lead_scale[lead]

                beat_start += current_beat
                beat_idx += 1

            # Add baseline wander (respiratory)
            ecg[lead] += 0.05 * np.sin(2 * np.pi * 0.2 * t)

            # Add noise
            ecg[lead] += self._rng.randn(n_samples) * 0.02

        return ecg.tolist()
