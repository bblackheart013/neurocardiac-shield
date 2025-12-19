"""
NeuroCardiac Shield — Gold Schema
==================================
Canonical internal data schema that all device adapters convert into.

This is the "single source of truth" for physiological packet format.
All adapters MUST convert their raw data into this schema before
passing to the processing pipeline.

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
NYU Tandon School of Engineering | ECE-GY 9953 | Fall 2025
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import IntFlag
import time
import json


class PacketQualityFlags(IntFlag):
    """
    Bit flags indicating data quality issues.

    These flags help downstream processing decide whether to trust
    certain aspects of the data.
    """
    NONE = 0
    EEG_SATURATION = 1 << 0       # One or more EEG channels saturated
    ECG_SATURATION = 1 << 1       # ECG signal saturated
    EEG_LEAD_OFF = 1 << 2         # EEG electrode disconnected
    ECG_LEAD_OFF = 1 << 3         # ECG electrode disconnected
    LOW_BATTERY = 1 << 4          # Device battery < 20%
    MOTION_ARTIFACT = 1 << 5      # High accelerometer activity
    POOR_SIGNAL = 1 << 6          # General signal quality warning
    CHECKSUM_INVALID = 1 << 7     # Packet checksum failed
    SYNTHETIC_DATA = 1 << 8       # Data is simulated (not real sensors)


@dataclass
class GoldPacket:
    """
    Canonical physiological data packet.

    This is the unified format that ALL device adapters must produce.
    The processing pipeline ONLY accepts this format.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds
        device_id: Unique device identifier string
        packet_seq: Monotonically increasing sequence number
        sample_rate_hz: Sampling rate of the data (typically 250 Hz)

        eeg: 8-channel EEG data in microvolts (µV)
             Order: [Fp1, Fp2, C3, C4, T3, T4, O1, O2]
             Expected range: -500 to +500 µV

        ecg: 3-lead ECG data in millivolts (mV)
             Order: [Lead I, Lead II, Lead III]
             Expected range: -2.0 to +3.0 mV

        spo2_percent: Blood oxygen saturation (0-100%)
        temp_celsius: Body temperature in Celsius
        accel_xyz_g: 3-axis accelerometer in g-force units

        quality_flags: Bit flags indicating data quality issues
        extra: Optional dictionary for adapter-specific metadata
    """

    # Timing
    timestamp_ms: int
    device_id: str
    packet_seq: int
    sample_rate_hz: float

    # EEG: 8 channels × N samples per packet (typically 25 samples @ 10 Hz packet rate)
    eeg: List[List[float]]  # Shape: [8][n_samples], units: µV

    # ECG: 3 leads × N samples per packet
    ecg: List[List[float]]  # Shape: [3][n_samples], units: mV

    # Auxiliary sensors
    spo2_percent: float = 0.0
    temp_celsius: float = 0.0
    accel_xyz_g: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Quality
    quality_flags: PacketQualityFlags = PacketQualityFlags.NONE

    # Extensibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['quality_flags'] = int(self.quality_flags)
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldPacket':
        """Create from dictionary."""
        data = data.copy()
        data['quality_flags'] = PacketQualityFlags(data.get('quality_flags', 0))
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'GoldPacket':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> List[str]:
        """
        Validate packet data.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check EEG shape
        if len(self.eeg) != 8:
            errors.append(f"EEG must have 8 channels, got {len(self.eeg)}")

        # Check ECG shape
        if len(self.ecg) != 3:
            errors.append(f"ECG must have 3 leads, got {len(self.ecg)}")

        # Check sample counts match
        if self.eeg and self.ecg:
            eeg_samples = len(self.eeg[0]) if self.eeg[0] else 0
            ecg_samples = len(self.ecg[0]) if self.ecg[0] else 0
            if eeg_samples != ecg_samples:
                errors.append(
                    f"Sample count mismatch: EEG={eeg_samples}, ECG={ecg_samples}"
                )

        # Check EEG amplitude range
        for ch_idx, ch_data in enumerate(self.eeg):
            for val in ch_data:
                if abs(val) > 1000:
                    errors.append(
                        f"EEG channel {ch_idx} value {val} µV exceeds ±1000 µV range"
                    )
                    break

        # Check ECG amplitude range
        for lead_idx, lead_data in enumerate(self.ecg):
            for val in lead_data:
                if abs(val) > 10:
                    errors.append(
                        f"ECG lead {lead_idx} value {val} mV exceeds ±10 mV range"
                    )
                    break

        # Check SpO2 range
        if not (0 <= self.spo2_percent <= 100):
            errors.append(f"SpO2 {self.spo2_percent}% out of 0-100 range")

        # Check temperature range
        if not (20 <= self.temp_celsius <= 45):
            errors.append(
                f"Temperature {self.temp_celsius}°C out of physiological range"
            )

        return errors

    @staticmethod
    def create_empty(device_id: str = "unknown") -> 'GoldPacket':
        """Create an empty/zeroed packet for initialization."""
        return GoldPacket(
            timestamp_ms=int(time.time() * 1000),
            device_id=device_id,
            packet_seq=0,
            sample_rate_hz=250.0,
            eeg=[[0.0] * 25 for _ in range(8)],
            ecg=[[0.0] * 25 for _ in range(3)],
            spo2_percent=0.0,
            temp_celsius=0.0,
            accel_xyz_g=[0.0, 0.0, 0.0],
            quality_flags=PacketQualityFlags.NONE,
        )


# Channel name mappings
EEG_CHANNEL_NAMES = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
ECG_LEAD_NAMES = ['Lead I', 'Lead II', 'Lead III']

# Expected ranges
EEG_AMPLITUDE_RANGE_UV = (-500, 500)
ECG_AMPLITUDE_RANGE_MV = (-2.0, 3.0)
SPO2_RANGE_PERCENT = (0, 100)
TEMPERATURE_RANGE_C = (30, 42)
