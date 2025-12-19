"""
NeuroCardiac Shield — Device Adapter Base Interface
=====================================================
Abstract base class defining the contract for all device adapters.

All adapters (Simulated, BLE, Serial) must implement this interface
to ensure consistent behavior across the system.

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
NYU Tandon School of Engineering | ECE-GY 9953 | Fall 2025
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

from .gold_schema import GoldPacket


class AdapterState(Enum):
    """Device adapter connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class AdapterMetadata:
    """
    Metadata about the connected device.

    This provides information that downstream processing
    may need to properly interpret the data.
    """
    device_id: str
    device_name: str
    adapter_type: str

    # Sampling characteristics
    sample_rate_hz: float
    samples_per_packet: int
    packet_rate_hz: float

    # Channel configuration
    eeg_channels: List[str]  # e.g., ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']
    ecg_leads: List[str]     # e.g., ['I', 'II', 'III']

    # Units
    eeg_unit: str = "µV"
    ecg_unit: str = "mV"
    temp_unit: str = "°C"
    accel_unit: str = "g"

    # Device capabilities
    has_spo2: bool = True
    has_temperature: bool = True
    has_accelerometer: bool = True

    # Version info
    firmware_version: str = "unknown"
    adapter_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'device_id': self.device_id,
            'device_name': self.device_name,
            'adapter_type': self.adapter_type,
            'sample_rate_hz': self.sample_rate_hz,
            'samples_per_packet': self.samples_per_packet,
            'packet_rate_hz': self.packet_rate_hz,
            'eeg_channels': self.eeg_channels,
            'ecg_leads': self.ecg_leads,
            'eeg_unit': self.eeg_unit,
            'ecg_unit': self.ecg_unit,
            'temp_unit': self.temp_unit,
            'accel_unit': self.accel_unit,
            'has_spo2': self.has_spo2,
            'has_temperature': self.has_temperature,
            'has_accelerometer': self.has_accelerometer,
            'firmware_version': self.firmware_version,
            'adapter_version': self.adapter_version,
        }


class DeviceAdapter(ABC):
    """
    Abstract base class for device adapters.

    All device adapters must implement these methods to provide
    a consistent interface for the processing pipeline.

    Lifecycle:
        1. Create adapter: adapter = MyAdapter(**config)
        2. Connect: adapter.connect()
        3. Start streaming: adapter.start_stream()
        4. Read packets: packet = adapter.read_packet()
        5. Stop: adapter.stop()

    Example:
        >>> adapter = SimulatedAdapter(seed=42)
        >>> adapter.connect()
        True
        >>> adapter.start_stream()
        >>> while running:
        ...     packet = adapter.read_packet()
        ...     process(packet)
        >>> adapter.stop()
    """

    def __init__(self):
        self._state = AdapterState.DISCONNECTED
        self._packet_seq = 0

    @property
    def state(self) -> AdapterState:
        """Current adapter state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._state in (AdapterState.CONNECTED, AdapterState.STREAMING)

    @property
    def is_streaming(self) -> bool:
        """Check if adapter is actively streaming data."""
        return self._state == AdapterState.STREAMING

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the device.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: If connection fails critically.
        """
        pass

    @abstractmethod
    def start_stream(self) -> None:
        """
        Begin data streaming from the device.

        Must be called after connect() succeeds.

        Raises:
            RuntimeError: If not connected.
        """
        pass

    @abstractmethod
    def read_packet(self, timeout_ms: int = 1000) -> Optional[GoldPacket]:
        """
        Read the next data packet from the device.

        Args:
            timeout_ms: Maximum time to wait for packet in milliseconds.

        Returns:
            GoldPacket if data available, None if timeout.

        Raises:
            RuntimeError: If not streaming.
            IOError: If read fails.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop streaming and disconnect from device.

        Safe to call multiple times.
        """
        pass

    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        """
        Get device metadata.

        Returns:
            AdapterMetadata with device information.
        """
        pass

    def _next_seq(self) -> int:
        """Get next packet sequence number."""
        seq = self._packet_seq
        self._packet_seq += 1
        return seq

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        self.start_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
