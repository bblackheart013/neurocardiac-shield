"""
NeuroCardiac Shield — Serial Device Adapter
=============================================
Adapter for microcontrollers streaming data over UART/Serial connection.

This adapter is useful for:
- Arduino/ESP32 prototypes
- Direct USB connection to development boards
- Wired debugging during hardware development

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
NYU Tandon School of Engineering | ECE-GY 9953 | Fall 2025
"""

import time
import struct
import threading
from typing import Optional, List
from dataclasses import dataclass
from queue import Queue, Empty

from .base import DeviceAdapter, AdapterMetadata, AdapterState
from .gold_schema import GoldPacket, PacketQualityFlags


@dataclass
class SerialConfig:
    """
    Configuration for serial device connection.

    Attributes:
        port: Serial port path (e.g., "/dev/ttyUSB0", "COM3")
        baudrate: Communication speed (default: 115200)
        timeout_sec: Read timeout in seconds (default: 1.0)
        packet_format: One of 'binary' or 'json' (default: 'binary')
    """
    port: str
    baudrate: int = 115200
    timeout_sec: float = 1.0
    packet_format: str = 'binary'  # 'binary' or 'json'


class SerialAdapter(DeviceAdapter):
    """
    Serial/UART device adapter for wired microcontroller connections.

    This adapter supports two packet formats:

    1. Binary Format (569 bytes) - Same as BLE:
       Efficient, low overhead, matches firmware output directly.

    2. JSON Format (variable length):
       Human-readable, easier to debug, higher overhead.
       Example:
       {
           "ts": 1234567890,
           "seq": 42,
           "eeg": [[...], [...], ...],
           "ecg": [[...], [...], [...]],
           "spo2": 98.5,
           "temp": 37.2,
           "accel": [0.01, -0.02, 0.98]
       }

    Hardware Setup:
    ---------------
    For Arduino/ESP32:
        1. Connect TX -> USB-Serial RX
        2. Connect GND -> USB-Serial GND
        3. Set matching baudrate (default 115200)

    Example:
        >>> config = SerialConfig(port="/dev/ttyUSB0", baudrate=115200)
        >>> adapter = SerialAdapter(config)
        >>> adapter.connect()
        True
        >>> adapter.start_stream()
        >>> packet = adapter.read_packet()
    """

    PACKET_SIZE_BINARY = 569
    SYNC_BYTE = 0xAA
    END_BYTE = 0x55

    def __init__(self, config: Optional[SerialConfig] = None):
        super().__init__()
        self.config = config or SerialConfig(port="")
        self._serial = None
        self._read_thread = None
        self._packet_queue = Queue(maxsize=100)
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """
        Open serial port connection.

        Requires pyserial: pip install pyserial
        """
        if not self.config.port:
            raise ValueError(
                "Serial port not configured. "
                "Create SerialConfig with port='/dev/ttyUSB0' or similar."
            )

        try:
            import serial
        except ImportError:
            raise ImportError(
                "pyserial library required. Install with: pip install pyserial"
            )

        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout_sec,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )

            # Clear any pending data
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            self._state = AdapterState.CONNECTED
            return True

        except Exception as e:
            self._state = AdapterState.ERROR
            raise ConnectionError(f"Failed to open serial port: {e}")

    def start_stream(self) -> None:
        """Start background thread to read serial data."""
        if self._state != AdapterState.CONNECTED:
            raise RuntimeError("Must connect before starting stream")

        self._stop_event.clear()

        # Start read thread
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

        # Send start command to device
        if self._serial:
            self._serial.write(b'START\n')

        self._state = AdapterState.STREAMING

    def read_packet(self, timeout_ms: int = 1000) -> Optional[GoldPacket]:
        """Get next packet from the queue."""
        if self._state != AdapterState.STREAMING:
            raise RuntimeError("Must start stream before reading packets")

        try:
            return self._packet_queue.get(timeout=timeout_ms / 1000.0)
        except Empty:
            return None

    def stop(self) -> None:
        """Stop streaming and close serial port."""
        self._stop_event.set()

        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b'STOP\n')
            except Exception:
                pass
            self._serial.close()

        if self._read_thread:
            self._read_thread.join(timeout=2.0)

        self._serial = None
        self._state = AdapterState.DISCONNECTED

    def metadata(self) -> AdapterMetadata:
        """Return serial device metadata."""
        return AdapterMetadata(
            device_id=f"serial-{self.config.port}",
            device_name="Serial Device",
            adapter_type="serial",
            sample_rate_hz=250.0,
            samples_per_packet=25,
            packet_rate_hz=10.0,
            eeg_channels=['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2'],
            ecg_leads=['I', 'II', 'III'],
            firmware_version="unknown",
        )

    def _read_loop(self) -> None:
        """Background thread to continuously read serial data."""
        buffer = b''

        while not self._stop_event.is_set():
            try:
                if not self._serial or not self._serial.is_open:
                    break

                # Read available data
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    buffer += data

                    # Process buffer based on format
                    if self.config.packet_format == 'binary':
                        buffer = self._process_binary_buffer(buffer)
                    else:
                        buffer = self._process_json_buffer(buffer)
                else:
                    time.sleep(0.01)  # Small delay if no data

            except Exception as e:
                # Log error but continue trying
                print(f"Serial read error: {e}")
                time.sleep(0.1)

    def _process_binary_buffer(self, buffer: bytes) -> bytes:
        """
        Process binary data in buffer, extract complete packets.

        Binary format uses sync/end bytes for framing:
        [SYNC_BYTE][569 bytes payload][END_BYTE]
        """
        while len(buffer) >= self.PACKET_SIZE_BINARY + 2:
            # Find sync byte
            sync_idx = buffer.find(bytes([self.SYNC_BYTE]))
            if sync_idx < 0:
                buffer = b''
                break

            # Discard data before sync
            if sync_idx > 0:
                buffer = buffer[sync_idx:]

            # Check if we have complete packet
            if len(buffer) < self.PACKET_SIZE_BINARY + 2:
                break

            # Check end byte
            if buffer[self.PACKET_SIZE_BINARY + 1] != self.END_BYTE:
                buffer = buffer[1:]  # Skip bad sync, try again
                continue

            # Extract packet
            packet_data = buffer[1:self.PACKET_SIZE_BINARY + 1]
            buffer = buffer[self.PACKET_SIZE_BINARY + 2:]

            try:
                packet = self._parse_binary_packet(packet_data)
                if not self._packet_queue.full():
                    self._packet_queue.put(packet)
            except Exception as e:
                print(f"Packet parse error: {e}")

        return buffer

    def _process_json_buffer(self, buffer: bytes) -> bytes:
        """
        Process JSON data in buffer, extract complete JSON objects.

        JSON format uses newline as delimiter:
        {"ts": ..., "eeg": [...], ...}\n
        """
        import json

        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)

            try:
                data = json.loads(line.decode('utf-8'))
                packet = self._parse_json_packet(data)
                if not self._packet_queue.full():
                    self._packet_queue.put(packet)
            except Exception as e:
                print(f"JSON parse error: {e}")

        return buffer

    def _parse_binary_packet(self, data: bytes) -> GoldPacket:
        """Parse 569-byte binary packet."""
        # Reuse BLE parser (same format)
        from .ble import BLEAdapter
        packet, _ = BLEAdapter.parse_binary_packet(data)
        return packet

    def _parse_json_packet(self, data: dict) -> GoldPacket:
        """Parse JSON packet into GoldPacket."""
        return GoldPacket(
            timestamp_ms=data.get('ts', int(time.time() * 1000)),
            device_id=data.get('device_id', 'serial-json'),
            packet_seq=data.get('seq', self._next_seq()),
            sample_rate_hz=data.get('sample_rate', 250.0),
            eeg=data.get('eeg', [[0.0] * 25 for _ in range(8)]),
            ecg=data.get('ecg', [[0.0] * 25 for _ in range(3)]),
            spo2_percent=data.get('spo2', 0.0),
            temp_celsius=data.get('temp', 0.0),
            accel_xyz_g=data.get('accel', [0.0, 0.0, 0.0]),
            quality_flags=PacketQualityFlags(data.get('quality', 0)),
        )

    @staticmethod
    def list_ports() -> List[dict]:
        """
        List available serial ports.

        Returns:
            List of dicts with 'port', 'description', 'hwid'
        """
        try:
            import serial.tools.list_ports
            ports = []
            for port in serial.tools.list_ports.comports():
                ports.append({
                    'port': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                })
            return ports
        except ImportError:
            raise ImportError(
                "pyserial library required. Install with: pip install pyserial"
            )
