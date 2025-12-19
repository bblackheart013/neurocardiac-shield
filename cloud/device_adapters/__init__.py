"""
NeuroCardiac Shield — Device Adapters
======================================
Pluggable device adapter layer for connecting real or simulated sensors.

This package provides a unified interface for:
- Simulated devices (for development and testing)
- BLE-connected wearables (for production use)
- Serial/UART microcontrollers (for prototyping)

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
NYU Tandon School of Engineering | ECE-GY 9953 | Fall 2025
"""

from .base import DeviceAdapter, AdapterMetadata
from .gold_schema import GoldPacket, PacketQualityFlags
from .simulated import SimulatedAdapter
from .ble import BLEAdapter
from .serial import SerialAdapter

__all__ = [
    'DeviceAdapter',
    'AdapterMetadata',
    'GoldPacket',
    'PacketQualityFlags',
    'SimulatedAdapter',
    'BLEAdapter',
    'SerialAdapter',
]

# Adapter registry for dynamic instantiation
ADAPTER_REGISTRY = {
    'simulated': SimulatedAdapter,
    'ble': BLEAdapter,
    'serial': SerialAdapter,
}


def get_adapter(adapter_type: str, **kwargs) -> DeviceAdapter:
    """
    Factory function to get an adapter instance by type.

    Args:
        adapter_type: One of 'simulated', 'ble', 'serial'
        **kwargs: Adapter-specific configuration

    Returns:
        Configured DeviceAdapter instance

    Example:
        >>> adapter = get_adapter('simulated', seed=42)
        >>> adapter.connect()
        >>> packet = adapter.read_packet()
    """
    if adapter_type not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[adapter_type](**kwargs)
