#!/usr/bin/env python3
"""
BLE Gateway - Firmware to Cloud Bridge
Reads binary packets from firmware output and sends to API.

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
"""

import struct
import time
import requests
import os
import sys

PACKET_SIZE = 569
BLE_OUTPUT_FILE = "/tmp/neurocardiac_ble_data.bin"
API_URL = "http://localhost:8000/api/v1/ingest"


def parse_packet(data):
    """Parse binary packet to JSON."""
    offset = 0

    timestamp_ms = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    packet_id = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    device_id = struct.unpack_from('<B', data, offset)[0]
    offset += 1
    status_flags = struct.unpack_from('<B', data, offset)[0]
    offset += 1

    eeg_data = []
    for _ in range(8):
        samples = []
        for _ in range(25):
            samples.append(struct.unpack_from('<h', data, offset)[0])
            offset += 2
        eeg_data.append(samples)

    ecg_data = []
    for _ in range(3):
        samples = []
        for _ in range(25):
            samples.append(struct.unpack_from('<h', data, offset)[0])
            offset += 2
        ecg_data.append(samples)

    spo2 = struct.unpack_from('<B', data, offset)[0]
    offset += 1
    temp = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    accel_x = struct.unpack_from('<h', data, offset)[0]
    offset += 2
    accel_y = struct.unpack_from('<h', data, offset)[0]
    offset += 2
    accel_z = struct.unpack_from('<h', data, offset)[0]
    offset += 2
    checksum = struct.unpack_from('<H', data, offset)[0]

    return {
        "timestamp_ms": timestamp_ms,
        "packet_id": packet_id,
        "device_id": device_id,
        "status_flags": status_flags,
        "eeg_data": eeg_data,
        "ecg_data": ecg_data,
        "spo2_percent": spo2,
        "temperature_celsius_x10": temp,
        "accel_x_mg": accel_x,
        "accel_y_mg": accel_y,
        "accel_z_mg": accel_z,
        "checksum": checksum
    }


def main():
    for _ in range(10):
        try:
            if requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        sys.exit(1)

    last_size = 0

    while True:
        try:
            if not os.path.exists(BLE_OUTPUT_FILE):
                time.sleep(0.5)
                continue

            current_size = os.path.getsize(BLE_OUTPUT_FILE)

            if current_size > last_size:
                with open(BLE_OUTPUT_FILE, 'rb') as f:
                    f.seek(last_size)
                    while True:
                        data = f.read(PACKET_SIZE)
                        if len(data) < PACKET_SIZE:
                            break
                        try:
                            packet = parse_packet(data)
                            requests.post(API_URL, json=packet, timeout=2)
                        except:
                            pass
                        last_size += PACKET_SIZE
                    last_size = f.tell()

            time.sleep(0.1)

        except KeyboardInterrupt:
            break
        except:
            time.sleep(1)


if __name__ == "__main__":
    main()
