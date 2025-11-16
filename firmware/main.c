/**
 * @file main.c
 * @brief NeuroCardiac Shield - Main Firmware Entry Point
 * @version 1.0.0
 * @date 2025-01-15
 *
 * @description
 * Main control loop for simulated multi-modal physiological data acquisition.
 * In production, this would interface with STM32 HAL for ADC, DMA, and BLE.
 * Current implementation generates realistic synthetic signals for development.
 *
 * @compliance IEC 62304 Class B - Medical Device Software
 * @author Mohd Sarfaraz Faiyaz
 * @contributor Vaibhav Devram Chandgir
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

// Forward declarations from modular components
extern void eeg_sim_init(void);
extern void eeg_sim_get_sample(float *channels, int num_channels);

extern void ecg_sim_init(void);
extern float ecg_sim_get_sample(void);

extern void spo2_sim_init(void);
extern uint8_t spo2_sim_get_sample(void);

extern float temp_sim_get_sample(void);
extern void accel_sim_get_sample(float *x, float *y, float *z);

extern void ble_stub_init(void);
extern void ble_stub_transmit(const uint8_t *data, size_t len);

// Configuration constants
#define EEG_NUM_CHANNELS    8     // Standard 10-20 system subset
#define ECG_NUM_LEADS       3     // Lead I, II, III
#define SAMPLE_RATE_HZ      250   // 250 Hz sampling (Nyquist for 0-125 Hz signals)
#define PACKET_INTERVAL_MS  100   // Send packets every 100ms (10 Hz update rate)
#define SAMPLES_PER_PACKET  25    // 250 Hz / 10 packets per sec = 25 samples

/**
 * @brief Unified physiological data packet structure
 * @note Packed structure for efficient BLE transmission
 * Total size: ~500 bytes per packet (optimize for MTU)
 */
typedef struct __attribute__((packed)) {
    uint32_t timestamp_ms;           // System timestamp
    uint16_t packet_id;              // Sequential packet counter
    uint8_t  device_id;              // Unique device identifier
    uint8_t  status_flags;           // Bit flags: [7:4] reserved, [3] accel, [2] temp, [1] SpO2, [0] valid

    // EEG data: 8 channels × 25 samples × 2 bytes (int16) = 400 bytes
    int16_t  eeg_data[EEG_NUM_CHANNELS][SAMPLES_PER_PACKET];

    // ECG data: 3 leads × 25 samples × 2 bytes = 150 bytes
    int16_t  ecg_data[ECG_NUM_LEADS][SAMPLES_PER_PACKET];

    // Ancillary vitals (sampled at lower rate, averaged)
    uint8_t  spo2_percent;           // SpO2: 0-100%
    int16_t  temperature_celsius_x10;// Temperature × 10 (e.g., 367 = 36.7°C)
    int16_t  accel_x_mg;             // Accelerometer X (millig)
    int16_t  accel_y_mg;             // Accelerometer Y
    int16_t  accel_z_mg;             // Accelerometer Z

    uint16_t checksum;               // CRC16 for data integrity
} PhysiologicalDataPacket;

/**
 * @brief Calculate CRC16-CCITT checksum
 */
static uint16_t calculate_crc16(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

/**
 * @brief Initialize all sensor modules and communication
 */
static void system_init(void) {
    printf("[FIRMWARE] NeuroCardiac Shield - Initializing...\n");

    // TODO: In production, initialize STM32 HAL, clocks, GPIOs
    // HAL_Init();
    // SystemClock_Config();
    // MX_GPIO_Init();
    // MX_ADC1_Init();
    // MX_DMA_Init();

    eeg_sim_init();
    ecg_sim_init();
    spo2_sim_init();
    ble_stub_init();

    printf("[FIRMWARE] All subsystems initialized.\n");
}

/**
 * @brief Main acquisition and transmission loop
 */
int main(void) {
    system_init();

    PhysiologicalDataPacket packet;
    uint16_t packet_counter = 0;
    uint32_t timestamp_ms = 0;

    float eeg_channels[EEG_NUM_CHANNELS];
    float accel_x, accel_y, accel_z;

    printf("[FIRMWARE] Entering main acquisition loop (Fs=%d Hz)...\n", SAMPLE_RATE_HZ);

    while (1) {
        // Prepare packet header
        memset(&packet, 0, sizeof(packet));
        packet.timestamp_ms = timestamp_ms;
        packet.packet_id = packet_counter++;
        packet.device_id = 0x01;  // TODO: Read from EEPROM/flash
        packet.status_flags = 0x0F;  // All sensors valid

        // Acquire signal samples for this packet window
        for (int sample_idx = 0; sample_idx < SAMPLES_PER_PACKET; sample_idx++) {
            // EEG: 8-channel acquisition
            eeg_sim_get_sample(eeg_channels, EEG_NUM_CHANNELS);
            for (int ch = 0; ch < EEG_NUM_CHANNELS; ch++) {
                // Convert to int16 (12-bit ADC emulation: ±200 µV typical EEG range)
                // Scaling: ±200 µV → ±2000 (int16 with gain of 10)
                packet.eeg_data[ch][sample_idx] = (int16_t)(eeg_channels[ch] * 10.0f);
            }

            // ECG: 3-lead acquisition
            float ecg_sample = ecg_sim_get_sample();
            // Convert to int16 (12-bit ADC: ±3 mV typical ECG range)
            packet.ecg_data[0][sample_idx] = (int16_t)(ecg_sample * 1000.0f);  // Lead I
            packet.ecg_data[1][sample_idx] = (int16_t)(ecg_sample * 1.1f * 1000.0f);  // Lead II (slightly different amplitude)
            packet.ecg_data[2][sample_idx] = (int16_t)(ecg_sample * 0.9f * 1000.0f);  // Lead III

            // Simulate ADC timing (1/250 Hz = 4 ms per sample)
            usleep(4000);  // TODO: Replace with hardware timer interrupt
        }

        // Acquire ancillary vitals (lower sampling rate, averaged over packet window)
        packet.spo2_percent = spo2_sim_get_sample();
        float temp_c = temp_sim_get_sample();
        packet.temperature_celsius_x10 = (int16_t)(temp_c * 10.0f);

        accel_sim_get_sample(&accel_x, &accel_y, &accel_z);
        packet.accel_x_mg = (int16_t)(accel_x * 1000.0f);
        packet.accel_y_mg = (int16_t)(accel_y * 1000.0f);
        packet.accel_z_mg = (int16_t)(accel_z * 1000.0f);

        // Calculate checksum
        packet.checksum = calculate_crc16((uint8_t*)&packet, sizeof(packet) - sizeof(uint16_t));

        // Transmit via BLE
        ble_stub_transmit((uint8_t*)&packet, sizeof(packet));

        timestamp_ms += PACKET_INTERVAL_MS;

        // Log telemetry (production: use UART debug or remove)
        if (packet_counter % 10 == 0) {
            printf("[FIRMWARE] Packet #%d | SpO2: %d%% | Temp: %.1f°C | ECG: %d mV\n",
                   packet_counter, packet.spo2_percent, temp_c, packet.ecg_data[0][0]);
        }
    }

    return 0;
}
