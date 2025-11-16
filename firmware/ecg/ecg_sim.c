/**
 * @file ecg_sim.c
 * @brief Realistic ECG Signal Simulator with PQRST Complex
 * @version 1.0.0
 *
 * @description
 * Generates synthetic single-lead ECG with physiologically accurate PQRST morphology:
 * - P wave: Atrial depolarization (~80 ms duration, 0.15 mV amplitude)
 * - QRS complex: Ventricular depolarization (~80 ms duration, 1.0-1.5 mV amplitude)
 * - T wave: Ventricular repolarization (~160 ms duration, 0.3 mV amplitude)
 * - Heart rate: ~70 BPM with realistic variability (HRV)
 *
 * Uses Gaussian approximations for PQRST waves.
 *
 * @note In production, interface with AD8232 ECG frontend or ADS1293
 * @author Mohd Sarfaraz Faiyaz
 * @contributor Vaibhav Devram Chandgir
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define SAMPLE_RATE 250.0  // Hz
#define BASE_HEART_RATE 70.0  // BPM (beats per minute)
#define RR_INTERVAL_MS (60000.0 / BASE_HEART_RATE)  // ~857 ms

// Internal state
static unsigned long sample_counter = 0;
static double next_beat_time_ms = 0.0;
static double heart_rate_variability = 0.0;  // RMSSD component

/**
 * @brief Gaussian pulse (used to construct PQRST waves)
 */
static double gaussian_pulse(double t, double center, double width, double amplitude) {
    double exponent = -((t - center) * (t - center)) / (2.0 * width * width);
    return amplitude * exp(exponent);
}

/**
 * @brief Initialize ECG simulator
 */
void ecg_sim_init(void) {
    srand((unsigned int)time(NULL) + 12345);  // Different seed from EEG
    next_beat_time_ms = RR_INTERVAL_MS;
    printf("[ECG_SIM] Initialized ECG simulator (HR=%.0f BPM, Fs=%.0f Hz)\n",
           BASE_HEART_RATE, SAMPLE_RATE);
}

/**
 * @brief Generate one ECG sample
 * @return ECG voltage in millivolts (mV)
 */
float ecg_sim_get_sample(void) {
    double time_ms = (double)sample_counter / SAMPLE_RATE * 1000.0;
    sample_counter++;

    // Check if we need to generate a new heartbeat
    if (time_ms >= next_beat_time_ms) {
        // Simulate heart rate variability (HRV): RMSSD ~ 30-50 ms for healthy adult
        double hrv_delta = ((double)rand() / RAND_MAX - 0.5) * 80.0;  // ±40 ms variation
        heart_rate_variability = hrv_delta;

        double rr_interval = RR_INTERVAL_MS + hrv_delta;
        next_beat_time_ms += rr_interval;
    }

    // Time relative to current beat cycle
    double beat_time = fmod(time_ms, RR_INTERVAL_MS);

    // Construct PQRST complex using Gaussian pulses
    // Times are relative to start of RR interval (ms)
    // Reference: MIT-BIH database morphology

    double p_wave = gaussian_pulse(beat_time, 100.0, 25.0, 0.15);   // P wave at ~100 ms
    double q_wave = gaussian_pulse(beat_time, 180.0, 8.0, -0.1);    // Q wave (negative deflection)
    double r_wave = gaussian_pulse(beat_time, 200.0, 12.0, 1.3);    // R wave (main peak)
    double s_wave = gaussian_pulse(beat_time, 220.0, 10.0, -0.2);   // S wave
    double t_wave = gaussian_pulse(beat_time, 380.0, 50.0, 0.25);   // T wave at ~380 ms

    // Baseline wander (simulate respiration artifact, ~0.2 Hz)
    double baseline = 0.05 * sin(2.0 * PI * 0.2 * time_ms / 1000.0);

    // High-frequency noise (muscle artifacts, powerline interference)
    double noise = 0.02 * ((double)rand() / RAND_MAX - 0.5);

    // Composite ECG signal
    double ecg = p_wave + q_wave + r_wave + s_wave + t_wave + baseline + noise;

    return (float)ecg;
}
