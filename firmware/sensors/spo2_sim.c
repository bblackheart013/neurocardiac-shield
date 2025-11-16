/**
 * @file spo2_sim.c
 * @brief SpO2 (Blood Oxygen Saturation) Simulator
 * @version 1.0.0
 *
 * @description
 * Simulates MAX30102 pulse oximeter readings.
 * Normal SpO2 range: 95-100% for healthy individuals.
 * Includes realistic variations due to motion, perfusion index changes.
 *
 * @note In production, interface with MAX30102 via I2C
 * @author Mohd Sarfaraz Faiyaz
 * @contributor Vaibhav Devram Chandgir
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BASELINE_SPO2 97  // Healthy baseline: 97%
#define SPO2_VARIATION 2   // ±2% typical variation

static unsigned int call_count = 0;

/**
 * @brief Initialize SpO2 simulator
 */
void spo2_sim_init(void) {
    srand((unsigned int)time(NULL) + 54321);
    printf("[SPO2_SIM] Initialized pulse oximeter simulator\n");
}

/**
 * @brief Get simulated SpO2 reading
 * @return SpO2 percentage (0-100%)
 */
unsigned char spo2_sim_get_sample(void) {
    call_count++;

    // Slow drift to simulate physiological changes
    int drift = (call_count / 100) % 3 - 1;  // -1, 0, +1% slow drift

    // Random variation
    int variation = (rand() % (SPO2_VARIATION * 2 + 1)) - SPO2_VARIATION;

    int spo2 = BASELINE_SPO2 + drift + variation;

    // Clamp to valid range
    if (spo2 < 85) spo2 = 85;  // Below 85% would be concerning
    if (spo2 > 100) spo2 = 100;

    return (unsigned char)spo2;
}
