/**
 * @file temp_sim.c
 * @brief Body Temperature Simulator
 * @version 1.0.0
 *
 * @description
 * Simulates MLX90614 or DS18B20 temperature sensor.
 * Normal body temperature: 36.5-37.5°C (97.7-99.5°F)
 * Includes circadian rhythm variation.
 *
 * @note In production, interface via I2C (MLX90614) or 1-Wire (DS18B20)
 * @author Mohd Sarfaraz Faiyaz
 * @contributor Vaibhav Devram Chandgir
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define BASELINE_TEMP_C 36.8  // Core body temperature baseline

static unsigned int sample_count = 0;

/**
 * @brief Get simulated temperature reading
 * @return Temperature in Celsius
 */
float temp_sim_get_sample(void) {
    sample_count++;

    // Circadian rhythm: body temp varies ~0.5°C over 24h cycle
    // Simulate with much faster cycle for testing (period = 10000 samples ≈ 40s at 250 Hz)
    double circadian = 0.3 * sin(2.0 * PI * sample_count / 10000.0);

    // Small random noise (sensor precision ±0.1°C)
    double noise = 0.1 * ((double)rand() / RAND_MAX - 0.5);

    float temperature = (float)(BASELINE_TEMP_C + circadian + noise);

    return temperature;
}
