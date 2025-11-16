/**
 * @file accel_sim.c
 * @brief 3-Axis Accelerometer Simulator
 * @version 1.0.0
 *
 * @description
 * Simulates ADXL345 or MPU6050 3-axis accelerometer for motion detection.
 * - Stationary: ~(0, 0, 1g) in standard orientation
 * - Used for: activity detection, artifact rejection, fall detection
 *
 * @note In production, interface via I2C or SPI
 * @author Mohd Sarfaraz Faiyaz
 * @contributor Vaibhav Devram Chandgir
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846
#define GRAVITY_G 1.0  // 1g = 9.81 m/s²

static unsigned int motion_phase = 0;

/**
 * @brief Get simulated 3-axis accelerometer reading
 * @param x Output: X-axis acceleration (g units)
 * @param y Output: Y-axis acceleration (g units)
 * @param z Output: Z-axis acceleration (g units)
 */
void accel_sim_get_sample(float *x, float *y, float *z) {
    motion_phase++;

    // Simulate subtle head movements (nod, shake)
    // Most of the time stationary with occasional motion bursts
    double movement = 0.0;
    if ((motion_phase % 500) < 50) {  // Brief movement every 2 seconds
        movement = 0.3 * sin(2.0 * PI * (motion_phase % 50) / 50.0);
    }

    // Base orientation: device upright, Z-axis pointing up
    *x = (float)(0.02 * ((double)rand() / RAND_MAX - 0.5) + movement);  // Noise + motion
    *y = (float)(0.02 * ((double)rand() / RAND_MAX - 0.5));
    *z = (float)(GRAVITY_G + 0.02 * ((double)rand() / RAND_MAX - 0.5));  // 1g ± noise
}
