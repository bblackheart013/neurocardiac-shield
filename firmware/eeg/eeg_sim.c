/**
 * @file eeg_sim.c
 * @brief Realistic EEG Signal Simulator
 * @version 1.0.0
 *
 * @description
 * Generates synthetic 8-channel EEG signals with physiologically realistic
 * frequency components (delta, theta, alpha, beta, gamma) and spatial correlations.
 * Simulates frontal, temporal, parietal, and occipital electrode placements.
 *
 * Frequency bands:
 * - Delta (0.5-4 Hz): Deep sleep, unconscious processes
 * - Theta (4-8 Hz): Drowsiness, meditation, memory
 * - Alpha (8-13 Hz): Relaxed wakefulness, closed eyes
 * - Beta (13-30 Hz): Active thinking, concentration
 * - Gamma (30-100 Hz): Cognitive processing, perception
 *
 * @note In production, replace with ADS1299 ADC interface
 * @author Mohd Sarfaraz Faiyaz
 * @contributor Vaibhav Devram Chandgir
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define SAMPLE_RATE 250.0  // Hz
#define NUM_CHANNELS 8

// Simulated electrode positions (10-20 system subset)
static const char* ELECTRODE_NAMES[NUM_CHANNELS] = {
    "Fp1", "Fp2",  // Frontal pole
    "C3", "C4",    // Central
    "T3", "T4",    // Temporal
    "O1", "O2"     // Occipital
};

// Internal state for continuous signal generation
typedef struct {
    double phase_delta;
    double phase_theta;
    double phase_alpha;
    double phase_beta;
    double phase_gamma;
    double noise_accumulator;
    double spatial_weight;  // Channel-specific weighting
} ChannelState;

static ChannelState channel_states[NUM_CHANNELS];
static unsigned long sample_counter = 0;

/**
 * @brief Generate pink noise (1/f spectrum) for realistic EEG background
 */
static double pink_noise(double *accumulator) {
    double white = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    *accumulator = 0.99 * (*accumulator) + 0.01 * white;
    return *accumulator;
}

/**
 * @brief Initialize EEG simulator with randomized phases
 */
void eeg_sim_init(void) {
    srand((unsigned int)time(NULL));

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        channel_states[ch].phase_delta = (double)rand() / RAND_MAX * 2.0 * PI;
        channel_states[ch].phase_theta = (double)rand() / RAND_MAX * 2.0 * PI;
        channel_states[ch].phase_alpha = (double)rand() / RAND_MAX * 2.0 * PI;
        channel_states[ch].phase_beta = (double)rand() / RAND_MAX * 2.0 * PI;
        channel_states[ch].phase_gamma = (double)rand() / RAND_MAX * 2.0 * PI;
        channel_states[ch].noise_accumulator = 0.0;

        // Spatial weighting: occipital channels have stronger alpha
        if (ch >= 6) {  // O1, O2
            channel_states[ch].spatial_weight = 1.5;
        } else if (ch >= 4) {  // T3, T4
            channel_states[ch].spatial_weight = 1.0;
        } else {  // Frontal/central
            channel_states[ch].spatial_weight = 0.8;
        }
    }

    printf("[EEG_SIM] Initialized 8-channel EEG simulator (Fs=%.0f Hz)\n", SAMPLE_RATE);
}

/**
 * @brief Generate one sample for all EEG channels
 * @param channels Output array (size NUM_CHANNELS), units: microvolts (µV)
 * @param num_channels Number of channels to generate
 */
void eeg_sim_get_sample(float *channels, int num_channels) {
    if (num_channels > NUM_CHANNELS) {
        num_channels = NUM_CHANNELS;
    }

    double time_s = (double)sample_counter / SAMPLE_RATE;
    sample_counter++;

    for (int ch = 0; ch < num_channels; ch++) {
        ChannelState *state = &channel_states[ch];

        // Frequency band synthesis with physiologically realistic amplitudes
        double delta = 20.0 * sin(state->phase_delta);  // 20 µV
        double theta = 15.0 * sin(state->phase_theta);  // 15 µV
        double alpha = 30.0 * sin(state->phase_alpha) * state->spatial_weight;  // 30-45 µV (dominant in occipital)
        double beta = 10.0 * sin(state->phase_beta);    // 10 µV
        double gamma = 3.0 * sin(state->phase_gamma);   // 3 µV

        // Pink noise background (physiological baseline noise)
        double noise = 5.0 * pink_noise(&state->noise_accumulator);

        // Composite signal
        double signal = delta + theta + alpha + beta + gamma + noise;

        // Add occasional artifacts (eye blinks, muscle noise)
        if ((sample_counter % 1500) < 50 && ch < 2) {  // Eye blink every 6s on frontal channels
            signal += 100.0 * exp(-((sample_counter % 1500) / 10.0));
        }

        channels[ch] = (float)signal;

        // Update phases for next sample
        state->phase_delta += 2.0 * PI * 2.0 / SAMPLE_RATE;   // 2 Hz delta
        state->phase_theta += 2.0 * PI * 6.0 / SAMPLE_RATE;   // 6 Hz theta
        state->phase_alpha += 2.0 * PI * 10.0 / SAMPLE_RATE;  // 10 Hz alpha (classic)
        state->phase_beta += 2.0 * PI * 20.0 / SAMPLE_RATE;   // 20 Hz beta
        state->phase_gamma += 2.0 * PI * 40.0 / SAMPLE_RATE;  // 40 Hz gamma

        // Wrap phases to [0, 2π]
        if (state->phase_delta > 2.0 * PI) state->phase_delta -= 2.0 * PI;
        if (state->phase_theta > 2.0 * PI) state->phase_theta -= 2.0 * PI;
        if (state->phase_alpha > 2.0 * PI) state->phase_alpha -= 2.0 * PI;
        if (state->phase_beta > 2.0 * PI) state->phase_beta -= 2.0 * PI;
        if (state->phase_gamma > 2.0 * PI) state->phase_gamma -= 2.0 * PI;
    }
}
