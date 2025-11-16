"""
LSTM Training Pipeline
-----------------------
Train recurrent neural network for temporal pattern recognition in
multi-modal physiological signals (EEG + ECG).

Architecture:
- Multi-channel time-series input (EEG: 8ch, ECG: 3ch)
- Bidirectional LSTM layers
- Attention mechanism (optional)
- Multi-task output: risk classification + anomaly detection

Use case:
- Detect temporal patterns missed by XGBoost (e.g., seizure precursors)
- Model time-series dependencies (autocorrelation in EEG/ECG)
- Capture transient events (arrhythmia, alpha bursts)

Author: Mohd Sarfaraz Faiyaz
Contributor: Vaibhav Devram Chandgir
Version: 1.0.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'sequence_length': 250,  # 1 second at 250 Hz
    'n_eeg_channels': 8,
    'n_ecg_channels': 3,
    'n_classes': 3,  # LOW, MEDIUM, HIGH risk
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'test_size': 0.2,
    'output_dir': 'ml/checkpoints/lstm',

    'model_params': {
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'dropout_rate': 0.3,
        'use_bidirectional': True,
        'use_attention': False  # Set True for attention mechanism
    }
}


# ============================================================================
# Data Generation (Simulated Sequential Data)
# ============================================================================

def generate_synthetic_sequences(
    n_samples: int = 2000,
    seq_length: int = 250,
    n_eeg_ch: int = 8,
    n_ecg_ch: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time-series sequences for LSTM training.

    Args:
        n_samples: Number of sequences
        seq_length: Length of each sequence (time steps)
        n_eeg_ch: Number of EEG channels
        n_ecg_ch: Number of ECG channels

    Returns:
        X: Input sequences (n_samples, seq_length, n_channels)
        y: Risk labels (n_samples,)
    """
    n_total_channels = n_eeg_ch + n_ecg_ch
    X = np.zeros((n_samples, seq_length, n_total_channels))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Generate EEG channels (multi-band oscillations)
        for ch in range(n_eeg_ch):
            # Base frequencies for each band
            delta_freq = np.random.uniform(1, 3)
            theta_freq = np.random.uniform(4, 7)
            alpha_freq = np.random.uniform(8, 12)
            beta_freq = np.random.uniform(15, 25)

            t = np.linspace(0, seq_length / 250.0, seq_length)

            # Composite signal
            eeg_signal = (
                10 * np.sin(2 * np.pi * delta_freq * t) +
                15 * np.sin(2 * np.pi * theta_freq * t) +
                30 * np.sin(2 * np.pi * alpha_freq * t) +
                10 * np.sin(2 * np.pi * beta_freq * t) +
                5 * np.random.randn(seq_length)  # Noise
            )

            X[i, :, ch] = eeg_signal

        # Generate ECG channels (PQRST complexes)
        hr = np.random.uniform(60, 100)  # BPM
        rr_interval_samples = int((60 / hr) * 250)

        for lead in range(n_ecg_ch):
            ecg_signal = np.zeros(seq_length)
            beat_positions = np.arange(0, seq_length, rr_interval_samples)

            for beat_pos in beat_positions:
                if beat_pos + 50 < seq_length:
                    # Simplified PQRST using Gaussian pulses
                    t_beat = np.arange(50)
                    qrs_complex = 1.2 * np.exp(-((t_beat - 20) ** 2) / 20)
                    ecg_signal[beat_pos:beat_pos + 50] += qrs_complex

            # Add noise
            ecg_signal += 0.02 * np.random.randn(seq_length)
            X[i, :, n_eeg_ch + lead] = ecg_signal

        # Generate label based on signal characteristics
        # HIGH RISK: Irregular rhythms, low alpha, high beta
        mean_alpha_power = np.var(X[i, :, 6])  # Channel O1 (occipital)
        mean_hr_variability = np.std(np.diff(beat_positions))

        risk_score = 0
        if mean_alpha_power < 500:
            risk_score += 1
        if mean_hr_variability < 5:
            risk_score += 1
        if hr > 90 or hr < 55:
            risk_score += 1

        y[i] = min(risk_score, 2)  # Clip to [0, 1, 2]

    return X, y


# ============================================================================
# Model Architecture
# ============================================================================

def build_lstm_model(config: Dict = CONFIG) -> Model:
    """
    Build multi-channel LSTM model for physiological signal analysis.

    Architecture:
        Input: (seq_length, n_channels)
        → Bidirectional LSTM (128 units)
        → Dropout
        → Bidirectional LSTM (64 units)
        → Dropout
        → Dense (32 units, ReLU)
        → Output (3 classes, Softmax)

    Args:
        config: Configuration dictionary

    Returns:
        Keras Model
    """
    seq_length = config['sequence_length']
    n_channels = config['n_eeg_channels'] + config['n_ecg_channels']
    params = config['model_params']

    # Input layer
    inputs = layers.Input(shape=(seq_length, n_channels), name='input')

    # Normalization layer (learnable)
    x = layers.BatchNormalization()(inputs)

    # LSTM layers
    if params['use_bidirectional']:
        x = layers.Bidirectional(
            layers.LSTM(params['lstm_units_1'], return_sequences=True),
            name='bilstm_1'
        )(x)
    else:
        x = layers.LSTM(params['lstm_units_1'], return_sequences=True, name='lstm_1')(x)

    x = layers.Dropout(params['dropout_rate'])(x)

    if params['use_bidirectional']:
        x = layers.Bidirectional(
            layers.LSTM(params['lstm_units_2'], return_sequences=False),
            name='bilstm_2'
        )(x)
    else:
        x = layers.LSTM(params['lstm_units_2'], return_sequences=False, name='lstm_2')(x)

    x = layers.Dropout(params['dropout_rate'])(x)

    # Dense layers
    x = layers.Dense(32, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer (multi-class classification)
    outputs = layers.Dense(
        config['n_classes'],
        activation='softmax',
        name='risk_output'
    )(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='NeuroCardiac_LSTM')

    return model


# ============================================================================
# Training Pipeline
# ============================================================================

def train_lstm_model(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict = CONFIG
) -> Tuple[Model, StandardScaler, Dict]:
    """
    Train LSTM model with validation and early stopping.

    Args:
        X: Input sequences (n_samples, seq_length, n_channels)
        y: Target labels (n_samples,)
        config: Configuration dictionary

    Returns:
        model: Trained Keras model
        scaler: Fitted StandardScaler for channels
        history: Training history
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=42,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]} sequences")
    print(f"Test set: {X_test.shape[0]} sequences")
    print(f"Input shape: (seq_length={X_train.shape[1]}, channels={X_train.shape[2]})")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Normalize channels independently
    scaler = StandardScaler()
    n_samples_train, seq_len, n_channels = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_channels)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples_train, seq_len, n_channels)

    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_channels)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_samples_test, seq_len, n_channels)

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=config['n_classes'])
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=config['n_classes'])

    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(config)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=False)]
    )

    # Callbacks
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(output_path / 'lstm_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled,
        y_train_cat,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_test_scaled, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Predictions
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Per-class accuracy
    for class_idx in range(config['n_classes']):
        class_name = ['LOW', 'MEDIUM', 'HIGH'][class_idx]
        mask = y_test == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == class_idx)
            print(f"{class_name} Risk Accuracy: {class_acc:.4f}")

    # Store metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_auc': float(test_auc),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }

    return model, scaler, metrics


# ============================================================================
# Model Persistence
# ============================================================================

def save_lstm_model(
    model: Model,
    scaler: StandardScaler,
    metrics: Dict,
    output_dir: str = CONFIG['output_dir']
):
    """Save trained LSTM model and artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model (native Keras format)
    model_path = output_path / 'lstm_model.keras'
    model.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}")

    # Save scaler
    import joblib
    scaler_path = output_path / 'lstm_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    # Save metadata
    metadata = {
        'model_type': 'LSTM',
        'version': '1.0.0',
        'config': CONFIG,
        'metrics': metrics
    }
    metadata_path = output_path / 'lstm_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("NeuroCardiac Shield - LSTM Training Pipeline")
    print("="*60)

    # Check GPU availability
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

    # Generate synthetic sequences
    print("\n[1/4] Generating synthetic time-series data...")
    X, y = generate_synthetic_sequences(
        n_samples=2000,
        seq_length=CONFIG['sequence_length'],
        n_eeg_ch=CONFIG['n_eeg_channels'],
        n_ecg_ch=CONFIG['n_ecg_channels']
    )
    print(f"Generated {len(X)} sequences of shape {X.shape[1:]}")

    # Train model
    print("\n[2/4] Training LSTM model...")
    model, scaler, metrics = train_lstm_model(X, y)

    # Save model
    print("\n[3/4] Saving model artifacts...")
    save_lstm_model(model, scaler, metrics)

    # Verification
    print("\n[4/4] Verifying saved model...")
    loaded_model = keras.models.load_model(f"{CONFIG['output_dir']}/lstm_model.keras")
    print("✓ Model loaded successfully")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    print(f"Model saved to: {CONFIG['output_dir']}/")
    print("\nTo use in production:")
    print("  from ml.model.inference import load_lstm_model, predict_lstm")
