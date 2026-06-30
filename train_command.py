"""
Train a voice command classifier for Lichee Pi Zero.

Usage:
  data/
    turn_on/
      sample1.wav
      ...
    turn_off/
      ...

  python train_command.py --data-dir ./data

At inference, if confidence < 0.5 the output is "nothing".
"""

import argparse
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configuration – tweak these for your data
# ---------------------------------------------------------------------------
SR = 16000               # sample rate (Hz)
DURATION = 1.0           # seconds per clip (pad / truncate to this)
N_MFCC = 13              # MFCC coefficients
MAX_PAD = 32             # time steps after MFCC (fixed width for the model)
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


def load_wav(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    # pad or truncate to exactly duration seconds
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC, max_pad=MAX_PAD):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)   # (n_mfcc, T)
    # pad / truncate time axis to max_pad
    if mfcc.shape[1] < max_pad:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_pad]
    return mfcc.T   # (max_pad, n_mfcc)  — each row is one time step


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    X, y = [], []
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for wav_path in label_dir.glob("*.wav"):
            y_val = extract_mfcc(load_wav(str(wav_path)))
            X.append(y_val)
            y.append(label)
    X = np.array(X, dtype=np.float32)
    return X, np.array(y)


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Path to data/ with subdirs per class")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--model-output", default="command_model.tflite")
    args = parser.parse_args()

    print("Loading dataset …")
    X, y = load_dataset(args.data_dir)
    print(f"  {len(X)} samples, shape {X.shape}, classes {np.unique(y)}")

    le = LabelEncoder()
    y_int = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=args.test_split, stratify=y_int, random_state=42
    )

    # Normalise per-feature across the training set
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = build_model(X.shape[1:], num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\nTraining …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            ),
        ],
    )

    print("\nEvaluating …")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy: {acc:.3f}")

    # ------------------------------------------------------------------
    # TFLite export (float32 — use post-training quantisation for
    # even smaller size / faster inference on the Pi Zero)
    # ------------------------------------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path = args.model_output
    Path(out_path).write_bytes(tflite_model)
    print(f"\nTFLite model written to {out_path} ({len(tflite_model)} bytes)")

    # Save the normalisation params + label encoder for inference
    np.savez("model_info.npz",
             mean=mean.squeeze(), std=std.squeeze(),
             classes=le.classes_)

    print("\nDone. Copy command_model.tflite and model_info.npz to your device.")


# -----------------------------------------------------------------------
# Stand-alone inference helper (run on Lichee Pi Zero with TFLite)
# -----------------------------------------------------------------------
def predict(wav_path, interpreter, mean, std, classes, threshold=0.5):
    """Run inference. Returns ("nothing", probs) if max confidence < threshold."""
    y = load_wav(wav_path)
    feat = extract_mfcc(y)
    feat = (feat - mean) / std
    feat = feat.astype(np.float32)[None, ...]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], feat)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    pred_idx = np.argmax(probs)
    if probs[pred_idx] < threshold:
        return "nothing", probs
    return classes[pred_idx], probs


if __name__ == "__main__":
    main()
