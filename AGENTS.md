# AGENTS.md — opencode

## Project

Train a voice command classifier for Lichee Pi Zero. Single-file Keras model with MFCC features exported to TFLite.

## Layout

- `train_command.py` — entrypoint, model definition, training loop, TFLite export, and an inference helper (`predict()`)
- `dataset/{label}/*.wav` — speech command wav files (16 kHz mono), e.g. `dataset/on/` and `dataset/off/`

## Commands

```bash
python train_command.py --data-dir ./dataset
```

Outputs `command_model.tflite` and `model_info.npz` (normalisation stats + label encoder). Default `--model-output` and `--test-split` 0.2.

## Key details

- Dependencies: `tensorflow`, `librosa`, `numpy`, `scikit-learn`
- Audio preprocessing: 16 kHz, 1.0 s clip, 13 MFCC coefficients → fixed `(32, 13)` per sample
- At inference, predictions with max confidence < 0.5 are mapped to `"nothing"` (see `predict()` at `train_command.py:161`)
- No tests, no lint/format config, no CI — run the script and inspect accuracy output
