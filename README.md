# VoiceCommand

Voice command classifier for Lichee Pi Zero. Uses MFCC features with a Conv1D Keras model and exports to TFLite for on-device inference.

## Requirements

```
tensorflow
librosa
numpy
scikit-learn
```

## Usage

### Train

Place `.wav` files (16 kHz mono) in `dataset/` organised by label:

```
dataset/
  on/
    sample1.wav
    ...
  off/
    ...
```

Then run:

```bash
python train_command.py --data-dir ./dataset
```

Outputs `command_model.tflite` and `model_info.npz`.

### Inference

```bash
python infer.py test.wav
```

Predictions with confidence < 0.5 are reported as `"nothing"`.

## Model

### Input

16 kHz audio, 1.0 s clips → 13 MFCC coefficients → `(32, 13)` per sample.

### Architecture

| # | Layer | Output Shape | Params | Activation |
|---|-------|-------------|--------|------------|
| 1 | Input | `(None, 32, 13)` | 0 | — |
| 2 | Conv1D (32 filters, kernel=3) | `(None, 32, 32)` | 1,280 | ReLU |
| 3 | BatchNormalization | `(None, 32, 32)` | 128 | — |
| 4 | MaxPool1D (pool=2) | `(None, 16, 32)` | 0 | — |
| 5 | Conv1D (64 filters, kernel=3) | `(None, 16, 64)` | 6,208 | ReLU |
| 6 | BatchNormalization | `(None, 16, 64)` | 256 | — |
| 7 | GlobalAveragePooling1D | `(None, 64)` | 0 | — |
| 8 | Dropout (0.3) | `(None, 64)` | 0 | — |
| 9 | Dense (32 units) | `(None, 32)` | 2,080 | ReLU |
| 10 | Dropout (0.3) | `(None, 32)` | 0 | — |
| 11 | Dense (num_classes) | `(None, num_classes)` | 32×C + C | Softmax |

**Total params**: ~9,952 (trainable) for 2 classes.

### Export

Float32 TFLite (ready for post-training quantisation).
