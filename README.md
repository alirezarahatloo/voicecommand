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

- Audio: 16 kHz, 1.0 s clips, 13 MFCC coefficients → `(32, 13)` per sample
- Architecture: Conv1D → BatchNorm → Pool → Conv1D → GAP → Dense → Softmax
- Export: float32 TFLite (ready for post-training quantisation)
