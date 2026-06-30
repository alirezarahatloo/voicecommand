import sys
import numpy as np
import tensorflow as tf
from train_command import load_wav, extract_mfcc

tflite_path = "command_model.tflite"
info_path = "model_info.npz"

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.load(info_path, allow_pickle=True)
mean, std = data["mean"], data["std"]
classes = data["classes"]

wav_path = sys.argv[1]
y = load_wav(wav_path)
feat = extract_mfcc(y)
feat = (feat - mean) / std
feat = feat.astype(np.float32)[None, ...]

interpreter.set_tensor(input_details[0]["index"], feat)
interpreter.invoke()
probs = interpreter.get_tensor(output_details[0]["index"])[0]
pred_idx = np.argmax(probs)
label = classes[pred_idx] if probs[pred_idx] >= 0.5 else "nothing"
print(f"Prediction: {label}  (probs: {probs})")
