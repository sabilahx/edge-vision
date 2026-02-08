# edge-vision
# Wafer Defect Detection — Edge AI (Edge Vision)

Lightweight CNN pipeline for semiconductor wafer defect detection, designed for edge deployment and Phase-1 hackathon submission.

Includes training, evaluation, and ONNX export for embedded workflows.

# Project Structure
WaferMap/
 └── data/
     ├── X.npy
     └── y.npy

models/
 └── wafer_model.h5

wafer_model.onnx
training_script.py
evaluation_notebook.ipynb
README.md

***Dataset***

Format: NumPy (.npy)

Image size: 64×64×1 (grayscale)

Classes (8):

Clean

Defect_1 – Defect_6

Other

Split: Train / Validation / Test

Model Architecture

Framework: TensorFlow / Keras

Architecture: Lightweight CNN

Layers:

Conv2D + BatchNorm + ReLU

Conv2D + BatchNorm + ReLU

Global Average Pooling

Dense (128)

Output Dense (8, sigmoid)

Loss: Binary Cross-Entropy
Metric: Binary Accuracy
Model size: ~60 KB

***Training***

Run:

python training_script.py


Output:

models/wafer_model.h5

Evaluation

Open:

evaluation_notebook.ipynb


Includes:

Confusion matrix

Accuracy, Precision, Recall, F1

Clean vs Defect analysis

ONNX Export

Install:

pip install tensorflow==2.13.0 keras==2.13.1 tf2onnx onnx


Convert:

import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("models/wafer_model.h5", compile=False)
tf2onnx.convert.from_keras(model, output_path="wafer_model.onnx")

ONNX Validation
import onnx
onnx_model = onnx.load("wafer_model.onnx")
onnx.checker.check_model(onnx_model)

***Notes***

Edge-friendly design

Phase-1 baseline

ONNX ready

Extendable to larger datasets
