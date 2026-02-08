# edge-vision
Wafer Defect Detection — Edge AI (Edge Vision)

This repository contains the training, evaluation, and deployment pipeline for a lightweight CNN-based semiconductor wafer defect detection system optimized for edge inference and Phase-1 hackathon submission.

The workflow covers dataset loading, model training, evaluation, and conversion to ONNX format for downstream embedded deployment.

📁 Project Structure
.
├── WaferMap/
│   └── data/
│       ├── X.npy                # Wafer images (64×64, grayscale)
│       └── y.npy                # One-hot encoded labels (8 classes)
├── models/
│   └── wafer_model.h5           # Trained Keras model
├── wafer_model.onnx             # Exported ONNX model
├── training_script.py           # Model training script
├── evaluation_notebook.ipynb    # Evaluation & visualization
└── README.md

 Dataset

Format: NumPy arrays (.npy)

Image Shape: 64 × 64 × 1 (grayscale)

Number of Classes: 8

Clean

Defect_1 – Defect_6

Other

Split: Train / Validation / Test

 Model Architecture

Framework: TensorFlow / Keras

Architecture: Lightweight CNN optimized for edge devices

Layers

Conv2D + BatchNorm + ReLU

Conv2D + BatchNorm + ReLU

Global Average Pooling

Dense (128 units)

Output Dense (8 units, sigmoid)

Loss Function: Binary Cross-Entropy

Metric: Binary Accuracy

Model Size: ~60 KB

 Training

Run the training script:

python training_script.py

Outputs

Trained model saved to:

models/wafer_model.h5

Evaluation

Evaluation and visual analysis are provided in:

evaluation_notebook.ipynb

Includes

Classification report

Confusion matrix

Accuracy, Precision, Recall, F1-score

Binary (Clean vs Defect) performance analysis

Model Export (ONNX)

The trained Keras model is converted to ONNX format for edge deployment.

Required Versions
pip install tensorflow==2.13.0 keras==2.13.1 tf2onnx onnx

Conversion Script
import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("models/wafer_model.h5", compile=False)

tf2onnx.convert.from_keras(
    model,
    output_path="wafer_model.onnx"
)

✅ ONNX Validation
import onnx

onnx_model = onnx.load("wafer_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

 Inference (ONNX)

The exported ONNX model can be deployed using:

ONNX Runtime

Edge AI SDKs

Embedded accelerators and SoCs

 Notes

Designed for low-latency, edge-friendly inference

Easily extendable to larger industrial wafer datasets

Compatible with embedded and edge AI workflows

Intended as a Phase-1 baseline for further optimization (quantization, pruning, NXP eIQ flow)
