# Multilayer Perceptron (MLP) from Scratch

A Python implementation of a Multilayer Perceptron neural network for binary classification, built from scratch without high-level ML libraries.

![MLP Architecture Diagram](mlp_ChatGPT Image Jun 1, 2025, 08_56_21 PM.png)  
*Figure: Architecture of the implemented MLP with two hidden layers.*

---

## 🔍 Overview
This project implements:
- **Feedforward propagation**
- **Backpropagation with gradient descent**
- **Modular layer design** (customizable hidden layers)
- **Softmax activation** for probabilistic outputs
- **Binary cross-entropy loss** evaluation
- Learning curve visualization

Trained on the **Wisconsin Breast Cancer Dataset** to classify tumors as benign/malignant.

---

## 🛠️ Dependencies
- make install	Install dependencies
- make split	Preprocess and split dataset
- make train	Train model with default args
- make predict	Run predictions on test data
- make clean	Delete generated files (models, plots)
- make all	Run full pipeline: install → data → train
