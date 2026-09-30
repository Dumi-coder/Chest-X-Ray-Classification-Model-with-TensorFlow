```markdown
# NIH CXR8 Chest X-Ray Classification Tutorial

This repository contains a deep learning project utilizing transfer learning to classify abnormalities in chest X-rays using a subset of the NIH CXR8 dataset. The model adapts the pre-trained **InceptionV3** architecture in TensorFlow/Keras to perform binary classification.

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training & Fine-Tuning](#training--fine-tuning)
5. [Performance & Evaluation](#performance--evaluation)


---

## 📖 Project Overview
Identifying pathology in medical imaging is critical for diagnostic workflows. This project demonstrates end-to-end medical image classification:
- **Problem Formulation:** Binary classification of chest X-rays (e.g., detecting *Cardiomegaly* vs. *No Finding*).
- **Data Prep:** Creating stratified train/test datasets and applying robust data augmentation.
- **Modeling:** Transfer learning using a base InceptionV3 model frozen on ImageNet weights, followed by targeted layers fine-tuning.
- **Evaluation:** ROC analysis, AUC calculations, confusion metrics (Sensitivity/Specificity), and interactive threshold optimization.

## 📊 Dataset
The project uses a subset of the hospital-scale **NIH ChestX-ray8 dataset**. 
- **Positive label:** Cardiomegaly (or any user-defined findings such as Atelectasis)
- **Negative label:** No Finding
- **Train/Test Split:** 80% / 20%
- **Resolution:** Resized to 299x299 pixels to match InceptionV3's default input shape.

## 🧠 Model Architecture
To overcome limited medical imaging dataset sizes, we employ **Transfer Learning**:
- **Base Model:** `tf.keras.applications.InceptionV3` (weights pre-trained on ImageNet, excluding the top dense classification head).
- **Data Augmentation Layer:** Prevents overfitting with random rotations, translations, zoom, contrast, and brightness adjustments.
- **Global Average Pooling & Dropout:** A 50% dropout rate is applied to mitigate overfitting prior to the final prediction.
- **Output Head:** Single node dense layer with a `sigmoid` activation function for binary probabilities.

## ⚙️ Training & Fine-Tuning
The training is structured in two distinct phases:
1. **Phase 1: Feature Extraction (Frozen Base)**
   - Base model weights are frozen.
   - Trained with the `Adam` optimizer at default learning rate on top layers for 40 epochs with Early Stopping.
2. **Phase 2: Fine-Tuning (Unfrozen Upper Layers)**
   - Base model is set to trainable (`base_model.trainable = True`).
   - Layers before index 249 are frozen, leaving the top-level convolutional layers open for fine-tuning.
   - Re-compiled with a micro learning rate of `1e-5` to slowly adapt features to the diagnostic signatures.

## 📈 Performance & Evaluation
The trained model achieved high validation accuracy with robust generalization:
- **Receiver Operating Characteristic (ROC) curve:** Area Under the Curve (**AUC = 0.867**).
- **Custom Cut-off Strategy:** At a custom confidence cutoff of `0.79`, the system achieves:
  - **Sensitivity (Recall):** 0.47
  - **Specificity:** 0.97

```
