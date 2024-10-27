# Drone Detection with Mask R-CNN

This repository contains the implementation and training of a Mask R-CNN model for detecting drones in images. The model is built using the Matterport Mask R-CNN implementation and trained on a custom drone dataset.

## Overview
The goal of this project is to accurately detect drones in images using a pre-trained COCO model and further fine-tune it on a custom dataset. The model's performance is evaluated based on various metrics, and its results are analyzed through different lenses.

## Dataset
- **Training Images**: 252 images labeled for training.
- **Validation Images**: 50 images labeled for validation.
- **Labelized**: with labelme 

## Model Architecture
The Mask R-CNN architecture is used with a ResNet backbone and consists of:
1. Region Proposal Network (RPN)
2. Mask Head for segmentation

## Training
- **Epochs**: 20
- **Learning Rate**: 0.001 (adjusted during training)
- **Optimizer**: Adam
- **Batch Size**: 1 (due to GPU limitations)

## Hyperparameters
- **Detection Confidence Threshold**: 0.9
- **Steps per Epoch**: 100

## Results
- **Training Logs**: Included in the `train_logs.txt` file.
- **Confusion Matrix**: [Link to confusion matrix]
- **Loss Graph**: [Link to loss graph]
- **Test Images**: [Link to test images]
- **Annotations and Images**: Samples from the dataset are included in the `data` directory.

## Observations and Comments
The model shows a decreasing trend in both training and validation losses, indicating effective learning. However, the RPN losses remain relatively high, suggesting a need for further tuning of anchor configurations or learning rate adjustments.

## Getting Started

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- Keras
- scikit-image

### Installation
```sh
pip install -r requirements.txt
