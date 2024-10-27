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
- **Training Logs**: Included in the [train_logs.txt](train_logs.txt) file.
- **Confusion Matrix**: [Link to confusion matrix]
- **Loss Graph**: ![Training vs Validation Loss](images/plot.PNG)
- **Test Images**: [Link to test images]
- **Annotations and Images**: 
- Samples from the dataset are included in the `dataset` directory:
- Training images and annotations are in `dataset/train`
- Validation images and annotations are in `dataset/val`

## Observations and Comments
The training process demonstrates a steady decline in both training and validation losses, reflecting effective learning and model performance. However, the RPN losses remain relatively elevated, indicating potential areas for improvement in anchor configurations or learning rate adjustments to enhance overall model efficacy.

## Getting Started

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- Keras
- scikit-image

### Installation
```sh
pip install -r requirements.txt
