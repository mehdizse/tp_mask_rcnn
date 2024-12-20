# Drone Detection with Mask R-CNN

This repository contains the implementation and training of a Mask R-CNN model for detecting drones in images. The model is built using Mask R-CNN implementation and trained on a custom drone dataset.

## 🎯 Overview
The goal of this project is to accurately detect drones in images using a pre-trained COCO model and further fine-tune it on a custom dataset. The model's performance is evaluated based on various metrics, and its results are analyzed through different lenses.

## 📊 Dataset
- **Training Images**: 252 images labeled for training.
- **Validation Images**: 50 images labeled for validation.
- **Annotation Tool**: labelme
- **Annotations**: Sample annotations with detailed labels for each object are shown below; however, the level of detail in these annotations may be quite complex.

![Annotated Training Image Exemple](images/drone_annotation_0.PNG)
![Annotated Training Image Exemple 2](images/drone_annotation_2.PNG)
![Annotated Training Image Exemple 3](images/drone_annotation_4.png)


## Dataset Structure

The dataset is organized into three main directories: [`train`](dataset/train), [`val`](dataset/val), and [`validation`](dataset/validation). Each folder serves a specific purpose for training, validation, and testing of the model:


### Directory Descriptions
- **[`train/`](dataset/train)**: Holds the training dataset with separate `images` and `annotations` directories.
- **[`val/`](dataset/val)**: Contains the validation dataset, organized with `images` and `annotations`.
- **[`validation/`](dataset/validation)**: Contains only test images in the `images` directory, used for final evaluation.

> **Note**: The `validation` folder includes only images, as it is typically used to assess model performance on unseen data without labels.


## Displaying an Image and its Associated Masks

Below are examples showing an image from the dataset and its associated object masks. Each mask highlights a detected object, making it possible to visualize how well the model can segment and locate each object within the image.

**Example 1**:  
![Example Image 1](images/mask0.png)  

**Example 2**:  
![Example Image 2](images/mask2.png)  

Each mask shows individual objects, making it easier to observe object locations and contours in the images.


## 🧠 Model Architecture
The Mask R-CNN architecture is used with a ResNet backbone and consists of:
1. Region Proposal Network (RPN)
2. Mask Head for segmentation

## 🚂 Training
- **Epochs**: 20 
- **Learning Rate**: 0.001 
- **Optimizer**: Adam
- **Batch Size**: 1 (due to GPU limitations)

## 🎛️ Hyperparameters
- **Detection Confidence Threshold**: 0.9
- **Steps per Epoch**: 100

## 📓 Jupyter Notebooks

This project includes two Jupyter notebooks for training and testing:

- **[train.ipynb](train.ipynb)**: This notebook contains the code and dataset for training the model. It includes preprocessing, model setup, and training steps and also the refine code.
- **[test.ipynb](test.ipynb)**: This notebook contains code for testing the trained model on real images. It allows you to evaluate the model's performance on new, unseen data.


## 📈 Results

## Training Logs

Below are screenshots of the training logs generated during the training process. These logs show the progression of the training and the losses calculated at each epoch:

1. **Training Log (Epoch 1-4)**  
   ![Training Log Epoch 1-4](images/train_result1.png)

2. **Training Log (Epoch 5-8)**  
   ![Training Log Epoch 5-8](images/train_result2.png)

3. **Training Log (Epoch 9-12)**  
   ![Training Log Epoch 9-12](images/train_result3.png)

4. **Training Log (Epoch 13-16)**  
   ![Training Log Epoch 13-16](images/train_result4.png)

4. **Training Log (Epoch 17-20)**  
   ![Training Log Epoch 17-20](images/train_result5.png)

Below is an image showing the files generated during the training process. These files include the model checkpoints saved at each epoch, which can be used for further evaluation or fine-tuning.
<br/>
<img src="images/fichiers_genere_annotation.PNG" alt="Training Logs" width="500"/>

- **Confusion Matrix**: 

The confusion matrix below represents the results of our drone detection model on the test dataset. As shown, all values in the matrix are zero, which indicates that the model did not make any correct or incorrect predictions for either the "background" or "drone" classes in this particular evaluation.

![Confusion Matrix](images/matrice_de_confusion.png)
- **Loss Graph**: ![Training vs Validation Loss](images/train_plot.png)
- **Test Images**: 

Below are examples of the model's predictions on test images. Each output includes the original image and the detected areas, highlighting where the model identified drones.

**Example Test Image 1**:  
![Test Image 1](images/result_0.png)

**Example Test Image 2**:  
![Test Image 2](images/result_2.png)

These examples demonstrate the model's capacity to detect drones within real-world images, although some segmentation results may be inaccurate or contain errors.


## 🛠️ Model Refinement

To improve the model’s performance, several key parameters were adjusted to optimize detection and segmentation of drones in the images. These refined settings were selected to make the model more sensitive to drone detection and enhance its efficiency given the available hardware constraints.

### 🔄 Refinement Parameters

The following parameters were adjusted:

- **DETECTION_MIN_CONFIDENCE**: Reduced to `0.5` to detect a broader range of objects, making the model more sensitive to potential drone objects in images.
- **DETECTION_NMS_THRESHOLD**: Set to `0.4` to increase the precision of the model’s bounding box outputs and reduce overlapping detections.
- **STEPS_PER_EPOCH**: Increased to `150` for a higher number of training iterations per epoch, which improves model learning.
- **VALIDATION_STEPS**: Set to `25` to ensure sufficient validation checks during each epoch,git  providing a balanced assessment of model performance.

These refinements were implemented to achieve a better trade-off between detection accuracy and computational efficiency, allowing the model to perform effectively on the target dataset and accurately segment drones in various image conditions.

### 📋 Refinement Results

Below are screenshots of the refinement logs generated during the refinement process. These logs show the progression of the training and the losses calculated at each epoch:

1. **Refinement Log (Epoch 21-26)**  
   ![Refinement Log Epoch 21-26](images/refine_result.PNG)

2. **Refinement Log (Epoch 27-30)**  
   ![Refinement Log Epoch 27-30](images/refine_result_2.PNG)

Below is an image showing the files generated during the refinement process. These files include the model checkpoints saved at each epoch, which can be used for further evaluation or fine-tuning.

<img src="images/fichiers_genere_retrain.PNG" alt="Training Logs" width="500"/>

- **Refinement Graph**: 

![Training vs Validation Loss](images/refinement_graphe.PNG)

## 📝 Observations and comments

### Initial Training (Epochs 1-20)
The model demonstrated progressive improvement in early stages:
- Training loss decreased steadily from 14.45 to approximately 1.68
- Validation loss was initially lower than training loss
- Validation loss started showing oscillations from epoch 8, ranging between 1.5 and 4.2

### Refinement Phase (Epochs 21-30)
During the refinement phase, we observed:
- Training loss decreased from 9.07 to 1.40
- Validation loss stabilized between 1.5 and 2.0
- MRCNN components (class, bbox, mask) showed zero values, indicating potential issues

### 📝 Observations and Detailed Analysis

#### Model Learning Patterns
The training process demonstrates a **steady decline** in both training and validation losses, reflecting **effective learning** and **model performance**. However, the **RPN losses** remain relatively elevated, particularly the **RPN bounding box loss**, which suggests that the model is struggling to accurately regress the bounding boxes for proposals. This could be due to **suboptimal anchor configurations** or an **insufficient learning rate** for the RPN.

#### Dataset Limitations
Our dataset lacks **diversity**, as we only used images of drones in **ideal positions**. This limitation impacts the model's ability to generalize effectively. Additionally, the oscillating validation loss from epoch 8 onwards suggests that the model might be struggling with consistent feature extraction.

## 🔧 Recommendations for Improvement

### Dataset Enhancement
1. Add diversity through:
   - **Negative samples** (images with no drones)
   - **Partially visible drones**
   - **Drones alongside similar objects**
   - Images with varying lighting conditions and backgrounds


### Annotation Optimization
1. Simplify overlapping instances
2. Create clearer boundaries
3. Standardize annotation guidelines



## 🚀 Getting Started

### Prerequisites
- Python 3.6 OR 3.7
- TensorFlow 1.x
- Keras 0.17.x
- scikit-image 2.1.x

### Installation
```sh
pip install -r requirements.txt
```


## 👥 Contributors

This drone detection project was developed as a collaborative effort:

- [**GHOMARI Mehdi**](mailto:mehdi.ghomari@etu.u-pec.fr)

- [**MEIGNAN Aito Florentin**](mailto:aito-florentin.meignan@etu.u-pec.fr)


## 📚 References

- Mask R-CNN Paper: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Original Mask R-CNN Implementation: [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.