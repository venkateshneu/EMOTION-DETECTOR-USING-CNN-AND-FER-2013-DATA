![Alt Text](https://github.com/venkateshneu/EMOTION-DETECTOR-USING-VGGNET16-AND-RESNET50-FER-2013-DATA/blob/main/Monday_September23_2024at1_58_00PM_default_8cf3a081-ezgif.com-video-to-gif-converter.gif)


# FER2013 Emotion Detection

## Overview

This project focuses on detecting human emotions from facial images using deep learning models. The FER2013 dataset is used for training and testing various models, including Custom CNN, VGG16, and ResNet50 with Transfer Learning. The goal is to classify facial expressions into one of the seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral.

## Dataset Description

The dataset used in this project is the FER2013 dataset, which is publicly available on Kaggle. It contains 48x48 pixel grayscale images of human faces, each labeled with one of seven emotions:

- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

The dataset is divided into training and testing sets, with the training set containing around 28,000 images and the testing set containing around 7,000 images.

## Objective

The primary objective of this project is to build and evaluate deep learning models capable of accurately classifying emotions from facial images. The project also explores the effectiveness of Transfer Learning by using pre-trained models such as VGG16 and ResNet50 on the FER2013 dataset.

## Models Implemented

### 1. Custom CNN
   - A Convolutional Neural Network (CNN) built from scratch.
   - Input: 48x48 grayscale images.
   - Architecture: Multiple convolutional layers followed by max-pooling layers and fully connected layers.
   - Optimizer: Adam.
   - Loss: Categorical Cross-Entropy.

### 2. VGG16 with Transfer Learning
   - Pre-trained VGG16 model used for feature extraction.
   - Input: 224x224 RGB images (resized from 48x48 grayscale).
   - Transfer learning: Fine-tuned on the FER2013 dataset.
   - Optimizer: Adam.
   - Loss: Categorical Cross-Entropy.

### 3. ResNet50 with Transfer Learning
   - Pre-trained ResNet50V2 model used for feature extraction.
   - Input: 224x224 RGB images (resized from 48x48 grayscale).
   - Transfer learning: Fine-tuned on the FER2013 dataset.
   - Optimizer: Adam.
   - Loss: Categorical Cross-Entropy.

## Training and Evaluation

### Training Process
Each model was trained on the FER2013 training dataset using the Adam optimizer and categorical cross-entropy loss. The training was performed over several epochs, with batch sizes varying based on the model and available memory resources.

### Evaluation
The models were evaluated on the FER2013 test dataset. The following metrics were used to assess performance:
- **Training Accuracy**
- **Test Accuracy**

### Results

| Model                    | Train Accuracy | Test Accuracy |
|--------------------------|----------------|---------------|
| Custom CNN                | 63.11%         | 55.52%        |
| VGG16 Transfer Learning   | 55.93%         | 55.00%        |
| ResNet50 Transfer Learning| 62.61%         | 60.80%        |

## Key Findings

- The **Custom CNN** model achieved the highest training accuracy, demonstrating its potential to learn from scratch when trained on a large dataset. However, its test accuracy indicates a certain level of overfitting.
- **VGG16** struggled with accuracy, possibly due to resizing the input images from 48x48 to 224x224, which might lead to a loss of important features.
- **ResNet50** outperformed the other models in terms of test accuracy, demonstrating the power of Transfer Learning when fine-tuned on specific datasets.

## Conclusion

Among the tested models, **ResNet50 with Transfer Learning** showed the best performance on the test set, making it a suitable choice for emotion detection tasks on the FER2013 dataset. However, further improvements could be made through hyperparameter tuning and better preprocessing techniques for image resizing.

## Future Work

- **Data Augmentation**: Implementing data augmentation techniques such as rotation, flipping, and zooming could help improve model performance by making it more robust to variations in the input data.
- **Model Optimization**: Further hyperparameter tuning and exploration of additional architectures like EfficientNet or MobileNet could yield better results.
- **Deployment**: This model could be deployed as a web application using tools such as TensorFlow.js or Flask for real-time emotion detection.
