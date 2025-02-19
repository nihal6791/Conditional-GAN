# Conditional GAN for Crack Image Generation (Class 4)

This repository contains a Conditional Generative Adversarial Network (CGAN) trained to generate synthetic crack images of class 4. This project is part of the MIIRAG-Task-2.

## Based On

This model is based on the implementation from [realBarry123/pytorch-conditional-gan](https://github.com/realBarry123/pytorch-conditional-gan/tree/main) with significant modifications to adapt it for crack image generation using the EDM600 dataset and focus on generating only class 4 crack images.

## Key Modifications

The following key modifications were made to the original repository:

*   **Dataset Adaptation:**
    *   The data loading pipeline was modified to work with the EDM600 crack image dataset, which is structured into different classes.
    *   Image transformations were applied to resize images to 64x64 pixels and normalize pixel values to the range [-1, 1]. A grayscale conversion step was also added.
*   **Model Architecture:**
    *   The original generator and discriminator architectures were adjusted to better suit the characteristics of the crack image dataset. The channel configuration and layer sizes were modified to enhance the quality of generated images.
    *   The latent dimension was set to 100.
*   **Class-Specific Generation:**
    *   The training process was modified to focus solely on generating class 4 crack images.
    *   Label conditioning was implemented to control the class of generated images during training and inference.
*   **Training Process:**
    *   Hyperparameters were adjusted to optimize the training process for the EDM600 dataset, including the learning rate, batch size, and number of epochs.
    *   Code added to automatically utilizes GPU if available, and falls back to CPU if not.
*   **Output and Saving:**
    *   The code was updated to support saving generated images and model weights after each epoch.
*   **Bug Fixes and Improvements:**
    *   The channel mismatch issue in the Discriminator was resolved by adjusting the number of input channels and modifying the forward pass.
    *   Various other minor bug fixes and improvements were implemented to enhance stability and performance.

## Dataset

The model was trained on a cleaned version of the EDM600 crack image dataset, which consists of crack images divided into 8 classes (class\_0 to class\_7). A subset of the dataset, focusing on class 4, was used to train the model. The dataset is structured into training, testing, and validation sets and stored in ImageFolder format. Dataset Link: https://github.com/WafaAlKathiri/MIIRAG-Task-2/tree/main/CleanDATA

## How To use it:
1. Upload your dataset.
2. Open train.py file and Locate line 33 in the notebook and replace it with the path to your dataset.
3. Upload the model files to the appropriate directory.
4. Run the notebook file.