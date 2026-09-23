# Image Denoising Autoencoder

A convolutional autoencoder built with TensorFlow/Keras to remove Gaussian noise from Fashion-MNIST images and reconstruct clean images.

## Problem Statement

Image noise can reduce the quality and usefulness of visual data. This project trains a convolutional autoencoder to learn the mapping from noisy Fashion-MNIST images to their corresponding clean images.

The model receives a noisy image as input and learns to reconstruct the original clean image.

## Dataset

The project uses the **Fashion-MNIST** dataset.

- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 pixels
- Grayscale images
- 10 clothing categories

The class labels are used only for visualization and are not used during autoencoder training.

## Approach

### 1. Data Preprocessing

- Loaded the Fashion-MNIST training and test datasets from CSV files.
- Separated labels from pixel values.
- Reshaped the flattened 784-pixel vectors into 28 × 28 images.
- Normalized pixel values from `[0, 255]` to `[0, 1]`.
- Added a channel dimension to create inputs of shape `(28, 28, 1)`.

### 2. Noise Generation

Gaussian noise is added to the normalized images:

- Mean: `0.0`
- Training noise standard deviation: `0.15`

The noisy images are clipped to the valid `[0, 1]` range.

The training data is split into:

- 54,000 training images
- 6,000 validation images
- 10,000 test images

The noisy images are used as inputs, while the original clean images are used as reconstruction targets.

## Model Architecture

The project uses a **Convolutional Autoencoder** consisting of an encoder and decoder.

### Encoder

- Conv2D: 32 filters
- MaxPooling2D
- Conv2D: 64 filters
- MaxPooling2D
- Bottleneck Conv2D: 128 filters

The bottleneck provides a compressed representation of the noisy input.

### Decoder

- Conv2DTranspose: 64 filters
- Conv2DTranspose: 32 filters
- Conv2D: 1 filter with sigmoid activation

The decoder reconstructs the image back to its original `28 × 28 × 1` dimensions.

## Training

The model was trained using:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Mean Squared Error (MSE)
- Batch size: `128`
- Maximum epochs: `50`

Training used:

- Early stopping with patience of 5
- Model checkpointing based on validation loss
- ReduceLROnPlateau for adaptive learning-rate reduction

## Results

On the test set with Gaussian noise of standard deviation `0.15`:

- **Test MSE: 0.00381**

The model was also tested at an **unseen noise level of 0.25** to evaluate robustness:

- Test MSE: `0.00995`
- Average PSNR: `20.29 dB`
- Average SSIM: `0.6959`

The notebook also visualizes:

- Original images
- Noisy images
- Denoised/reconstructed images
- Training vs validation loss

## Model Verification

The trained model was saved in Keras format and loaded again to verify that the saved model produces the same output as the original model.

The mean absolute difference between the original and reloaded model outputs was:

`0.00000000`

## Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- scikit-image

## Project Structure

```text
image-denoising-autoencoder/
│
└── Image_Denoising_Autoencoder.ipynb
