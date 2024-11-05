# Flower-Classification-using-VGG16-in-CommandLine
This is the extension to my previous classification project allowing users to run the model in the command line.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Files](#files)

## Project Overview

This project focuses on developing an image classification model capable of predicting the species of flowers in the Oxford Flowers dataset. Using transfer learning with VGG16 for feature extraction, the model achieves high performance by leveraging pre-trained weights from a large-scale dataset. Data augmentation, regularization, and fine-tuning techniques were also employed to enhance the model's robustness.

## Dataset

- **Name**: [Oxford Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/)
- **Categories**: The dataset contains images from 102 different flower categories.
- **Structure**: The images are divided into training, validation, and test sets.
- **Format**: Images are organized by class and come with labeled annotations.

## Model Architecture

This project utilizes the pre-trained VGG16 model for feature extraction, followed by custom dense layers for classification. The VGG16 model, pre-trained on ImageNet, enables efficient feature extraction, while the dense layers adapt the model for classifying flower species.

**Key components:**
- **VGG16**: Used as a feature extractor by freezing its layers.
- **Custom Dense Layers**: Added on top of VGG16 for classification, tailored to the number of flower species in the dataset.

## Installation

Clone the repository:
```bash
git clone https://github.com/BhaswanthReddyI/Flower-Classification-using-VGG16-in-CommandLine.git
cd Flower-Classification-using-VGG16-in-CommandLine
```

## Usage

### Training the Model

To train the model, run the following command in the terminal:

```bash
python train.py <data_dir> [--save_dir <save_dir>] [--arch <architecture>] [--learning_rate <learning_rate>] [--hidden_units <hidden_units>] [--epochs <epochs>] [--gpu]
```

**Arguments:**
- `<data_dir>`: Directory containing the dataset. The dataset should have subdirectories for training, validation, and testing.
- `--save_dir`: Directory to save the trained model checkpoint (default: current directory).
- `--arch`: Model architecture to use (default: `vgg13`).
- `--learning_rate`: Learning rate for the optimizer (default: `0.001`).
- `--hidden_units`: Number of hidden units in the classifier (default: `512`).
- `--epochs`: Number of epochs to train the model (default: `20`).
- `--gpu`: Use GPU for training if available.

### Making Predictions

To make predictions on a new image, use the following command:

```bash
python predict.py <image_path> <checkpoint> [--top_k <top_k>] [--category_names <category_names>] [--gpu]
```
**Arguments:**
- `<image_path>`: Path to the image file you want to predict.
- `<checkpoint>`: Path to the saved model checkpoint.
- `--top_k`: Return top K most likely classes (default: `5`).
- `--category_names`: Path to the JSON file mapping category indices to names.
- `--gpu`: Use GPU for inference if available.

## Files

- `train.py`: Script for training the flower classification model.
- `predict.py`: Script for making predictions using the trained model.
- `utils.py`: Utility functions for data loading, processing, and model checkpointing.
- `README.md`: Project documentation.



