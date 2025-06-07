# Pokémon Type Classification

## Project Overview

This project aims to classify Pokémon based on their **primary type** using their images. Pokémon have distinct animal characteristics and colors that often correspond to their types. For example, Fire-type Pokémon tend to be red, Grass-types are usually green, and Dark-types are often black. This project leverages these visual features to build a Convolutional Neural Network (CNN) model that predicts a Pokémon's primary type from its image.

---

## Goals

- **Classify Pokémon primary types** using image data.
- Build and train a CNN model on a curated dataset of Pokémon images.
- Evaluate model performance on a held-out test set.
- Visualize and analyze classification results.

---

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- Pandas & NumPy
- Matplotlib & Seaborn (for visualization)
- scikit-learn (for train-test split and metrics)
- PIL (Python Imaging Library) for image processing
- OS & shutil libraries for file handling

---

## Dataset

- The dataset consists of Pokémon images and a CSV file containing Pokémon names, primary and secondary types, and evolution information.
- Primary types include categories such as Water, Grass, Fire, Psychic, Rock, Electric, and many others.
- The images are sourced from two directories:  
  - Training and validation images from the original dataset folder.  
  - Test images from a separate Pokémon image dataset.
- The dataset is split into **train**, **validation**, and **test** sets, stratified by primary type.

---

## Data Preparation

- Images are organized into folders by primary type under `train/`, `val/`, and `test/` directories to be compatible with Keras' `ImageDataGenerator`.
- The split proportions are approximately:  
  - Training set: 67%  
  - Validation set: 33%  
  - Test set: separate dataset

---

## Key Steps

1. **Data Loading and Mapping:**  
   Read Pokémon metadata CSV, map images to Pokémon names and types.

2. **Data Splitting:**  
   Stratified splitting of data into train and validation sets.

3. **Folder Organization:**  
   Create directory structure for train, val, and test datasets with subfolders per primary type.

4. **Image Copying:**  
   Copy images into corresponding folders for training, validation, and testing.

5. **Model Building (CNN):**  
   Build and compile a CNN model to classify Pokémon images based on their primary type.


## How to Run

1. Go to Kaggle link: https://www.kaggle.com/code/allenlu112220/pokemon-type-classification
