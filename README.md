# Optical Character Recognition 
#### Authors: Jigar M Dhemeliya, Rhythm Manish Shah, Siddharth Reddy Busreddy
___

### Group Project for COMP 3106 - Introduction to AI, Fall 2023

#### `./generate_model.py`
- Creates a model(see below) and saves to disk
-  Uses ByClass dataset from 
    [EMNIST on Kaggle](https://www.kaggle.com/datasets/crawford/emnist/) 
    as training data to learn the model
- The model is a simple Convoluted Neural Network with 
    `CP -> CP -> FC -> Softmax` layers

#### `./image_transform.py`
- Extracts individual letters from an image, transforms, and saves the data as a *csv* file to disk

#### `./predict.py`
- Loads model from disk and predicts letters from the csv file created by running the file [above](#image_transformpy)


