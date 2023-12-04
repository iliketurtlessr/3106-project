import time
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# Constants
letter_mappings = [
    # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
model_path = 'D:/Fall 23/3106 assignments/3106-project/3106-model.keras'
test_letters_path = 'D:/Fall 23/3106 assignments/3106-project/images_to_test.csv'
image_height, image_width = 28, 28
image_size = image_height * image_width


def prepareData(file_path):

    # Read data
    start_time = time.time()
    data = pd.read_csv(file_path, header=None).to_numpy()
    print(f"Read data in {time.time() - start_time} secs")

    # Pre-process data
    x = data / 255.
    x = x.reshape((x.shape[0], image_height, image_width, 1))    
    return x


def plotImgs(X, preds):
    """Plots images of X"""
    fig = plt.figure(figsize=(8, 6))
    n = X.shape[0]
    for i in range(n):
        ax = fig.add_subplot(math.ceil(n / 10), 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape((image_height, image_width)), cmap='gray')
        ax.set_title(preds[i])
    plt.show()
    return

# Load saved model
saved_model = keras.models.load_model(model_path)

# Read and pre-process image to test
images = prepareData(test_letters_path)

# Predict letters
predictions = saved_model(images)
pred_labels = [letter_mappings[np.argmax(pred)] for pred in predictions]

# Print predictions
print(pred_labels)

# Show images along with predictions
plotImgs(images, pred_labels)
