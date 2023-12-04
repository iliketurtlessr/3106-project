import pprint
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt
from extra_keras_datasets import emnist


# Constants
letter_mappings = [
    # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
num_classes = len(letter_mappings)
batch_size = 500
n_epochs = 20
model_name = '3106-model'


def plotImgs(X, y):
    """Plots first 30 images of X"""
    fig = plt.figure(figsize=(16, 6))
    for i in range(30):
        ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i], cmap='gray')
        ax.set_title(letter_mappings[y[i]])
    plt.show()


# Fetch EMNIST-byClass data
(train_images, train_labels), (test_images, test_labels) = emnist.load_data(type='byclass')
print("=== og data ===",
      f"train: {train_images.shape}, {train_labels.shape}",
      f"test: {test_images.shape}, {test_labels.shape}", sep='\n', end='\n\n')


# Keep alphabet data only i.e., discard digit images and labels
letter_indices = np.logical_and(train_labels >= 10, train_labels < 62)
train_x, train_y = train_images[letter_indices], train_labels[letter_indices]

letter_indices_test = np.logical_and(test_labels >= 10, test_labels < 62)
test_x, test_y = test_images[letter_indices_test], test_labels[letter_indices_test]
print("=== after slicing digits ===",
      f"train x, y: {train_x.shape}, {train_y.shape}",
      f"test x, y: {test_x.shape}, {test_y.shape}", sep='\n', end='\n\n')


# Pre-process data
shape = (28, 28, 1)
train_x = train_x.reshape(train_x.shape[0], shape[0], shape[1], shape[2])
test_x = test_x.reshape(test_x.shape[0], shape[0], shape[1], shape[2])
train_x, test_x = train_x / 255.0, test_x / 255.0       # normalize image data
train_y, test_y = train_y - 10, test_y - 10             # shift labels to 10 left
plotImgs(train_x, train_y)                              # show first 30 train images
train_y, test_y = to_categorical(train_y, num_classes), to_categorical(test_y, num_classes)


# Create Model
model = keras.Sequential([
    # CP
    Conv2D(64, (3, 3), activation='relu', input_shape=shape),
    MaxPool2D((2, 2)),
    BatchNormalization(),
    
    # CP
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    BatchNormalization(),
    
    Dropout(0.5), Flatten(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
], name='CNN-model')
model.summary()                 # print model architecture
model.compile(
    optimizer='adam',           # default lr: 0.001 
    loss='categorical_crossentropy', metrics=['accuracy']
)

# Define callbacks to pass into model
checkpoint_cb = ModelCheckpoint(
    model_name+'.keras', monitor='val_accuracy', mode='max',
    verbose=1, save_weights_only=False, save_best_only=True
)
earlystopping_cb = EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=3, mode='max',
    verbose=1, restore_best_weights = True
)
reduceLr_cb = ReduceLROnPlateau(
    monitor='val_loss', mode='min', factor=0.2, patience=3, min_lr=0.0001,
    verbose=1
)

# Learn model
history = model.fit(
    train_x, train_y, batch_size=batch_size, epochs=n_epochs, 
    validation_split=0.2, verbose=2,
    callbacks=[checkpoint_cb, earlystopping_cb, reduceLr_cb],
)
h = history.history
pprint.pprint(h)


# Plot metrics
epochs_ran = len(h['accuracy'])
train_losses, train_accs = h['loss'], h['accuracy']
val_losses, val_accs = h['val_loss'], h['val_accuracy']
plt.plot(range(1, epochs_ran + 1), train_losses, color='red', label='Train Loss')
plt.plot(range(1, epochs_ran + 1), val_losses, color='blue', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend()
plt.show()

plt.plot(range(1, epochs_ran + 1), train_accs, color='orange', label='Training Accuracy')
plt.plot(range(1, epochs_ran + 1), val_accs, color='green', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Test model with test data
loss, accuracy = model.evaluate(test_x, test_y, verbose=2)
print(f"=== Test data metrics ===",
      f"Loss: {loss}",
      f"Accuracy: {accuracy*100:.3f}%", sep="\n")

