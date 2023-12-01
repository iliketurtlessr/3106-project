
import os
import pprint
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt


# Constants
letter_mappings = ['_' ,'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
num_classes = 26 + 1
train_letters_path = 'D:/EMNIST/emnist-letters-train.csv'
test_letters_path = 'D:/EMNIST/emnist-letters-test.csv'
model_path = 'D:/Fall 23/3106 assignments/3106-project/3106-model.h5'
image_height, image_width = 28, 28
image_size = image_height * image_width
batch_size = 64
n_epochs = 20


def plotImgs(X):
    # plot the first 20 images of X
    fig = plt.figure(figsize=(16, 6))
    for i in range(15):
        ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape((image_height, image_width)), cmap='gray')
    plt.show()
    return


def prepareData(file_path):

    # Read data
    start_time = time.time()
    data = pd.read_csv(file_path, header=None).to_numpy()
    print(f"Read data in {time.time() - start_time} secs")

    # Pre-process data
    y = data[:, 0]
    x = data[:, 1:] / 255.
    x = x.reshape((x.shape[0], image_height, image_width, 1))
    print(f"x: {x.shape}, y:{y.shape}")
    
    return x, y

def train_model(train_images, train_labels):
    # train_images, train_labels = prepareData(train_letters_path)

    # Show 15 training images
    # print([letter_mappings[label] for label in train_labels[20:20+15]])
    # plotImgs(train_images[20:])

    train_labels = keras.utils.to_categorical(train_labels, num_classes)

    model = keras.Sequential([ 
        keras.layers.Conv2D(32, kernel_size=3, input_shape=(image_height, image_width, 1)),
        keras.layers.MaxPooling2D(2, strides=2),

        keras.layers.Flatten(input_shape=(image_height, image_width, 1)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()

    # Configure model
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint_cb = ModelCheckpoint(
        model_path, monitor='val_accuracy', verbose=1, save_best_only=True, 
        mode='max'
    )
    earlystopping_cb = EarlyStopping(
        monitor='val_accuracy', min_delta=0, patience=3, verbose=1, 
        mode='max', restore_best_weights = True
    )
    reduceLr_cb = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=0.0001
    )
    # Learn model
    history = model.fit(
        train_images, train_labels, batch_size=batch_size, epochs=n_epochs, 
        validation_split=0.2,
        # validation_data=(x_test, y_test), 
        callbacks=[checkpoint_cb, earlystopping_cb, reduceLr_cb]
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

    return model


if __name__ == '__main__':

    # Read and pre-format training data
    train_images, train_labels = prepareData(train_letters_path)

    # Train Model
    model = train_model(train_images, train_labels)
    print(model)

    # Load saved model
    # saved_model = keras.models.load_model(model_path)

    # saved_model.summary()




