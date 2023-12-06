import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow as tf
import keras
import sys
from termcolor import colored

image_height, image_width = 28, 28
image_size = image_height * image_width
letter_mappings = [
    # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
model_path = 'D:\COMP3106\\3106-project-readme-draft\\3106-model.keras'
predicted_dict = {}


def readAndPredict(input_file):
    # Read and convert image to B/W
    print(colored("Reading the image....",'green', attrs=['bold', 'underline']))
    image = cv2.imread(input_file)
    greyed_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Apply theshold
    _, threshed_img = cv2.threshold(greyed_img.copy(), 60, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('threshed_img', threshed_img)

    # Dilate image to take consider letters like 'i', and 'j' as one single letter
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
    morphed_img = cv2.dilate(threshed_img, rect_kernel, iterations=1)
    # cv2.imshow('morphed_img', morphed_img)

    # Find contours on the dilated image
    contours, _ = cv2.findContours(morphed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    recognized_letters = []
    print(colored("Converting the images into an array....",'green', attrs=['bold', 'underline']))
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # Draw a rectangle around the letter
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        # Single out the letter
        letter = threshed_img[y:y+h, x:x+w]
        
        # Resize image to (24, 24)
        resized_letter = cv2.resize(letter, (24,24))
        
        # Padding the image with 2 pixels of black on all sides
        padded_letter = np.pad(resized_letter, ((2,2),(2,2)), "constant", constant_values=0)
        
        # Adding the preprocessed letter to the list of preprocessed letters
        recognized_letters.append(padded_letter.flatten())
    plt.imshow(image, cmap="gray")

    # Convert images to np array
    X = np.array(recognized_letters)
    X = X / 255.
    X = X.reshape((X.shape[0], image_height, image_width, 1))
    # print(X.shape)

    saved_model = keras.models.load_model(model_path)

    images = X

    # Predict letters
    print(colored("Predicting the letters in the image",'green', attrs=['bold', 'underline']))
    print()
    predictions = saved_model(images)
    pred_labels = [letter_mappings[np.argmax(pred)] for pred in predictions]

    # Print predictions
    # print(colored(''.join(pred_labels),'yellow',attrs=['bold','underline']))
    predicted_dict[os.path.basename(input_file)] = ''.join(pred_labels)

def main():
    directory_path = sys.argv[1]
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith('.png')]
    print(colored("Starting to read and predict the text in the images...",'yellow',attrs=['bold','underline']))
    for file in files:
        readAndPredict(os.path.join(directory_path, file))
    print(colored("DONE...",'yellow',attrs=['bold','underline']))
    print()
    print(colored("All the predictions are as below...",'blue',attrs=['bold','underline']))
    for key in predicted_dict:
        print(colored(f"\t\'{key}\' = \'{predicted_dict[key]}\'", 'magenta', attrs=['bold', 'underline']))
main()
    
