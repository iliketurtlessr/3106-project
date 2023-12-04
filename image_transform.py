import numpy as np
import matplotlib.pyplot as plt
import cv2


# Constants
input_file = "D:/Fall 23/3106 assignments/3106-project/sample_text.png"
dest_file = 'images_to_test.csv'


def plotImgs(X):
    """Plots first 30 images of X"""
    fig = plt.figure(figsize=(8, 6))
    for i in range(30 if X.shape[0] > 30 else X.shape[0]):
        ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape((28, 28)), cmap='gray')
    plt.show()
    return


# Read and convert image to B/W
image = cv2.imread(input_file)
greyed_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

# Apply theshold
_, threshed_img = cv2.threshold(greyed_img.copy(), 60, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('thresh', thresh)

# Dilate image to take consider letters like 'i', and 'j' as one single letter
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
dilated_img = cv2.dilate(threshed_img, rect_kernel, iterations=1)
# cv2.imshow('dilation', dilated_img)

# Find contours on the dilated image
contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
recognized_letters = []

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
plt.show()

# Convert images to np array
X = np.array(recognized_letters)
print(X.shape)

# Show images
plotImgs(X)

# Save images to file
np.savetxt(dest_file, X, fmt='%.1d', delimiter=',')




