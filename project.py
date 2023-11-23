from PIL import Image
import numpy as np
import os

def bmp_to_pixel(filepath):

    # Open the BMP image 
    image = Image.open(".\img\\" + filepath)

    image = image.convert("L")
    inverted_image = Image.eval(image, lambda x: 255 - x)
    inverted_pixel_values = list(inverted_image.getdata())

    # Flatten the 2D list into a 1D NumPy array (1x728)
    inverted_pixel_array = np.array(inverted_pixel_values).flatten()
    return inverted_pixel_array

def all_images(dir):
    for ele in dir:
        if (ele[-4:] == ".bmp"):
            bmp_to_pixel(ele)
        
    
def main():
    cwd = os.getcwd()
    dir = os.listdir(cwd+'\img')
    all_images(dir)

main()