import cv2
import numpy as np

def read_image(file_path):
    """
    Reads an image using OpenCV.
    """
    return cv2.imread(file_path)

def Gray_image(image):
    """
    Converts image's color to GRAYSCALE using OpenCV
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def threshold(image):
    """
    Applies a threshold to the grayscale version of the image.
    Returns the threshold value.
    """
    gray_image = Gray_image(image)
    threshold_value = np.mean(gray_image)
    return threshold_value

def simple_halftoning(image, block_size=4):
    """
    Applies a simple halftoning technique without error diffusion.
    """
    gray_image = Gray_image(image)
    row, column = gray_image.shape
    simple_halftoned_image = np.zeros_like(gray_image)

    for i in range(row):
        for j in range(column):
            simple_halftoned_image[i,j] = 255 if gray_image[i,j] > 128 else 0

    return simple_halftoned_image

def advanced_halftoning(image):
    """
    Applies advanced halftonig technique
    """
    gray_image = Gray_image(image)
    img_array = np.array(gray_image, dtype=np.float32)  # float to handle errors
    row, column = img_array.shape

    for i in range(row):
        for j in range(column):
            new_pixel = 255 if img_array[i, j] > 128 else 0
            error = img_array[i, j] - new_pixel
            img_array[i, j] = new_pixel
            # Error diffusion to neighboring pixels
            if i < row - 1: 
                img_array[i + 1, j] += (5 / 16) * error
            if i < row - 1 and j > 0:
                img_array[i + 1, j - 1] += (3 / 16) * error
            if i < row - 1 and j < column - 1:
                img_array[i + 1, j + 1] += (1 / 16) * error
            if j < column - 1:
                img_array[i, j + 1] += (7 / 16) * error

    # Clip values to be in the valid range [0, 255]
    img_array = np.clip(img_array, 0, 255)
    return img_array.astype(np.uint8)  

def histogram(imag):
    """
    Applies histogram on the seleceted image
    """
    return 0

def simple_edge_sobel(image):
    """
    Applies simple edge detection(sobel)
    """
    return 0

def simple_edge_prewitt(image):
    """
    Applies simple edge detection(prewitt)
    """
    return 0

def simple_edge_kirsch(image):
    """
    Applies simple edge detection(kirsch) and get its edge direction
    """
    return 0


def advanced_edge_homogeneity(image):
    """
    Applies advanced edge detection(homogeneity)
    """
    return 0

def advanced_edge_difference(image):
    """
    Applies advanced edge detection(difference)
    """
    return 0

def advanced_edge_differenceofGaussians(image):
    """
    Applies advanced edge detection(difference of Gaussians 7*7 and 9*9)
    """
    return 0

def advanced_edge_contrastBased(image):
    """
    Applies advanced edge detection(contrast-based)
    """
    return 0

def high_bass_filtering(image):
    """
    Applies filtering(high-bass)
    """
    return 0

def low_bass_filtering(image):
    """
    Applies filtering(low-bass)
    """
    return 0

def add_image(image):
    """
    Applies operations on the image(add to its copy)
    """
    return 0

def subtract_image(image):
    """
    Applies operations on the image(subtract from its copy)
    """
    return 0

def histogram_segementation(image):
    """
    """
    return 0