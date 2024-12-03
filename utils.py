import cv2
import numpy as np
import math

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

def simple_halftoning(image):
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

def generateRowColumnSobelGradients():
    """Generates the x-component and y-component of Sobel operators."""
    rowGradient = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel kernel for row (horizontal)
    colGradient = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel kernel for column (vertical)
    return rowGradient, colGradient

def simple_edge_sobel(image):
    """
    Perform Sobel edge detection from scratch.

    Args:
        image (numpy array): Input grayscale image as a 2D array.

    Returns:
        result_rgb (numpy array): Sobel edge-detected image in RGB format.
    """
    # Ensure image is in grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    rows, cols = image.shape

    # Initialize result array
    result = np.zeros((rows, cols), dtype=float)

    # Pad the image to handle edges
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    # Generate Sobel kernels
    rowGradient, colGradient = generateRowColumnSobelGradients()

    # Apply Sobel operator
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Extract the 3x3 subregion
            subimage = padded_image[i-1:i+2, j-1:j+2]

            # Compute row and column gradients
            rowSum = np.sum(rowGradient * subimage)
            colSum = np.sum(colGradient * subimage)

            # Compute gradient magnitude
            result[i-1, j-1] = math.sqrt(rowSum**2 + colSum**2)

    # Normalize the result to 0-255
    result = (result / result.max() * 255).astype(np.uint8)

    # Convert the result to RGB for display purposes
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result_rgb

def simple_edge_prewitt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Define Prewitt kernels
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # X gradient
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Y gradient

    # Apply the Prewitt operator
    prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)  # Use CV_64F for better precision
    prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)

   
    magnitude = np.sqrt(prewittx**2 + prewitty**2)
    
    # Normalize to preserve contrast
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = cv2.convertScaleAbs(magnitude)  # Convert to 8-bit

    return magnitude



def simple_edge_kirsch(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]), 
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]), 
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])   
    ]
    
    # Apply Kirsch kernels
    kirsch_edges = [cv2.filter2D(gray, -1, kernel) for kernel in kirsch_kernels]
    
    
    kirsch_magnitude = np.max(kirsch_edges, axis=0)
    kirsch_direction = np.argmax(kirsch_edges, axis=0)  
    
    
    kirsch_magnitude = cv2.convertScaleAbs(kirsch_magnitude)
    
    return kirsch_magnitude, kirsch_direction


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
    #defining the high pass filter
    mask_high_pass= np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]
                              ], dtype=np.float32)
    # Check if the image is in color (has 3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def conv(image, mask1):
        result=cv2.filter2D(image, -1, mask1)
        return result
    
    highpass=conv(image,mask_high_pass)
    return highpass

def low_bass_filtering(image):
    #defining the low pass filter
    mask_low_pass= np.array([[0, 1/6, 0],
                              [1/6, 2/6, 1/6],
                              [0, 1/6, 0]
                              ], dtype=np.float32)
    # Check if the image is in color (has 3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def conv(image, mask1):
        result=cv2.filter2D(image, -1, mask1)
        return result
    
    lowpass=conv(image,mask_low_pass)
    return lowpass

def median_filtering(image):
    # Check if the image is in color (has 3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    kernel_size=5
    image_height, image_width = image.shape
        
    # Padding to handle borders (the filter will not work at borders without padding)
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
        
    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)
        
    # Iterate over the image
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest (kernel area)
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
                
            # Find the median of the region
            median_value = np.median(region)
                
            # Set the pixel in the filtered image to the median value
            filtered_image[i, j] = median_value

    return filtered_image

def add_image(image):
    image = Gray_image(image)
    image2 = image
    height, width = image.shape
    added_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            added_image[i, j] = image[i, j] + image2[i, j]
            added_image[i, j] = max(0, min(added_image[i, j],255))
    return added_image

def subtract_image(image):
    image = Gray_image(image)
    image2 = image
    height, width = image.shape
    subtracted_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            subtracted_image[i, j] = image[i, j] - image2[i, j]
            subtracted_image[i, j] = max(0, min(subtracted_image[i, j],255))
    return subtracted_image

def invert_image(image):
    image = Gray_image(image)
    height, width = image.shape
    Inverted_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            Inverted_image[i, j] = 255 - image[i, j]

    return Inverted_image

def histogram_segementation(image):
    """
    """
    return 0
