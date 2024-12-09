import cv2
import numpy as np
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


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
                img_array[i + 1, j] += (5 / 16) * error         # down
            if i < row - 1 and j > 0:
                img_array[i + 1, j - 1] += (3 / 16) * error     # down -> left
            if i < row - 1 and j < column - 1:
                img_array[i + 1, j + 1] += (1 / 16) * error     # down -> right
            if j < column - 1:
                img_array[i, j + 1] += (7 / 16) * error         # right

    # Clip values to be in the valid range [0, 255]
    img_array = np.clip(img_array, 0, 255)
    return img_array.astype(np.uint8)  

def histogram(image):
    """
    Applies histogram on the seleceted image and equalize it
    """
    gray_image = Gray_image(image)
    histogram_arr = np.zeros(256, dtype=int)
    equalized_histogram = np.zeros(256, dtype=int)

    for pixel in gray_image.flatten():
        histogram_arr[pixel] += 1

    # Normalize the histogram
    total_pixels = gray_image.size  # Total number of pixels in the image
    # histogram_arr = histogram_arr / total_pixels  # Divide by total number of pixels to get probabilities
    cdf = histogram_arr.cumsum()  # Cumulative sum of the histogram
    # cdf_normalized = (cdf * 255).astype(np.uint8)  # Scale the CDF values to fit 0-255 range
    
    # Step 3: Normalize the CDF to the range [0, 255]
    cdf_normalized = np.uint8(255 * (cdf - cdf.min()) / (cdf.max() - cdf.min()))
    
    # Step 4: Map the original pixel values to the new ones based on the CDF
    equalized_image = cdf_normalized[gray_image]
    for pixel in equalized_image.flatten():
        equalized_histogram[pixel] += 1
    
    return equalized_image, histogram_arr, equalized_histogram
   
def generateRowColumnSobelGradients():
    """Generates the x-component and y-component of Sobel operators."""
    rowGradient = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel kernel for row (horizontal)
    colGradient = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel kernel for column (vertical)
    return rowGradient, colGradient

def simple_edge_sobel(image,threshold=3):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Sobel kernels for X and Y gradients
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X gradient (vertical edges)
    kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Y gradient (horizontal edges)

    # Get the height and width of the image
    height, width = gray.shape

    # Create empty arrays to store the gradients
    sobelx = np.zeros_like(gray, dtype=np.float32)
    sobely = np.zeros_like(gray, dtype=np.float32)

    # Apply convolution manually for each pixel (excluding borders)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 region around the pixel (i, j)
            region = gray[i - 1:i + 2, j - 1:j + 2]
            
            # Apply the kernel for X and Y gradients
            grad_x = np.sum(region * kernelx)
            grad_y = np.sum(region * kernely)

            # Store the gradients in the respective arrays
            sobelx[i, j] = grad_x
            sobely[i, j] = grad_y

    # Calculate the gradient magnitude (edge strength)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize the magnitude to the range [0, 255]
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    # Thresholding: Keep strong edges, discard weak ones
    thresholded_result = np.zeros_like(magnitude)
    thresholded_result[magnitude >= threshold] = 255  # Assign 255 to strong edges

    # Print the edge array after thresholding
    print("Edge Array (after thresholding):")
    print(thresholded_result)

    # Return both the magnitude and the thresholded result
    return magnitude

def simple_edge_prewitt(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Prewitt kernels for X and Y gradients
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # X gradient (horizontal edges)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Y gradient (vertical edges)

    # Get the height and width of the image
    height, width = gray.shape

    # Create empty arrays to store the gradients
    prewittx = np.zeros_like(gray, dtype=np.float32)
    prewitty = np.zeros_like(gray, dtype=np.float32)

    # Apply convolution manually for each pixel (excluding borders)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 region around the pixel (i, j)
            region = gray[i - 1:i + 2, j - 1:j + 2]
            
            # Apply the kernel for X and Y gradients
            grad_x = np.sum(region * kernelx)
            grad_y = np.sum(region * kernely)

            # Store the gradients in the respective arrays
            prewittx[i, j] = grad_x
            prewitty[i, j] = grad_y

    # Calculate the gradient magnitude (edge strength)
    magnitude = np.sqrt(prewittx**2 + prewitty**2)

    # Normalize the magnitude to the range [0, 255]
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    return magnitude

def simple_edge_kirsch(image):
    """
    Apply Kirsch edge detection on an input image.

    Args:
        image: Input image (BGR format).

    Returns:
        kirsch_magnitude: The edge magnitude image (grayscale).
        dominant_direction: The dominant edge direction label (compass direction) for the entire image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Kirsch kernels for 8 compass directions
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # N
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # NW
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # W
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # SW
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # S
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # SE
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # E
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])   # NE
    ]

    # Apply Kirsch kernels to the image
    kirsch_edges = [cv2.filter2D(gray, -1, kernel) for kernel in kirsch_kernels]

    # Calculate magnitude (maximum response) and direction (index of maximum response)
    kirsch_magnitude = np.max(kirsch_edges, axis=0)
    kirsch_direction_indices = np.argmax(kirsch_edges, axis=0)

    # Map indices to compass labels
    compass_labels = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]

    # Compute the dominant direction over the entire image
    dominant_direction_index = np.bincount(kirsch_direction_indices.flatten()).argmax()
    dominant_direction = compass_labels[dominant_direction_index]

    return kirsch_magnitude, dominant_direction    # Convert to grayscale
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
    Applies advanced edge detection (homogeneity)
    """
    threshold_value = 5
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    homogeneity_image = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = gray_image[i, j]
            differences = [
                abs(float(center) - float(gray_image[i-1, j-1])),
                abs(float(center) - float(gray_image[i-1, j])),
                abs(float(center) - float(gray_image[i-1, j+1])),
                abs(float(center) - float(gray_image[i, j-1])),
                abs(float(center) - float(gray_image[i, j+1])),
                abs(float(center) - float(gray_image[i+1, j-1])),
                abs(float(center) - float(gray_image[i+1, j])),
                abs(float(center) - float(gray_image[i+1, j+1])),       
            ]
            homogeneity_value = max(differences)
            homogeneity_image[i, j] = homogeneity_value if homogeneity_value > threshold_value else 0

    return homogeneity_image

def advanced_edge_difference(image):
    """
    Applies advanced edge detection(difference)
    """
    threshold_value = 10
    gray_image = Gray_image(image)
    height, width = gray_image.shape
    difference_image = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Remove trailing commas and unpack the tuples
            difference1 = abs(gray_image[i-1, j-1] - gray_image[i+1, j+1])
            difference2 = abs(gray_image[i-1, j+1] - gray_image[i+1, j-1])
            difference3 = abs(gray_image[i, j-1] - gray_image[i, j+1])
            difference4 = abs(gray_image[i-1, j] - gray_image[i+1, j])
            
            # Create a list of differences and use max()
            differences = [difference1, difference2, difference3, difference4]
            max_difference = max(differences)
            
            difference_image[i, j] = max_difference if max_difference > threshold_value else 0

    return difference_image

def advanced_edge_differenceofGaussians(image):
    """
    Applies advanced edge detection(difference of Gaussians 7*7 and 9*9)
    """
    gray_image = Gray_image(image)
    
    mask_7x7 = np.array([
        [0, 0, -1, -1, -1, 0, 0],
        [0, -2, -3, -3, -3, -2, 0],
        [-1, -3, 5, 5, 5, -3, -1],
        [-1, -3, 5, 16, 5, -3, -1],
        [-1, -3, 5, 5, 5, -3, -1],
        [0, -2, -3, -3, -3, -2, 0],
        [0, 0, -1, -1, -1, 0, 0]
    ], dtype=np.float32)

    mask_9x9 = np.array([
        [0, 0, 0, -1, -1, -1, 0, 0, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [-1, -3, -1, 9, 19, 9, -1, -3, -1],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0]
    ], dtype=np.float32)

    image_7x7 = cv2.filter2D(gray_image, -1, mask_7x7)
    image_9x9 = cv2.filter2D(gray_image, -1, mask_9x9)
    DifferenceOfGaussian = np.abs(image_7x7 - image_9x9)
    
    return DifferenceOfGaussian

def advanced_edge_contrastBased(image):
    """
    Applies advanced edge detection (contrast-based) with brightness, gamma correction, and contrast enhancement.
    
    Parameters:
        image (numpy.ndarray): Input image (RGB or grayscale).
        epsilon (float): Small constant to avoid division by zero.
    
    Returns:
        numpy.ndarray: Contrast-enhanced edge-detected image.
    """
    epsilon = 1e-10
    gray_image = Gray_image(image)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    contrast_enhanced = clahe.apply(gray_image)

    # Apply gamma correction to brighten the image
    gamma = 0.5  
    gamma_corrected = np.power(contrast_enhanced / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    # Define Laplacian edge-detection mask
    edge_mask = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])

    # Define smoothing mask 
    smoothing_mask = np.ones((3, 3)) / 9

    # Apply edge detection
    image_edge = cv2.filter2D(gamma_corrected, -1, edge_mask)

    # Apply smoothing
    image_smooth = cv2.filter2D(gamma_corrected, -1, smoothing_mask).astype(np.float32)

    # Avoid division by zero
    image_smooth += epsilon

    # Compute contrast-enhanced edge image
    contrast_image_edge = image_edge / image_smooth

    # Normalize the result to the range 0–255
    contrast_image_edge = cv2.normalize(contrast_image_edge, None, 0, 255, cv2.NORM_MINMAX)

    return contrast_image_edge.astype(np.uint8)

def advanced_edge_varianceBased(image):
    """
    Applies variance operator
    """
    gray_image=Gray_image(image)
    variance_edge_image =np.zeros_like(gray_image)
    height,width=gray_image.shape
    for i in range(1,height-1):
        for j in range(1,width-1):
            neighborhood=gray_image[i-1:i+2,j-1:j+2]
            mean=np.mean(neighborhood)
            variance= np.sum((neighborhood-mean)**2)/9
            variance_edge_image[i,j]=variance

    return variance_edge_image

def advanced_edge_rangeBased(image):
    """
    Applies range operator
    """
    gray_image=Gray_image(image)
    range_edge_image =np.zeros_like(gray_image)
    height,width=gray_image.shape
    for i in range(1,height-1):
        for j in range(1,width-1):
            neighborhood=gray_image[i-1:i+2,j-1:j+2]
            range_value= np.max(neighborhood)-np.min(neighborhood)
            range_edge_image[i,j]=range_value

    return range_edge_image

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

def manual_segmentation(image, low_thresh, high_thresh, value=255):
    """
    Manual segmentation of an image using a low and high threshold. Where pixels within the threshold are set to `value`, and all other pixels are set to 0.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image)
    mask[(image >= low_thresh) & (image <= high_thresh)] = value
    return mask

def histogram_peaks_segmentation(image, num_clusters=3, value=255, color_palette='tab10'):
    """
    Segmentation of an image using histogram peaks.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the histogram of the image
    hist = cv2.calcHist([image], [0], None, [256], [0, 255]).ravel()

    peaks = find_hist_peaks(hist)
    
    print(f'All Peaks: {peaks}')

    # If fewer peaks than requested clusters, raise an error
    if len(peaks) < num_clusters:
        raise ValueError(f"Not enough distinct peaks ({len(peaks)}) for {num_clusters} clusters")

    # Select top peaks
    top_peaks = peaks[:num_clusters]
    
    print(f'Selected Peaks: {top_peaks}')

    # If 2 clusters, use original binary segmentation method
    if num_clusters == 2:
        low_thresh, high_thresh = calc_threshs(top_peaks)
        mask = np.zeros_like(image)
        mask[(image >= low_thresh) & (image <= high_thresh)] = value
        return mask, None
    
    # Generate unique colors using a colormap
    colors = plt.colormaps[color_palette](np.linspace(0, 1, num_clusters))[:, :3] * 255
    
    # Create a color-coded segmentation image
    segmented_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cluster_masks = []
    
    # Create thresholds between peaks
    thresholds = [0] + [(top_peaks[i] + top_peaks[i+1]) // 2 for i in range(len(top_peaks)-1)] + [255]
    
    # Segment the image
    for i in range(num_clusters):
        # Create binary mask for this cluster
        cluster_mask = np.zeros_like(image, dtype=bool)
        cluster_mask = (image >= thresholds[i]) & (image < thresholds[i+1])
        
        # Color the segmented image
        color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        color_image[cluster_mask] = colors[i]
        segmented_image[cluster_mask] = color_image[cluster_mask]
        
        # Create and store binary mask with value
        binary_mask = np.zeros_like(image)
        binary_mask[cluster_mask] = value
        cluster_masks.append(binary_mask)
    
    return segmented_image, cluster_masks

def calc_threshs(peaks):
    """
    Calculate the thresholds based on the histogram peaks.
    """
    peak1, peak2 = peaks
    low_thresh = (peak1 + peak2) // 2
    high_thresh = peak2

    return low_thresh, high_thresh

def histogram_valleys_segmentation(image):
    """
    Segmentation of an image using histogram valleys.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 255]).ravel()

    # Find peaks in the histogram
    peaks = find_hist_peaks(hist)[:2]
    
    print(f'All Peaks: {peaks}')

    # Find valleys in the histogram
    valley_point = find_hist_valley(peaks, hist)

    print(f'Valley Point (low threshold): {valley_point}')

    low_thresh, high_thresh = valley_point, peaks[1]

    mask = np.zeros_like(image)
    mask[(image >= low_thresh) & (image <= high_thresh)] = 255

    return mask

def find_hist_peaks(hist):
    """
    Find peaks in the histogram.
    """
    peaks, _ = find_peaks(hist, height=0)
    return sorted(peaks, key=lambda x: hist[x], reverse=True)

def find_hist_valley(peaks, hist):
    """
    Find valley point in the histogram.
    """
    point = 0
    min_val = float('inf')
    start, end = peaks
    for i in range(start, end+1):
        if hist[i] < min_val:
            min_val = hist[i]
            point = i
    return point

def histogram_adaptive_segmentation(image):
    """
    Segmentation of an image using adaptive histogram thresholding.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 255]).ravel()

    # Find peaks in the histogram
    peaks = find_hist_peaks(hist)[:2]

    print(f'All Peaks: {peaks}')

    # Find valleys in the histogram
    valley_point = find_hist_valley(peaks, hist)

    print(f'Valley Point (low threshold): {valley_point}')

    low_thresh, high_thresh = valley_point, peaks[1]

    mask = np.zeros_like(image)
    mask[(image >= low_thresh) & (image <= high_thresh)] = 255

    # Apply adaptive thresholding
    background_mean, object_mean = calc_means(mask, image)

    peaks = [int(background_mean), int(object_mean)]
    low_thresh, high_thresh = find_hist_valley(peaks, hist), peaks[1]

    print(f'Adaptive Thresholds: {low_thresh}, {high_thresh}')

    mask = np.zeros_like(image)
    mask[(image >= low_thresh) & (image <= high_thresh)] = 255

    return mask

def calc_means(mask, image):
    """
    Calculate the mean pixel values for the background and object regions.
    """
    foreground = image[mask == 255]
    background = image[mask == 0]

    return np.mean(background), np.mean(foreground)