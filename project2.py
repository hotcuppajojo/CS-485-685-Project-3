# JoJo Petersky
# CS 485/685 Spring '24 Project2
# 2024/04/07
# project2.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors

# Image Loading & Displaying Functions
# ------------------------------------    

def load_img(file_name, grayscale=False):
    """
    Loads an image from a file.

    Parameters:
    - file_name: The path to the image file.
    - grayscale: A boolean flag indicating whether to load the image as grayscale.

    Returns:
    - The loaded image as a numpy array.

    Raises:
    - IOError: If the image cannot be loaded.
    """
    try:
        # Choose the color mode based on the grayscale flag
        # This allows the function to be flexible and load both color and grayscale images
        color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

        # Attempt to load the image
        # cv2.imread is used because it can load images in different color modes
        img = cv2.imread(file_name, color_mode)

        # Check if the image was loaded successfully
        # cv2.imread returns None if it cannot load the image, so we raise an exception in this case
        if img is None:
            raise ValueError(f"The file {file_name} cannot be loaded as an image.")
        
        return img
    except Exception as e:
        # Handle other potential exceptions (e.g., file not found, no read permissions)
        # This is done to provide a more informative error message to the user
        raise IOError(f"An error occurred when trying to load the image: {e}")
    
def display_img(image):
    """
    Displays an image using matplotlib's imshow function.

    Parameters:
    - image: A numpy ndarray representing the image to be displayed.

    Raises:
    - ValueError: If the input image is None.
    - TypeError: If the input is not a numpy ndarray.
    - ValueError: If the input is not a 2D (grayscale) or 3D (color) image.
    """
    # Check if the image is None
    if image is None:
        raise ValueError("Input image cannot be None.")
    
    # Check if the image is a numpy ndarray
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray.")
    
    # Check if the image is 2D (grayscale) or 3D (color)
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D (grayscale) or 3D (color) image.")
    
    # If the image is grayscale, display it with a grayscale colormap
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        # If the image is color, convert it from BGR to RGB (as OpenCV uses BGR instead of RGB)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes as they are not necessary for image display
    plt.show()  # Display the plot

# Keypoint Functions
# ------------------

def plot_keypoints(image, keypoints):
    """
    Plots keypoints on an image.

    Parameters:
    - image: A numpy ndarray representing the image.
    - keypoints: A list of tuples representing the keypoints.

    Returns:
    - The image with the keypoints plotted on it.
    """

    # Create a copy of the image to avoid modifying the original
    image_copy = image.copy()

    # Check if image is color
    # This is done because the cv2.circle function expects a 3-channel image
    # If the image is grayscale (2D), it needs to be converted to RGB (3D)
    if len(image_copy.shape) == 3:
        image_rgb = image_copy
    else:
        # Convert the grayscale image to RGB
        # cv2.cvtColor is used because it can convert images between different color spaces
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)

    # Plot keypoints
    # This is done by drawing a small circle at each keypoint location
    # The color of the circle is set to red (255, 0, 0) for visibility
    for keypoint in keypoints:
        cv2.circle(image_rgb, keypoint, 1, (255, 0, 0), -1)

    return image_rgb

def extract_LBP(image, keypoint):
    """
    Extracts the Local Binary Pattern (LBP) descriptor from the image at the given keypoint.

    Parameters:
    - image: A numpy ndarray representing the image.
    - keypoint: A tuple representing the x and y coordinates of the keypoint.

    Returns:
    - lbp: A numpy ndarray representing the LBP descriptor.
    """

    # Define the LBP parameters
    # P: Number of circularly symmetric neighbour set points (quantization of the angular space)
    # R: Radius of circle (spatial resolution of the operator)
    # These parameters are chosen to capture the texture information at different scales
    P = 8  
    R = 1  

    # Get the coordinates of the keypoint
    x, y = keypoint

    # Initialize the LBP feature vector
    # This will hold the LBP descriptor for the keypoint
    lbp = np.zeros((P,), dtype=int)

    # If the image is a color image, compute the LBP descriptor for each color channel
    # This is done because the texture information can be different in each color channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        lbp = []
        for channel in range(3):
            lbp_channel = extract_LBP_channel(image[:, :, channel], x, y, P, R)
            lbp.append(lbp_channel)
        # Concatenate the LBP descriptors from all channels to form the final descriptor
        lbp = np.concatenate(lbp)
    else:  
        # If the image is a grayscale image, compute the LBP descriptor for the single channel
        # This is because grayscale images only have one channel
        lbp = extract_LBP_channel(image, x, y, P, R)

    return lbp

def extract_LBP_channel(image, x, y, P, R):
    """
    Extracts the Local Binary Pattern (LBP) descriptor from a single channel of an image at a given keypoint.

    Parameters:
    - image: A 2D numpy ndarray representing the image channel.
    - x, y: The coordinates of the keypoint.
    - P: The number of circularly symmetric neighbour set points.
    - R: The radius of the circle.

    Returns:
    - lbp: A one-dimensional numpy ndarray representing the LBP descriptor.
    """

    # Initialize the LBP feature vector for the channel
    # This will hold the LBP descriptor for the keypoint in this channel
    lbp = np.zeros((P,), dtype=int)

    # Compute the LBP code for the channel
    # This is done by comparing each pixel in a circular neighborhood around the keypoint to the pixel at the keypoint
    for i in range(P):
        # Compute the coordinates of the circularly symmetric neighbour
        # This is done using the polar coordinates of the neighbour
        xi = x + R * np.cos(2 * np.pi * i / P)
        yi = y - R * np.sin(2 * np.pi * i / P)

        # Perform bilinear interpolation to get the pixel value
        # This is done because the coordinates of the neighbour are likely to be non-integer
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = int(np.ceil(xi)), int(np.ceil(yi))

        # Ensure x0, x1, y0, and y1 are within the valid range of indices for the image array
        # This is done to avoid accessing out-of-bounds elements of the array
        x0 = max(0, min(x0, image.shape[1] - 1))
        x1 = max(0, min(x1, image.shape[1] - 1))
        y0 = max(0, min(y0, image.shape[0] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))

        # Compute the interpolated pixel value
        pixel = (image[y0, x0] * (x1 - xi) + image[y0, x1] * (xi - x0)) * (y1 - yi) + \
                (image[y1, x0] * (x1 - xi) + image[y1, x1] * (xi - x0)) * (yi - y0)

        # Update the LBP code
        # This is done by comparing the interpolated pixel value to the pixel value at the keypoint
        lbp[i] = int(pixel > image[y, x])

    # Convert the LBP code to a single number and return it as a one-dimensional array
    # This is done to make the LBP descriptor easier to work with
    lbp = np.array([np.dot(2**np.arange(P), lbp)])

    return lbp

def extract_HOG(image, keypoint, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Extracts the Histogram of Oriented Gradients (HOG) descriptor from an image at a given keypoint.

    Parameters:
    - image: A 2D numpy ndarray representing the image.
    - keypoint: A tuple representing the coordinates of the keypoint.
    - cell_size: A tuple representing the size of the cells.
    - block_size: A tuple representing the size of the blocks.
    - nbins: The number of orientation bins.

    Returns:
    - hog: A one-dimensional numpy ndarray representing the HOG descriptor.
    """

    # Get the coordinates of the keypoint
    x, y = keypoint

    # Define the region of interest (ROI) around the keypoint
    # The ROI is defined as a square region centered at the keypoint
    # The size of the ROI is determined by the cell size and the block size
    # This is done to ensure that the HOG descriptor is computed over a consistent region around the keypoint
    x_start = max(0, x - cell_size[1] * block_size[1] // 2)
    x_end = min(image.shape[1], x + cell_size[1] * block_size[1] // 2)
    y_start = max(0, y - cell_size[0] * block_size[0] // 2)
    y_end = min(image.shape[0], y + cell_size[0] * block_size[0] // 2)

    roi = image[y_start:y_end, x_start:x_end]

    # Compute the gradients in the x and y directions
    # The gradients are computed using the Sobel operator
    # This is done to capture the edge information in the ROI
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)

    # Compute the gradient magnitude and orientation
    # This is done to capture the direction and strength of the edges in the ROI
    mag, ang = cv2.cartToPolar(gx, gy)

    # Quantize the orientation to the specified number of bins
    # This is done to reduce the dimensionality of the HOG descriptor
    bins = np.int32(nbins * ang / (2 * np.pi))

    # Initialize the HOG feature vector
    # This will hold the HOG descriptor for the keypoint
    hog = []

    # Compute the HOG features for each cell in the block
    # This is done by computing a histogram of the quantized orientations in each cell
    # The histograms are then normalized and concatenated to form the HOG descriptor
    for i in range(block_size[0]):
        for j in range(block_size[1]):
            # Get the bin values for the current cell
            bin_values = bins[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]

            # Compute the histogram for the current cell
            hist, _ = np.histogram(bin_values, bins=nbins, range=(0, nbins))

            # Normalize the histogram
            # This is done to reduce the effect of lighting variations on the HOG descriptor
            eps = 1e-7  # To avoid division by zero
            hist = hist.astype('float64')  # Convert hist to float64
            hist /= np.sqrt(np.sum(hist**2) + eps**2)

            # Append the histogram to the HOG feature vector
            hog.append(hist)

    # Flatten the HOG feature vector
    # This is done to convert the HOG descriptor to a format that is easier to work with
    hog = np.hstack(hog)

    return hog

# Feature Matching Functions
# ------------------------------------------------

def feature_matching(image1, image2, detector, extractor):
    """
    Matches features between two images using the specified detector and extractor.

    Parameters:
    - image1, image2: The images to match features between.
    - detector: The feature detector to use. Must be 'Moravec' or 'Harris'.
    - extractor: The feature extractor to use. Must be 'LBP' or 'HOG'.

    Returns:
    - matched_pairs: A list of tuple pairs representing the matched features in the two images.
    """

    # Validate the detector and extractor
    # This is done to ensure that the function can handle the specified detector and extractor
    if detector not in ['Moravec', 'Harris']:
        raise ValueError('Invalid detector. Must be "Moravec" or "Harris".')
    if extractor not in ['LBP', 'HOG']:
        raise ValueError('Invalid extractor. Must be "LBP" or "HOG".')

    # Detect keypoints in the images
    # This is done using the specified detector
    # The detector is expected to return a list of keypoints for each image
    keypoints1 = moravec_detector(image1) if detector == 'Moravec' else harris_detector(image1)
    keypoints2 = moravec_detector(image2) if detector == 'Moravec' else harris_detector(image2)

    # Check if any keypoints were detected in both images
    # If not, return an empty list of matches
    # This is done to avoid unnecessary computation and potential errors in the rest of the function
    if not keypoints1 or not keypoints2:
        return []

    # Extract descriptors from the keypoints
    # This is done using the specified extractor
    # The extractor is expected to return a descriptor for each keypoint
    descriptors1 = [extract_LBP(image1, kp) if extractor == 'LBP' else extract_HOG(image1, kp) for kp in keypoints1]
    descriptors2 = [extract_LBP(image2, kp) if extractor == 'LBP' else extract_HOG(image2, kp) for kp in keypoints2]

    # Reshape the descriptors to a 2D array
    # This is done to ensure that the descriptors are in a suitable shape for the matching function
    descriptors1 = np.array(descriptors1).reshape(len(descriptors1), -1)
    descriptors2 = np.array(descriptors2).reshape(len(descriptors2), -1)

    # Match the descriptors between the two images
    # This is done using a function that performs cross-check matching
    # Cross-check matching ensures that each match is consistent in both directions
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Convert the matches to a list of tuple pairs
    # Each pair consists of a matched keypoint in the first image and a matched keypoint in the second image
    # This is done to provide a more convenient format for the matches
    matched_pairs = [(keypoints1[match[0]], keypoints2[match[1]]) for match in matches]

    return matched_pairs

def plot_matches(image1, image2, matches):
    """
    Plots the matched features between two images.

    Parameters:
    - image1, image2: The images to match features between.
    - matches: A list of tuple pairs representing the matched features in the two images.

    Returns:
    - new_image: A new image with the input images and the matched features plotted.
    """

    # Create a new image that can fit both input images
    # This is done to provide a canvas on which to plot the matches
    new_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype="uint8")

    # Place the input images in the new image
    # This is done to provide a visual context for the matches
    new_image[:image1.shape[0], :image1.shape[1]] = image1
    new_image[:image2.shape[0], image1.shape[1]:] = image2

    # Adjust the x-coordinates of the keypoints in the second image for the offset in the x direction
    # This is done because the second image is placed to the right of the first image in the new image
    offset_x = image1.shape[1]

    for match in matches:
        pt1, pt2 = match[0], (match[1][0] + offset_x, match[1][1])

        # Draw a circle at the location of each keypoint
        # This is done to visually indicate the location of the keypoints
        cv2.circle(new_image, pt1, 5, (0, 0, 255), -1)
        cv2.circle(new_image, pt2, 5, (0, 0, 255), -1)
        # Draw a line connecting each pair of matched keypoints
        # This is done to visually indicate the matches
        cv2.line(new_image, pt1, pt2, (0, 0, 255), 1)

    return new_image

def moravec_detector(image, percentile=90):
    """
    Detects corners in an image using the Moravec corner detection algorithm.

    Parameters:
    - image: The image to detect corners in.
    - percentile: The percentile to use for thresholding the Moravec responses.

    Returns:
    - keypoints: A list of tuples representing the detected corners.
    """

    # Convert the image to grayscale if it is a color image
    # This is done because the Moravec algorithm operates on grayscale images
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the list of keypoints
    # This will be populated with the detected corners
    keypoints = []

    # Get the dimensions of the image
    # This is done to facilitate the iteration over the pixels of the image
    height, width = image.shape

    # Define the size of the window to use for the Moravec algorithm
    # This is a parameter of the algorithm and can be adjusted to change the scale of the detected corners
    window_size = 3
    offset = window_size // 2

    # Initialize an array to store the Moravec responses
    # This will be used to store the response for each pixel in the image
    responses = np.zeros((height, width))

    # Compute the Moravec response for each pixel in the image
    # This is done by computing the sum of squared differences (SSD) for each direction and taking the minimum SSD
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Initialize the minimum response value
            # This will be updated with the minimum SSD for each direction
            min_response = float('inf')

            # Compute the SSD for each direction
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the current direction if it is the center pixel

                    # Calculate the shifted window to match the size of the original window
                    # This is done to ensure that the SSD is computed over the same number of pixels for each direction
                    window_orig = image[y-offset:y+offset+1, x-offset:x+offset+1]
                    y_shifted_start = max(0, y + dy - offset)
                    y_shifted_end = min(height, y + dy + offset + 1)
                    x_shifted_start = max(0, x + dx - offset)
                    x_shifted_end = min(width, x + dx + offset + 1)

                    # Ensure that the shifted window has the same size as the original window
                    # This is done to ensure that the SSD is computed over the same number of pixels for each direction
                    window_shift = image[y_shifted_start:y_shifted_end, x_shifted_start:x_shifted_end]
                    if window_shift.shape != window_orig.shape:
                        continue  # Skip the current direction if the sizes don't match

                    # Compute the SSD for the current direction
                    ssd = np.sum((window_orig - window_shift) ** 2)
                    min_response = min(min_response, ssd)

            # Store the minimum response for the current pixel
            responses[y, x] = min_response

    # Compute the threshold for the Moravec responses
    # This is done by taking the specified percentile of the non-zero responses
    # The percentile is a parameter of the algorithm and can be adjusted to change the sensitivity of the corner detection
    threshold = np.percentile(responses[np.nonzero(responses)], percentile)

    # Detect the corners by thresholding the Moravec responses
    # This is done by iterating over the responses and adding the coordinates of the pixels with a response above the threshold to the list of keypoints
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if responses[y, x] > threshold:
                keypoints.append((x, y))

    return keypoints

def harris_detector(image, window_size=3, k=0.04):
    """
    Finds keypoints in an image using the Harris corner detection algorithm.
    
    Parameters:
    - image: The input image as a 2D numpy array.
    - window_size: The size of the window to consider for corner detection.
    - k: Harris detector parameter.
    
    Returns:
    - A list of keypoints, where each keypoint is a tuple (x, y).
    """
    # Convert to grayscale if necessary
    # This is done because the Harris algorithm operates on grayscale images
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute image gradients
    # This is done to capture the intensity changes in the image, which are indicative of edges and corners
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=window_size)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=window_size)
    
    # Compute products of gradients
    # This is done to capture the directionality of the intensity changes
    Ixx = gaussian_filter(Ix**2, sigma=1)
    Ixy = gaussian_filter(Ix*Iy, sigma=1)
    Iyy = gaussian_filter(Iy**2, sigma=1)
    
    # Compute Harris response
    # This is done to measure the corner strength at each pixel
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    R = det - k * trace**2
    
    # Threshold for detecting corners
    # This is done to filter out weak corners and reduce the number of keypoints
    threshold = 0.01 * R.max()
    
    # Find keypoints where response is greater than the threshold
    # This is done to select the pixels that are likely to be corners
    keypoints = np.argwhere(R > threshold)
    # Convert (row, col) to (x, y)
    # This is done to match the convention of representing points as (x, y) pairs
    keypoints = [tuple(reversed(point)) for point in keypoints]

    return keypoints

# Gaussian Filter Function
# ------------------------
    
def gaussian_filter(image, sigma=1):
    """
    Applies Gaussian blur to an image using OpenCV.
    
    Parameters:
    - image: The input image as a 2D numpy array.
    - sigma: Standard deviation for Gaussian kernel.
    
    Returns:
    - The blurred image as a 2D numpy array.
    """
    # Calculate the kernel size based on sigma
    # The kernel size is calculated as 6 times sigma plus 1, which is a common heuristic for choosing the kernel size
    # This ensures that the kernel is large enough to capture the structure of the image at the scale defined by sigma
    ksize = int(6*sigma + 1)
    
    # Ensure that the kernel size is odd
    # This is done because the Gaussian kernel needs to be centered on a pixel, which requires an odd kernel size
    if ksize % 2 == 0:
        ksize += 1

    # Apply the Gaussian blur to the image
    # This is done to reduce noise and detail in the image, which can improve the performance of subsequent image processing steps
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)