# JoJo Petersky
# CS 485/685 Spring '24 Project3
# 2024/04/07
# project3.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import match_descriptors, peak_local_max
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.svm import SVC
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from sklearn.metrics import silhouette_score

# Image Loading & Displaying Functions
# ------------------------------------    

def load_img(file_name):
    """
    Loads an image from a file in grayscale.

    Parameters:
    - file_name: The path to the image file.

    Returns:
    - The loaded grayscale image as a numpy array.

    Raises:
    - IOError: If the image cannot be loaded.
    """
    try:
        # Set the color mode to grayscale
        color_mode = cv2.IMREAD_GRAYSCALE

        # Attempt to load the image
        img = cv2.imread(file_name, color_mode)

        # Check if the image was loaded successfully
        if img is None:
            raise ValueError(f"The file {file_name} cannot be loaded as an image.")
        
        return img
    except Exception as e:
        # Handle other potential exceptions
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

# Object Recognition Functions
# ----------------------------

def generate_vocabulary(train_data_file):
    """
    Generates a visual vocabulary (Bag of Words) by clustering image features.

    Parameters:
    - train_data_file: The path to the file containing paths to the training images.

    Returns:
    - An array containing the cluster centers representing the vocabulary.
    """
    all_descriptors = []
    max_clusters=20 # The maximum number of clusters to evaluate.

    with open(train_data_file, 'r') as file:
        for line in file:
            img_path, label = line.strip().split(' ')  # Split the line into the image path and the label
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            sift = cv2.SIFT_create()
            _, descriptors = sift.detectAndCompute(img, None)

            if descriptors is not None:
                all_descriptors.extend(descriptors)

    all_descriptors = np.array(all_descriptors)

    # Ensure there are enough descriptors
    if all_descriptors.shape[0] == 0:
        raise ValueError("No valid descriptors found in training images.")

    # Determine the optimal number of clusters using the silhouette score
    best_score = -1  # Silhouette scores range between -1 and 1
    optimal_clusters = 2  # Default to at least two clusters

    K_range = range(2, max_clusters + 1)
    for k in K_range:
        # Initialize centroids using KMeans++ initialization method
        distances = np.full(all_descriptors.shape[0], np.inf)
        centroids = np.empty((k, all_descriptors.shape[1]))
        for i in range(k):
            if i == 0:
                centroids[i] = all_descriptors[np.random.randint(all_descriptors.shape[0])]
            else:
                distances = np.minimum(distances, np.sum((all_descriptors - centroids[i - 1])**2, axis=1))
                probabilities = distances / np.sum(distances)
                centroids[i] = all_descriptors[np.random.choice(all_descriptors.shape[0], p=probabilities)]

        # Optimize centroids iteratively
        max_iters = 10 * k
        for _ in range(max_iters):
            distances = np.sqrt(((all_descriptors - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([all_descriptors[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        # Calculate the silhouette score
        score = silhouette_score(all_descriptors, labels)
        if score > best_score:
            best_score = score
            optimal_clusters = k

    # Final KMeans clustering with the optimal number of clusters
    distances = np.full(all_descriptors.shape[0], np.inf)
    centroids = np.empty((optimal_clusters, all_descriptors.shape[1]))
    for i in range(optimal_clusters):
        if i == 0:
            centroids[i] = all_descriptors[np.random.randint(all_descriptors.shape[0])]
        else:
            distances = np.minimum(distances, np.sum((all_descriptors - centroids[i - 1])**2, axis=1))
            probabilities = distances / np.sum(distances)
            centroids[i] = all_descriptors[np.random.choice(all_descriptors.shape[0], p=probabilities)]

    max_iters = 10 * optimal_clusters
    for _ in range(max_iters):
        distances = np.sqrt(((all_descriptors - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([all_descriptors[labels == i].mean(axis=0) for i in range(optimal_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids

def extract_features(image, vocabulary):
    """
    Extracts SIFT features from an image and maps them to a Bag of Visual Words (BoVW) vector.

    Parameters:
    - image: The image to extract features from.
    - vocabulary: The BoVW vocabulary (i.e., the cluster centers from K-Means clustering).

    Returns:
    - The BoVW vector representing the image.
    """
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Compute the SIFT descriptors for the image
    _, descriptors = sift.detectAndCompute(image, None)

    # Initialize the BoVW vector as a zero vector
    bow_vector = np.zeros(len(vocabulary))

    # If descriptors were found, map them to the BoVW vector
    if descriptors is not None:
        for descriptor in descriptors:
            # Compute the Euclidean distances between the descriptor and the vocabulary
            distances = distance.cdist([descriptor], vocabulary, 'euclidean')

            # Find the closest cluster in the vocabulary
            closest_cluster = np.argmin(distances)

            # Increment the corresponding element in the BoVW vector
            bow_vector[closest_cluster] += 1

    return bow_vector

def train_classifier(train_data_file, vocabulary):
    """
    Trains a Support Vector Machine (SVM) classifier using a set of training images.

    Parameters:
    - train_data_file: The path to a file containing the paths and labels of the training images.
    - vocabulary: The BoVW vocabulary.

    Returns:
    - The trained SVM classifier.
    """
    # Initialize the list of feature vectors and labels
    feature_vectors = []
    labels = []

    # Open the training data file
    with open(train_data_file, 'r') as file:
        for line in file:
            # Split the line into the image path and the label
            img_path, label = line.strip().split(' ')

            # Load the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Extract the BoVW feature vector from the image
            feature_vector = extract_features(img, vocabulary)

            # Add the feature vector and label to their respective lists
            feature_vectors.append(feature_vector)
            labels.append(label)

    # Create and train the SVM classifier
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(np.array(feature_vectors), labels)

    return classifier

def classify_image(classifier, test_img, vocabulary):
    """
    Classifies an image using a trained SVM classifier and a BoVW vocabulary.

    Parameters:
    - classifier: The trained SVM classifier.
    - test_img: The image to classify.
    - vocabulary: The BoVW vocabulary.

    Returns:
    - The predicted label of the image.
    """
    # Extract the BoVW feature vector from the image
    feature_vector = extract_features(test_img, vocabulary)

    # Reshape the feature vector to a 2D array
    feature_vector = feature_vector.reshape(1, -1)

    # Use the classifier to predict the label of the image
    return classifier.predict(feature_vector)[0]

# Image Segmentation Functions
# ----------------------------

def threshold_image(image, low_thresh, high_thresh):
    """
    Thresholds an image using hysteresis thresholding.

    Parameters:
    - image: A grayscale image.
    - low_thresh: The low threshold for hysteresis.
    - high_thresh: The high threshold for hysteresis.

    Returns:
    - A binary image resulting from the thresholding.
    """
    # Check if the image is grayscale
    if len(image.shape) > 2:
        raise ValueError("The input image should be grayscale.")

    # Initialize the output image
    strong_edges = np.zeros_like(image, dtype=bool)
    weak_edges = np.zeros_like(image, dtype=bool)

    # Apply the high threshold
    strong_edges[image > high_thresh] = True

    # Apply the low threshold
    weak_edges[(image <= high_thresh) & (image > low_thresh)] = True

    # Define the DFS function
    def dfs(i, j):
        stack = [(i, j)]
        while stack:
            i, j = stack.pop()
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if (0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and
                        weak_edges[ni, nj] and not strong_edges[ni, nj]):
                        strong_edges[ni, nj] = True
                        stack.append((ni, nj))

    # Apply the DFS function to every strong edge pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if strong_edges[i, j]:
                dfs(i, j)

    return strong_edges

def grow_regions(image):
    """
    Grows regions in an image using the watershed algorithm.

    Parameters:
    - image: A binary image.

    Returns:
    - An image with the grown regions.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(image)
    # Find local maxima in the distance transform
    coords = peak_local_max(distance, min_distance=20, labels=image)
    # Create a mask of the local maxima
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    # Compute markers from the local maxima
    markers, _ = ndi.label(mask)
    # Apply the watershed algorithm to grow regions
    labels = watershed(-distance, markers, mask=image)
    return labels

def split_regions(image):
    """
    Splits regions in an image using the watershed algorithm.

    Parameters:
    - image: A labeled image.

    Returns:
    - An image with the split regions.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(image)
    # Apply the watershed algorithm to split regions
    labels = watershed(-distance, image, mask=image)
    return labels

def merge_regions(image):
    """
    Merges regions in an image using the watershed algorithm.

    Parameters:
    - image: A labeled image.

    Returns:
    - An image with the merged regions.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(image)
    # Apply the watershed algorithm to merge regions
    labels = watershed(distance, image, mask=image)
    return labels

def segment_image(image):
    """
    Segments an image using thresholding, region growing, splitting and merging.

    Parameters:
    - image: A grayscale image.

    Returns:
    - Three segmented images.
    """
    # Compute the thresholds
    mean = np.mean(image)
    std = np.std(image)
    low_thresh = mean - std
    high_thresh = mean + std

    # Threshold the image
    binary = threshold_image(image, low_thresh, high_thresh)

    # Grow regions in the binary image
    labels1 = grow_regions(binary)

    # Split regions in the binary image
    labels2 = split_regions(binary)

    # Merge regions in the binary image
    labels3 = merge_regions(binary)

    return labels1, labels2, labels3

# K-Means Segmentation Function
# -----------------------------

def kmeans_segment(image):
    """
    Segments an image using K-Means clustering.

    Parameters:
    - image: A color or grayscale image.

    Returns:
    - A segmented image.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        pixels = image.reshape(-1, 1)
        centroid_shape = (1,)
    else:
        # Color image
        pixels = image.reshape(-1, 3)
        centroid_shape = (3,)

    # Determine the optimal number of clusters using the Elbow method
    distortions = []
    K = range(2, 10)  # Check for number of clusters from 2 to 10
    for k in K:
        # Initialize centroids using K-Means++
        distances = np.full(pixels.shape[0], np.inf)
        centroids = np.empty((k,) + centroid_shape)
        for i in range(k):
            if i == 0:
                centroids[i] = pixels[np.random.randint(pixels.shape[0])]
            else:
                distances = np.minimum(distances, np.sum((pixels - centroids[i-1])**2, axis=1))
                probabilities = distances / np.sum(distances)
                centroids[i] = pixels[np.random.choice(pixels.shape[0], p=probabilities)]

        old_distortion = np.inf
        # Determine max_iters based on the size of the image
        max_iters = 10 * k
        for _ in range(max_iters):
            # Assign each pixel to the closest centroid
            distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Compute new centroids as the mean of the pixels in each cluster
            new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])

            # Compute the distortion (sum of squared distances from each point to its centroid)
            distortion = ((pixels - centroids[labels])**2).sum()

            # If the centroids didn't change or the change in distortion is very small, we're done
            if np.all(centroids == new_centroids) or np.abs(old_distortion - distortion) < 1e-5:
                break

            centroids = new_centroids
            old_distortion = distortion

        distortions.append(distortion)

    # The optimal number of clusters is the one that causes a significant decrease in distortion
    elbow_point = np.argmin(np.diff(distortions, 2)) + 1

    # Apply KMeans clustering with the optimal number of clusters
    centroids = pixels[np.random.choice(pixels.shape[0], size=elbow_point, replace=False)]
    max_iters = 10 * elbow_point
    for _ in range(max_iters):
        distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(elbow_point)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Reshape the labels back to the original image shape
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image