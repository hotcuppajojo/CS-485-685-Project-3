# CS 485/685 - Project 3: Object Recognition & Segmentation
**Author:** JoJo Petersky  
**Date:** 2024/05/07  
**Course:** CS 485/685 Spring '24  

## Introduction
This project tackles object recognition and image segmentation techniques using Python and various libraries like OpenCV, NumPy, Matplotlib, and Scikit-Image. The project involves extracting features from images, training an object recognition classifier, testing on new images, and implementing several segmentation algorithms. This README outlines the project's implementation details and structure.

## Project Structure
Contained within a single Python file, `project3.py`, this project encapsulates all the requisite functions, including image loading and display, feature extraction, classifier training and testing, and different segmentation techniques.

## Setup and Execution
Ensure Python 3.x is installed on your system, along with the following packages:
- `numpy`
- `opencv-python` (cv2)
- `matplotlib`
- `sklearn`
- `scikit-image`

Install them via pip:
```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image
```

To run the script, execute:
```bash
python project3.py
```

## Implementation Details
### 1. Load and Display Images
- **`load_img(file_name)`**:  
  Loads an image from a specified path in grayscale using OpenCV's `cv2.imread` function. Grayscale was chosen for simplicity, as the project requires grayscale images exclusively.
- **`display_img(image)`**:  
  Displays an image using Matplotlib. Matplotlib was chosen for its ease of visualization and compatibility with various image formats. This function can handle both grayscale and color images and converts BGR to RGB if needed.

### 2. Object Recognition (50 points)
- **`generate_vocabulary(train_data_file)`**:  
  Generates a visual vocabulary (Bag of Words) by clustering features extracted from training images using OpenCV's SIFT descriptors. Clustering is performed with k-means using a silhouette score to find the optimal cluster count. The Bag of Words (BOW) model was selected for its ability to represent visual features efficiently.

- **`extract_features(image, vocabulary)`**:  
  Extracts features from an image based on the given vocabulary, generating a BOW count vector. The BOW count vector enables quantifying the frequency of visual "words" to represent image features.

- **`train_classifier(train_data_file, vocabulary)`**:  
  Trains an SVM classifier on features extracted from training images using the vocabulary. The SVM with a linear kernel was chosen for its efficiency in handling high-dimensional data and robustness in object classification tasks.

- **`classify_image(classifier, test_img, vocabulary)`**:  
  Classifies a test image using the trained classifier and vocabulary, returning the predicted label. This function demonstrates the classifier's predictive capability by converting test image features to BOW vectors and predicting based on training.

### 3. Image Segmentation (50 points)
- **`threshold_image(image, low_thresh, high_thresh)`**:  
  Implements hysteresis thresholding to create a binary image. This approach was chosen due to its effectiveness in distinguishing edges or objects within grayscale images.

- **`grow_regions(image)`**:  
  Grows regions in a binary image using the watershed algorithm. The watershed algorithm is a suitable choice as it can delineate and group distinct regions effectively.

- **`split_regions(image)`**:  
  Splits labeled regions in an image using the watershed algorithm. This splitting method ensures fine-grained separation of regions within an image.

- **`merge_regions(image)`**:  
  Merges labeled regions in an image using the watershed algorithm, ideal for consolidating small segments into larger meaningful regions.

- **`segment_image(image)`**:  
  Combines thresholding, region growing, splitting, and merging to generate three segmentation maps. This function provides multiple segmentation strategies and highlights the flexibility of combining different techniques.

### 4. Image Segmentation with K-Means (20 points extra credit)
- **`kmeans_segment(image)`**:  
  Segments an image using k-means clustering. The optimal number of clusters is determined using the Elbow method, which identifies the ideal value of k by evaluating distortions. This approach provides flexibility by not assuming a fixed number of segments and ensures better region separation.

## Additional Notes
- **Packages Used**:  
  The project uses OpenCV, NumPy, Matplotlib, scikit-image, and scikit-learn for feature extraction, image display, and segmentation. Each package was selected based on its particular strengths:
  - **OpenCV**: Efficient image processing and feature extraction.
  - **NumPy**: Fast numerical computation.
  - **Matplotlib**: Clear visualization of images and data.
  - **scikit-learn**: Reliable machine learning models and clustering.
  - **scikit-image**: Advanced image processing and segmentation.

- **Test Script**:  
  A test script (`test_script.py`) is provided to verify the correctness of the implemented functions.

- **Image Requirements**:  
  The project assumes all input images are grayscale unless explicitly stated.

## License
This project is intended for educational use and is the intellectual property of JoJo Petersky and the instructors of CS 485/685.