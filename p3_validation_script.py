import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import sklearn
import project3 as p3

import numpy as np
import project3 as p3

def check_non_empty_output(output, description):
    """Raise an error if the output is empty or None."""
    if output is None or len(output) == 0:
        raise ValueError(f"{description} resulted in an empty output.")
    return True

def check_image_validity(image, description):
    """Ensure that the image is a valid numpy array and not empty."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(f"{description} is not a valid image.")
    return True

# Verify training data file
train_data_file = "train_data.txt"
test_img_file = "test_img.jpg"

# Vocabulary Generation Validation
try:
    vocab = p3.generate_vocabulary(train_data_file)
    check_non_empty_output(vocab, "Vocabulary generation")
    print("Vocabulary generation succeeded.")
except Exception as e:
    raise RuntimeError(f"Error generating vocabulary: {e}")

# Classifier Training Validation
try:
    classifier = p3.train_classifier(train_data_file, vocab)
    print("Classifier training succeeded.")
except Exception as e:
    raise RuntimeError(f"Error training classifier: {e}")

# Test Image Loading and Classification Validation
try:
    test_img = p3.load_img(test_img_file)
    check_image_validity(test_img, "Test image")
    print("Test image loaded successfully.")

    # Perform classification
    out = p3.classify_image(classifier, test_img, vocab)
    if out:
        print(f"Test Image Classification Output: {out}")
    else:
        raise ValueError("Classification output is empty or invalid.")
except IOError as e:
    raise RuntimeError(f"Error loading test image: {e}")
except Exception as e:
    raise RuntimeError(f"Error classifying test image: {e}")

# Image Segmentation Validation
try:
    img = p3.load_img(test_img_file)
    check_image_validity(img, "Image for segmentation")
    print("Segmentation image loaded successfully.")

    # Segment the image
    im1, im2, im3 = p3.segment_image(img)

    # Validate outputs of the segmentation
    for i, im in enumerate([im1, im2, im3], start=1):
        check_image_validity(im, f"Segmentation map {i}")
        print(f"Segmentation map {i} is valid.")

    # Display the segmentation maps
    p3.display_img(im1)
    p3.display_img(im2)
    p3.display_img(im3)
except Exception as e:
    raise RuntimeError(f"Error segmenting or displaying image: {e}")