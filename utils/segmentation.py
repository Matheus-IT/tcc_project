import numpy as np
import cv2 as cv


def get_high_intensity_cluster_kmeans(image):
    original_shape = image.shape
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    # Convert to float type only for supporting cv.kmeans
    pixel_vals = image.flatten().astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3  # Choosing number of clusters
    compactness, labels, centers = cv.kmeans(
        pixel_vals, k, None, criteria, 10, cv.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    # Reshape the labels to match the image shape
    labels = labels.flatten()

    # Retrieve the segmented image by assigning each pixel to its corresponding cluster center
    segmented_image = centers[labels]

    # Reshape the segmented image to the original image shape
    segmented_image = segmented_image.reshape(original_shape)

    # Find the cluster with the highest intensity value
    centers_max = np.max(centers, axis=1)
    highest_intensity_cluster_idx = np.argsort(centers_max)[-1]

    # Extract the pixels belonging to the highest intensity cluster
    highest_intensity_pixels = pixel_vals[
        np.where(labels == highest_intensity_cluster_idx)[0]
    ]

    # Create a new image containing just the higher intensity pixels
    img_high_intensity = np.zeros_like(pixel_vals)
    img_high_intensity[
        np.where(labels == highest_intensity_cluster_idx)[0]
    ] = highest_intensity_pixels
    return img_high_intensity.reshape(image.shape)
