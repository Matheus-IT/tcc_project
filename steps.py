import cv2 as cv
import numpy as np
from utils.filters import opening_filter, closing_filter, get_circular_kernel
from PIL import Image
from utils.presentation import display_side_by_side


def segment_breast_tissue(image, original_image):
    # normalize to 8 bits
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    image = cv.GaussianBlur(image, (5, 5), 0)

    # Convert the new image to binary
    ret, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

    # get largest contour
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    big_contour = max(contours, key=cv.contourArea)

    # draw largest contour as white filled on black background as mask
    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.drawContours(mask, [big_contour], 0, 255, cv.FILLED)

    image = cv.bitwise_and(original_image, original_image, mask=mask)

    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    return image


def enhance_contrast(image):
    return cv.equalizeHist(image)


def apply_global_threshold(image):
    ret, image = cv.threshold(image, 140, 255, cv.THRESH_BINARY)
    return image


def get_roi_from_mask(mask):
    # Perform a morphological opening filter to remove false positives
    mask = opening_filter(mask, iter=1, kernel_size=3)
    mask = closing_filter(mask, iter=2, kernel_size=4)
    return mask


def detect_contours_of_artifacts(original, roi):
    # detect contours
    roi = cv.Canny(roi, 100, 200)
    # increase thickness
    kernel = get_circular_kernel(8)
    roi = cv.dilate(roi, kernel, iterations=1)

    # Find contours in the Canny image
    contours, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Function to calculate the distance between two points
    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Grouping circles based on proximity
    grouped_circles = []
    min_distance_to_group = 300  # Minimum distance to group circles (adjust as needed)

    for contour in contours:
        (x, y), radius = cv.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Check if the circle is close to any existing group
        found_group = False
        for group in grouped_circles:
            for circle in group:
                if distance(center, circle) < min_distance_to_group:
                    group.append(center)
                    found_group = True
                    break

        # If not close to any existing group, start a new group
        if not found_group:
            grouped_circles.append([center])

    # Draw rectangles around the groupings
    for group in grouped_circles:
        if len(group) >= 3:  # Minimum number of circles to form a grouping
            xs, ys = zip(*group)

            padding_top_left = 60
            padding_right_bottom = 100
            x, y, w, h = (
                min(xs) - padding_top_left,
                min(ys) - padding_top_left,
                max(xs) - min(xs) + padding_right_bottom,
                max(ys) - min(ys) + padding_right_bottom,
            )
            cv.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 255), 15)

    return roi


def paint_fragments_in_red(img):
    roi = img.copy()
    roi = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
    # Set the red channel values to 255
    blue_channel = roi[:, :, 0]  # Extract the blue channel (channel index 0)
    green_channel = roi[:, :, 1]  # Extract the green channel (channel index 1)
    roi[0] = 255  # set red channel to maximum
    roi[green_channel > 0, 1] = 0  # Set green channel values to 0
    roi[blue_channel > 0, 2] = 0  # Set blue channel values to 0
    return roi


def mark_roi_in_original_image(original, roi):
    # Mark roi in original image
    original = cv.normalize(original, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    original = cv.cvtColor(original, cv.COLOR_GRAY2BGR)
    modified = cv.bitwise_or(original, roi)
    return modified


def apply_contrast_stretching(image, low_percentile=98.5, high_percentile=100):
    # Calculate the low and high intensity thresholds
    low_threshold, high_threshold = np.percentile(
        image, (low_percentile, high_percentile)
    )

    # Clip pixel intensities to the specified thresholds
    stretched_image = np.clip(image, low_threshold, high_threshold)

    # Scale the pixel intensities to the full 8-bit range (0 to 255)
    stretched_image = (
        255 * (stretched_image - low_threshold) / (high_threshold - low_threshold)
    )

    # Convert the array to 8-bit unsigned integer (uint8) type
    stretched_image = stretched_image.astype(np.uint8)

    return stretched_image
