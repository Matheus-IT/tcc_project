import pydicom
from utils.presentation import display_side_by_side
from utils.segmentation import get_high_intensity_cluster_kmeans
from utils.filters import opening_filter, closing_filter, high_pass_filter
import cv2 as cv
from utils.image_normalization import normalize
import numpy as np
from utils.elapsed_time import Timer, sample_time_expensive_calls
from PIL import Image
from steps import (
    segment_breast_tissue,
    apply_contrast_stretching,
    paint_fragments_in_red,
    apply_global_threshold,
    get_roi_from_mask,
    detect_contours_of_artifacts,
    mark_roi_in_original_image,
)

def main():
    MAMMOGRAPHY_DATASET_PATH = "/home/matheuscosta/Documents/mammography-dataset/my_subdataset/subdataset_v4/D3-0051/1-1.dcm"
    ds = pydicom.dcmread(MAMMOGRAPHY_DATASET_PATH)
    original = ds.pixel_array

    modified = original.copy()

    modified = segment_breast_tissue(modified, original)

    modified = high_pass_filter(modified)

    modified = apply_global_threshold(modified)

    roi = get_roi_from_mask(modified)

    roi = detect_contours_of_artifacts(original, roi)

    roi = paint_fragments_in_red(roi)

    modified = mark_roi_in_original_image(original, roi)

    display_side_by_side(original, modified)


if __name__ == "__main__":
    main()
