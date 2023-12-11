import sys
import argparse
import pydicom
from utils.presentation import display_side_by_side
from utils.segmentation import get_high_intensity_cluster_kmeans
from utils.filters import opening_filter, closing_filter, high_pass_filter
import cv2 as cv
from utils.image_normalization import normalize
import numpy as np
from utils.elapsed_time import time_elapse_measure
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


def detect(args):
    ds = pydicom.dcmread(args.path)

    original = ds.pixel_array

    if ds.PhotometricInterpretation == 'MONOCHROME1':
        original = cv.normalize(original, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        original = np.invert(original)

    modified = original.copy()

    modified = segment_breast_tissue(modified, original)

    modified = high_pass_filter(modified)

    modified = apply_global_threshold(modified)

    roi = get_roi_from_mask(modified)

    roi = detect_contours_of_artifacts(original, roi)
    if not len(roi):
        return False

    roi = paint_fragments_in_red(roi)

    modified = mark_roi_in_original_image(original, roi)

    if args.show == 'y' or args.show == 'yes' or args.show == 's' or args.show == 'sim':
        display_side_by_side(original, modified, scale_factor=0.1)

    return True


def main():
    parser = argparse.ArgumentParser(description='Script para detecção em imagens')
    parser.add_argument('path', metavar='PARAM', type=str, help='Caminho para arquivo DICOM da imagem')
    parser.add_argument('--show', default='n', type=str, help='Indique se a imagem resultante deverá ser mostrada')

    args = parser.parse_args()

    if not args.path:
        raise Exception('É necessário passar o caminho do arquivo DICOM')
    
    result = detect(args)
    print(result)


if __name__ == "__main__":
    main()
