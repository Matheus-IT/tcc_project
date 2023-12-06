import numpy as np
from pydicom import FileDataset
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2 as cv


draw_line = lambda: print("=" * 50)


def p_dicom(ds: FileDataset, field: str):
    if "," in field:
        # field is a tag
        try:
            print(f"{field}: {ds[eval(field)].value}")
        except KeyError:
            print(f"{field}: None")
        finally:
            return

    if ds.get(field) is not None:
        print(
            f"{field}: {ds.get(field) if isinstance(ds.get(field), (int, float, str)) else ds.get(field).value}"
        )
    else:
        print(f"{field}: None")


def show_presentation(ds: FileDataset):
    """
    The fields I'm using here I found with ds.dir() this way:
    ds.dir('id') -> [
        'DeidentificationMethod', 'DeidentificationMethodCodeSequence',
        'DetectorID', 'FrameOfReferenceUID', 'Grid', 'PatientID', 'PatientIdentityRemoved',
        'SOPClassUID', 'SOPInstanceUID', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID',
        'WindowWidth',
    ]
    ds.dir('number') -> [
        'AccessionNumber', 'DeviceSerialNumber', 'InstanceNumber', 'SeriesNumber',
    ]
    """
    draw_line()
    p_dicom(ds, "PatientID")
    p_dicom(ds, "StudyID")
    p_dicom(ds, "SOPClassUID")
    p_dicom(ds, "SOPInstanceUID")
    p_dicom(ds, "SeriesInstanceUID")
    p_dicom(ds, "StudyInstanceUID")
    p_dicom(ds, "InstanceNumber")
    p_dicom(ds, "SeriesNumber")
    draw_line()


def compare(ds1: FileDataset, ds2: FileDataset):
    print("Dataset1:")
    show_presentation(ds1)

    print("Dataset2:")
    show_presentation(ds2)


def show_pixel_info(ds: FileDataset):
    draw_line()
    p_dicom(ds, "PhotometricInterpretation")
    p_dicom(ds, "SamplesPerPixel")
    p_dicom(ds, "BitsAllocated")
    p_dicom(ds, "BitsStored")
    p_dicom(ds, "HighBit")
    p_dicom(ds, "PixelRepresentation")
    p_dicom(ds, "NumberOfFrames")
    draw_line()


def show_voi_lut_module(ds: FileDataset):
    draw_line()
    p_dicom(ds, "(0x0028,0x3010)")  # VOILUTSequence
    p_dicom(ds, "(0x0028,0x3002)")  # LUTDescriptor
    p_dicom(ds, "(0x0028,0x3003)")  # LUTExplanation
    p_dicom(ds, "(0x0028,0x3006)")  # LUTData
    p_dicom(ds, "(0x0028,0x1050)")  # WindowCenter
    p_dicom(ds, "(0x0028,0x1051)")  # WindowWidth
    p_dicom(ds, "(0x0028,0x1055)")  # WindowCenter&WidthExplanation
    p_dicom(ds, "(0x0028,0x1056)")  # VOILUTFunction
    draw_line()


def show_file_dataset_info(ds: FileDataset):
    draw_line()
    print("ds.preamble", ds.preamble)
    print("ds.file_meta", ds.file_meta)
    print("ds.fileobj_type", ds.fileobj_type)
    print("ds.is_implicit_VR", ds.is_implicit_VR)
    print("ds.is_little_endian", ds.is_little_endian)
    print("ds.default_element_format", ds.default_element_format)
    print("ds.default_sequence_element_format", ds.default_sequence_element_format)
    print("ds.indent_chars", ds.indent_chars)
    print("ds.is_original_encoding", ds.is_original_encoding)
    draw_line()


def show_pixel_array_info(pixel_array: np.ndarray):
    draw_line()
    print("data", pixel_array.data)
    print("dtype", pixel_array.dtype)
    print("flags", pixel_array.flags)
    print("itemsize", pixel_array.itemsize)
    print("size", pixel_array.size)
    print("ndim", pixel_array.ndim)
    print("shape", pixel_array.shape)
    draw_line()


def compare_image_filter(img: Image, operation: callable, gray=False):
    fig = plt.figure(figsize=(8, 5))
    fig.add_subplot(1, 2, 1)

    if gray:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title("Original")

    fig.add_subplot(1, 2, 2)

    filtered = operation(img)
    if gray:
        plt.imshow(filtered, cmap="gray")
    else:
        plt.imshow(filtered)
    plt.title("Filtered")
    plt.show()
    return filtered


def display_side_by_side(img1_array, img2_array):
    if img1_array.dtype == "uint16":
        img1_array = cv.normalize(img1_array, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    img1 = Image.fromarray(img1_array)

    if img2_array.dtype == "uint16":
        img2_array = cv.normalize(img2_array, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    img2 = Image.fromarray(img2_array)

    new_img = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    new_img.show()
