import cv2 as cv
import numpy as np


def get_circular_kernel(size):
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))


def opening_filter(image, iter=1, kernel_size=3):
    kernel = get_circular_kernel(kernel_size)
    image = cv.erode(image, kernel, iterations=iter)
    return cv.dilate(image, kernel, iterations=iter)


def closing_filter(image, iter=1, kernel_size=3):
    kernel = get_circular_kernel(kernel_size)
    image = cv.dilate(image, kernel, iterations=iter)
    return cv.erode(image, kernel, iterations=iter)


def high_pass_filter(image):
    # Aplica o filtro de Sobel para detecção de bordas
    sobel_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=9)
    sobel_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=9)

    # Calcula o gradiente aproximado da imagem
    gradient_image = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normaliza a imagem para o intervalo [0, 255]
    gradient_image = cv.normalize(
        gradient_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
    )

    crop_image(gradient_image)

    return gradient_image


def crop_image(image):
    height, width = image.shape

    # cropping 0.8% of the image width and height
    width_size = int(width * (0.8 / 100))
    height_size = int(height * (0.8 / 100))

    image[:height_size, :] = 0  # Top border
    image[-height_size:, :] = 0  # Bottom border
    image[:, :width_size] = 0  # Left border
    image[:, -width_size:] = 0  # Right border
    return image
