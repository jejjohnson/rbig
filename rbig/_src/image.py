import numpy as np


def flatten_image(image):
    H, W, C = image.shape
    return image.reshape(H * W, C)


def unflatten_image(data, height, width):
    H_W, C = data.shape
    return data.reshape(height, width, C)
