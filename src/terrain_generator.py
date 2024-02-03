import numba
import math
import random
import imageio
import numpy as np
from noise import *
from hydraulic_erosion import erode

numba.config.THREADING_LAYER = 'omp'

width, height = 1024, 1024

max_pixel_value : int = 65535
min_pixel_value : int = 0

mask_upscale : float = 1

frequency = 4
amplitude = 1

x_offset, y_offset = random.randint(-1000000, 1000000), random.randint(-1000000, 1000000)

print(x_offset, y_offset)

@numba.njit(parallel=True)
def generate_noise(img : np.ndarray, noise_function):
    length = width * height

    for i in numba.prange(length):
        x = i % width
        y = math.floor(i / width)

        a = max(width, height)

        sample_x = ((x + x_offset) / a)
        sample_y = ((y + y_offset) / a)

        noise = noise_function(sample_x, sample_y)

        img[y, x] = noise * amplitude

@numba.njit()
def sigmoid(a : float, numerator : float=1) -> float:
    return numerator / (1 + pow(2.718, a))

@numba.njit()
def clamp(t, a, b):
    return min(b, max(a, t))

@numba.njit()
def mountain_noise(x : float, y : float) -> float:
    a = fBm(perlin, x, y, 10, 1.75, 0.6) * 0.5 + 0.5
    a = (10 ** a) / 10

    b = 1 - abs(fBm(perlin, x, y, 4, 1.5, 0.5))
    b = b ** 2
    
    noise = (a * b) ** 0.75

    return noise

#TODO: Replace constants with variables and notate everything
@numba.njit()
def noise_function(x : float, y : float) -> float:
    x *= frequency
    y *= frequency

    return mountain_noise(x, y)


arr = np.zeros((height, width), dtype=np.float32)

generate_noise(arr, noise_function)

arr /= max(arr.flatten())
arr *= max_pixel_value

imageio.imwrite("./result.png", arr.astype(np.uint16))
