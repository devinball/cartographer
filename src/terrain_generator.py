import numba
import math
import random
import imageio
import numpy as np
from noise import *
from hydraulic_erosion import erode

numba.config.THREADING_LAYER = 'omp'

width, height = 1024, 1024

max_pixel_value : int = 65525
min_pixel_value : int = 0

mask_upscale : float = 1

frequency = 16

mountain_amplitude = 1
flatlands_amplitude = 1

x_offset, y_offset = random.randint(-1000000, 1000000), random.randint(-1000000, 1000000)

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

        img[y, x] = noise * max_pixel_value

@numba.njit()
def sigmoid(a : float, numerator : float=1) -> float:
    return numerator / (1 + pow(2.718, a))

@numba.njit()
def clamp(t, a, b):
    return min(b, max(a, t))

@numba.njit()
def noise_function(x : float, y : float) -> float:
    x *= frequency
    y *= frequency

    a = fBm(perlin, x, y, 8, 1.5, 0.6) * 0.5 + 0.5

    b = fBm(ridged_perlin, x / 6, y / 6, 4, 1.5, 0.25)
    b = pow(b, 2)
    b = sigmoid(-10 * (b - 0.3), numerator=0.9) + 0.1

    c = normalized_perlin(x / 8, y / 8) ** 3

    noise = a * b * c

    return clamp(noise, 0, 1)

arr = np.zeros((height, width), dtype=np.float32)

generate_noise(arr, noise_function)

imageio.imwrite("./result.png", arr.astype(np.uint16))
