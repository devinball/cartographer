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

x_offset, y_offset = random.randint(-1000000, 1000000), random.randint(-1000000, 1000000) #493157, -617065 

print(f"X: {x_offset}, Y: {y_offset}")

@numba.njit(parallel=True)
def generate_noise(img : np.ndarray, noise_function) -> None:
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
    #TODO: Replace constants with variables and notate everything
    a = fBm(perlin, x, y, 10, 1.75, 0.6) * 0.5 + 0.5
    
    # Controls how bumpy the landscape is, higher values make it bumpier
    bumpiness = 10
    a = (bumpiness ** a) / bumpiness

    # Higher values will emphasize the ridges more
    ridge_steepness = 2

    b = 1 - abs(fBm(perlin, x, y, 4, 1.5, 0.5))
    b = b ** ridge_steepness
    
    # Higher values will bring up the peaks and lower the valleys, lower values will make everything closer
    overall_steepness = 0.75

    noise = (a * b) ** overall_steepness

    return noise

@numba.njit()
def plains_noise(x : float, y : float) -> float:
    noise = fBm(normalized_perlin, x, y, 2, 2, 0.5)
    return noise

@numba.njit()
def dune_noise(x : float, y : float) -> float:
    noise = fBm(normalized_perlin, x, y, 2, 2, 0.5)
    return noise

@numba.njit()
def canyon_noise(x : float, y : float) -> float:
    noise = fBm(normalized_perlin, x, y, 2, 2, 0.5)
    return noise

@numba.njit()
def river_noise(x : float, y : float) -> float:
    noise = fBm(normalized_perlin, x, y, 2, 2, 0.5)
    return noise

@numba.njit()
def biome_transition_noise(x : float, y : float, biome_change_rate : float = 10, biome_frequency : float = 1) -> float:
    noise = normalized_perlin(x * biome_frequency, y * biome_frequency)
    # sigmoid is used to push values to the extremes
    # this is useful to control how quickly biomes change

    noise = sigmoid(biome_change_rate * (noise - 0.5))
    return noise

@numba.njit()
def noise_function(x : float, y : float) -> float:
    x *= frequency
    y *= frequency

    #return biome_transition_noise(x, y, biome_change_rate=50, biome_frequency=1/8) * mountain_noise(x, y)

    return mountain_noise(x, y)

def main():
    arr = np.zeros((height, width), dtype=np.float32)
    generate_noise(arr, noise_function)

    print(f"Largest pixel value: {max(arr.flatten())}")

    arr /= max(arr.flatten())
    arr *= max_pixel_value

    imageio.imwrite("./result.png", arr.astype(np.uint16))

if __name__ == "__main__":
    main()
