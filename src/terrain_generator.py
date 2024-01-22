import numba
import math
import numpy as np
from PIL import Image
from noise import *
from hydraulic_erosion import erode

numba.config.THREADING_LAYER = 'omp'

#width, height = 1024, 1024

max_pixel_value : int = 255
min_pixel_value : int = 0

mask_upscale : float = 1

mountain_frequency = 16
flatlands_frequency = 8

mountain_amplitude = 1
flatlands_amplitude = 1

@numba.njit(parallel=True)
def generate_noise(img : np.ndarray, noise_function) -> np.ndarray:
    """
    Numba won't let me create an array inside this function
    so instead an array has to be passed in and copied
    """

    output = img.copy()

    length = width * height

    for i in numba.prange(length):
        x = i % width
        y = math.floor(i / width)

        a = max(width, height)

        sample_x = (x / a)
        sample_y = (y / a)

        noise = noise_function(sample_x, sample_y)

        output[y, x] = noise

    return output

@numba.njit
def ridge_terrain(x : float, y : float) -> float:
    return mountain_amplitude * ((1 - abs(fBm(perlin, x * mountain_frequency, y * mountain_frequency, 4, 2, 0.5))) ** 2)

@numba.njit
def hilly_terrain(x : float, y: float) -> float:
    return flatlands_amplitude * (fBm(normalized_perlin, x * flatlands_frequency, y * flatlands_frequency, 4, 2, 0.5))

print("Loading Mountain Mask")
mountain_mask_img = Image.open("./terrain_masks/mountain_mask.png")
mountain_mask_img = mountain_mask_img.convert("L")
mountain_mask_img = mountain_mask_img.resize((mountain_mask_img.size[0] * mask_upscale, mountain_mask_img.size[1] * mask_upscale))

mountain_mask = np.array(mountain_mask_img)
mountain_mask = mountain_mask / max_pixel_value
height, width = mountain_mask.shape

heightmap = np.zeros((height, width))

print(f"Generating map with Width: {width} Height: {height}")

#print("Generating Hilly Noise")
#hilly_noise = generate_noise(np.zeros((height, width), dtype=np.float32), hilly_terrain)

#heightmap += hilly_noise * mountain_mask

print("Generating Mountain Noise")
ridge_noise = generate_noise(np.zeros((height, width), dtype=np.float32), ridge_terrain)

heightmap += ridge_noise * mountain_mask**1.5

# Scale height map to between 0 and 1
heightmap = heightmap / heightmap.max()

print("Done!")
heightmap *= max_pixel_value
Image.fromarray(heightmap).show()

# First we generate a 'world map'
# the world map could be one map or a series of maps, it could also be broken into seperate maps for things like water, terrain, structures, etc
# it would contain important geographic features, like mountains, rivers, lakes etc
# the world map would have a scale - for example one pixel would be 200 feet
# at runtime we then create finer detail only when needed, idealy down to around a foot
# then we don't need to store detail down to 1 foot, only to 200 feet
# we could use simple noise to create the finer detail, using the world map as guidance
# if the scale is 200 feet, then the finer detail probably shouldn't vary in height by more than say 20 feet
# the eventual result is something like google maps, but for a procedual (or partially procedual) world
