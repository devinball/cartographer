import numba
import numpy as np
from PIL import Image
from noise import perlin, fBm
from hydraulic_erosion import erode

width, height = 1024, 1024

frequency = 2

img = Image.new("L", (width, height))

@numba.njit
def get_noise(img : np.ndarray):
    for x in range(width):
        for y in range(height):
            noise = fBm(perlin, (x / width) * frequency, (y / height) * frequency, 8, 2, 0.5)
            img[x, y] = noise * 255


arr = np.array(img, dtype=np.float32)

get_noise(arr)
#Image.fromarray(arr).show()

erode(arr)
Image.fromarray(arr).show()
