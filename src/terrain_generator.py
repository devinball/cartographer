import numba
import numpy as np
from PIL import Image
from noise import perlin

width, height = 1024, 1024

frequency = 10

#fBm(perlin, x, y, 4, 2.0, 0.5)


img = Image.new("L", (width, height))

@numba.njit
def a(img : np.ndarray):
    for x in range(width):
        for y in range(height):
            img[x, y] = int(perlin((x / width) * frequency, (y / height) * frequency) * 255)

arr = np.array(img)

a(arr)

Image.fromarray(arr).show()
