import numba
import math
import numpy
import random
from PIL import Image
import scipy.ndimage as spi

WIDTH, HEIGHT = 1024, 1024
ITERATIONS = 5000
BOUND_EXPANSION_RATE = 0.04

@numba.njit()
def is_a_neighbor_filled(tex : numpy.ndarray, coord : tuple[int, int]):
    s = 0

    f = lambda c : tex[clamp(c[0], 0, tex.shape[0]-1), clamp(c[1], 0, tex.shape[1]-1)]

    s += f((coord[0] + 1, coord[1] + 1))
    s += f((coord[0] + 0, coord[1] + 1))
    s += f((coord[0] - 1, coord[1] + 1))

    s += f((coord[0] + 1, coord[1] + 0))
    s += f((coord[0] - 1, coord[1] + 0))

    s += f((coord[0] + 1, coord[1] - 1))
    s += f((coord[0] + 0, coord[1] - 1))
    s += f((coord[0] - 1, coord[1] - 1))

    return s > 0

@numba.njit()
def clamp(t : float, minimum : float, maximum : float) -> float:
    return min(max(t, minimum), maximum)

@numba.njit()
def get_random_direction() -> tuple[int, int]:
    a = random.randint(0, 3)
    if a == 0:
        return (1, 0)
    elif a == 1:
        return (0, 1)
    elif a == 2:
        return (-1, 0)
    else:
        return (0, -1)

@numba.njit()
def generate_fractal() -> numpy.ndarray:
    #for x in range(-10, 10):
    #    for y in range(-10, 10):
            #if math.sqrt((int(WIDTH/2) + x + -(int(WIDTH)))** + (int(HEIGHT/2) + y + -(int(HEIGHT)))**2) < 10:
            #    tex[clamp(int(), 0, WIDTH-1), clamp(c[1], 0, HEIGHT-1)

    tex = numpy.zeros((WIDTH, HEIGHT))

    start_coord = (int(WIDTH/2), int(HEIGHT/2))
    tex[start_coord] = 255

    # TODO: Adjust upper bounds independantly for each axis


    upper_bound_x = int(WIDTH/2) + 1
    lower_bound_x = int(WIDTH/2) - 1

    upper_bound_y = int(HEIGHT/2) + 1
    lower_bound_y = int(HEIGHT/2) - 1

    for i in range(ITERATIONS):
        upper_bound_x_i = int(upper_bound_x)
        lower_bound_x_i = int(lower_bound_x)
        upper_bound_y_i = int(upper_bound_y)
        lower_bound_y_i = int(lower_bound_y)

        pixel_coord = (random.randint(lower_bound_x_i, upper_bound_x_i), random.randint(lower_bound_y_i, upper_bound_y_i))
        while not is_a_neighbor_filled(tex, pixel_coord):
            direction = get_random_direction()
            pixel_coord = (
                clamp(pixel_coord[0] + direction[0], lower_bound_x_i, upper_bound_x_i),
                clamp(pixel_coord[1] + direction[1], lower_bound_y_i, upper_bound_y_i)
            )

        upper_bound_y += BOUND_EXPANSION_RATE
        upper_bound_x += BOUND_EXPANSION_RATE
        lower_bound_y -= BOUND_EXPANSION_RATE
        lower_bound_x -= BOUND_EXPANSION_RATE

        upper_bound_y = clamp(upper_bound_y, int(WIDTH/2) + 1, WIDTH)
        upper_bound_x = clamp(upper_bound_x, int(HEIGHT/2) + 1, HEIGHT)
        lower_bound_y = clamp(lower_bound_y, 0, int(WIDTH/2) - 1)
        lower_bound_x = clamp(lower_bound_x, 0, int(WIDTH/2) - 1)

        tex[pixel_coord[0], pixel_coord[1]] = int(sigmoid(i/ITERATIONS) * 255) #int(255 * (1 - (i/ITERATIONS)))

    return tex

def sigmoid(x : float) -> float:
    return 1 / (1 + pow(2.718, -x))

def blur(tex : numpy.ndarray, sigma : float) -> numpy.ndarray:
    # TODO: Add an option to have the value of a pixel affect it's area of effect
    return spi.gaussian_filter(tex, sigma=sigma)

def main():
    heightmap = generate_fractal()
    heightmap = blur(heightmap, 2)
    Image.fromarray(heightmap).show()

if __name__ == "__main__":
    main()
