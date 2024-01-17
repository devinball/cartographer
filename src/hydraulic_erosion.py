import numba
import numpy as np
import random

iterations : int = 1000
raindrop_lifetime : int = 1000
raindrop_capacity : float = 0.5
raindrop_speed : float = 1
erosion_amount : float = 0.01


@numba.njit
def set_point_in_bounds(img : np.ndarray, x : int, y : int, col : float) -> None:
    img_width = img.shape[0]
    img_height = img.shape[1]
    x = max(0, min(x, img_width-1)) 
    y = max(0, min(y, img_height-1))
    img[x, y] = col


@numba.njit
def get_point_in_bounds(img : np.ndarray, x : int, y : int) -> float:
    img_width = img.shape[0]
    img_height = img.shape[1]
    x = max(0, min(x, img_width-1)) 
    y = max(0, min(y, img_height-1))
    return img[x, y]

#TODO: Precompute the directions, store the derivative in a 2d array of vectors
@numba.njit
def get_direction(heightmap : np.ndarray, x : int, y : int) -> tuple[int, int]:
    heights = [
        (get_point_in_bounds(heightmap, x+1, y+1),  1,  1),
        (get_point_in_bounds(heightmap, x  , y+1),  0,  1),
        (get_point_in_bounds(heightmap, x-1, y+1), -1,  1),
        (get_point_in_bounds(heightmap, x+1, y  ),  1,  0),
        (get_point_in_bounds(heightmap, x-1, y  ), -1,  0),
        (get_point_in_bounds(heightmap, x+1, y-1),  1, -1),
        (get_point_in_bounds(heightmap, x  , y-1),  0, -1),
        (get_point_in_bounds(heightmap, x-1, y-1), -1, -1),
    ]

    smallest = heights[0]
    for i in heights:
        if i[0] < smallest[0]:
            smallest = i

    delta : float = abs(get_point_in_bounds(heightmap, x, y) - smallest[0])

    return (float(smallest[1]), float(smallest[2]))


@numba.njit
def erode(heightmap : np.ndarray):
    width : int = heightmap.shape[0]
    height : int = heightmap.shape[1]

    for iteration in range(iterations):
        downhill_path : list[tuple[int, int]] = []

        pos_x : int = random.randint(0, width)
        pos_y : int = random.randint(0, height)

        for lifetime in range(raindrop_lifetime):
            start_height = get_point_in_bounds(heightmap, int(pos_x), int(pos_y))
            downhill_path.append((pos_x,pos_y))

            dir_x, dir_y = get_direction(heightmap, int(pos_x), int(pos_y))

            pos_x += dir_x
            pos_y += dir_y

            end_height = get_point_in_bounds(heightmap, int(pos_x), int(pos_y))
            delta = abs(start_height - end_height)
            set_point_in_bounds(heightmap, int(pos_x), int(pos_y), end_height - min(delta, erosion_amount))

"""
        width = heightmap.shape[0]
        height = heightmap.shape[1]

        pos_x : int = random.randint(0, width)
        pos_y : int = random.randint(0, height)

        vel_x : float = 0
        vel_y : float = 0

        for lifetime in range(raindrop_lifetime):
            start_height = get_point_in_bounds(heightmap, int(pos_x), int(pos_y))
            dir_x, dir_y = get_direction(heightmap, int(pos_x), int(pos_y))

            delta = abs(start_height - get_point_in_bounds(heightmap, int(pos_x), int(pos_y)))

            print(delta)

            pos_x += dir_x
            pos_y += dir_y

            a = min(float(get_point_in_bounds(heightmap, int(pos_x), int(pos_y))) - erosion_amount, delta)

            set_point_in_bounds(heightmap, int(pos_x), int(pos_y), a)
"""