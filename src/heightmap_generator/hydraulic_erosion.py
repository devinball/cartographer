import numba
import math
import numpy as np
import random

max_pixel_value : int = 255
min_pixel_value : int = 0

iterations : int = 1000
lifetime : int = 30

gradient_epsilon : float = 0.001

erosion_rate : float = 0.9
deposition_rate : float = 0.02
sediment_capacity_factor : float = 1
initial_water : float = 1
evaporation_rate : float = 0.01
min_slope : float = 0.05
inertia : float = 0.1
gravity : float = 20
sediment_factor : float = 1

@numba.njit
def set_point_in_bounds(img : np.ndarray, x : int, y : int, col : float) -> None:
    img_width = img.shape[0]
    img_height = img.shape[1]
    x = max(0, min(x, img_width-1)) 
    y = max(0, min(y, img_height-1))
    # clamp color to within pixel range
    col = min(max_pixel_value, max(min_pixel_value, col))
    img[x, y] = col

@numba.njit
def get_point_in_bounds(img : np.ndarray, x : int, y : int) -> float:
    img_width = img.shape[0]
    img_height = img.shape[1]
    x = max(0, min(x, img_width-1)) 
    y = max(0, min(y, img_height-1))
    return img[x, y]

@numba.njit
def change_map(heightmap, x, y, amount):
    h00 = get_point_in_bounds(heightmap, x, y)
    h10 = get_point_in_bounds(heightmap, x+1, y)
    h01 = get_point_in_bounds(heightmap, x, y+1)
    h11 = get_point_in_bounds(heightmap, x+1, y+1)

    set_point_in_bounds(heightmap, x, y, h00 + amount)
    set_point_in_bounds(heightmap, x+1, y, h10 + amount)
    set_point_in_bounds(heightmap, x, y+1, h01 + amount)
    set_point_in_bounds(heightmap, x+1, y+1, h11 + amount)

@numba.njit
def erode(heightmap : np.ndarray) -> np.ndarray:
    width : int = heightmap.shape[0]
    height : int = heightmap.shape[1]

    for iteration in range(iterations):
        pos_x = random.randint(0, width-1)
        pos_y = random.randint(0, height-1)

        ipos_x = int(pos_x)
        ipos_y = int(pos_y)

        flow_speed = 0
        water = initial_water
        carried_sediment = 0

        h00 = get_point_in_bounds(heightmap, ipos_x, ipos_y)
        h10 = get_point_in_bounds(heightmap, ipos_x+1, ipos_y)
        h01 = get_point_in_bounds(heightmap, ipos_x, ipos_y+1)
        h11 = get_point_in_bounds(heightmap, ipos_x+1, ipos_y+1)

        dx = 0
        dy = 0

        for step in range(lifetime):
            current_height = h00

            # gradient calculation from http://ranmantaru.com/blog/2011/10/08/water-erosion-on-heightmap-terrain/
            gradient_x = h00 + h01 - h10 - h11
            gradient_y = h00 + h10 - h01 - h11            

            dx = (dx - gradient_x) * inertia + gradient_x
            dy = (dy - gradient_y) * inertia + gradient_y

            gradient_length = math.sqrt(dx*dx+dy*dy)

            if gradient_length < gradient_epsilon:
                gradient_x = random.random()
                gradient_y = random.random()
            else:
                dx /= gradient_length
                dy /= gradient_length

            next_pos_x = pos_x + dx
            next_pos_y = pos_y + dy

            next_ipos_x = int(next_pos_x)
            next_ipos_y = int(next_pos_y)            

            next_h00 = get_point_in_bounds(heightmap, next_ipos_x, next_ipos_y)
            next_h10 = get_point_in_bounds(heightmap, next_ipos_x+1, next_ipos_y)
            next_h01 = get_point_in_bounds(heightmap, next_ipos_x, next_ipos_y+1)
            next_h11 = get_point_in_bounds(heightmap, next_ipos_x+1, next_ipos_y+1)

            fraction_next_pos_x = next_pos_x - next_ipos_x
            fraction_next_pos_y = next_pos_y - next_ipos_y

            # this is bilinear interpolation - also from http://ranmantaru.com/blog/2011/10/08/water-erosion-on-heightmap-terrain/
            next_height = (next_h00 * (1 - fraction_next_pos_x) + next_h10) * (1 - fraction_next_pos_y) + (next_h01 * (1 - fraction_next_pos_x) + next_h11 * fraction_next_pos_x) * fraction_next_pos_y
            #next_height = (next_h00 * (1 - fraction_next_pos_x) * (1 - fraction_next_pos_y) + next_h10 * fraction_next_pos_x * (1 - fraction_next_pos_y) + next_h01 * (1 - fraction_next_pos_x) * fraction_next_pos_y + next_h11 * fraction_next_pos_x * fraction_next_pos_y)

            delta_height = 0

            if next_height >= current_height:
                delta_height = next_height - current_height

                # the next height is greater than the current height, so we will get stuck in a pit
                if carried_sediment > delta_height:
                    # we have enough sediment to get out, so we drop enough to get out and keep on going
                    carried_sediment -= delta_height
                    change_map(heightmap, ipos_x, ipos_y, delta_height)
                else:
                    # we don't have enough sediment to get out, so we drop what we have and exit this iteration
                    change_map(heightmap, ipos_x, ipos_y, carried_sediment)
                    break

                next_h00 = get_point_in_bounds(heightmap, next_ipos_x, next_ipos_y)
                next_h10 = get_point_in_bounds(heightmap, next_ipos_x+1, next_ipos_y)
                next_h01 = get_point_in_bounds(heightmap, next_ipos_x, next_ipos_y+1)
                next_h11 = get_point_in_bounds(heightmap, next_ipos_x+1, next_ipos_y+1)

                next_height = (next_h00 * (1 - fraction_next_pos_x) + next_h10) * (1 - fraction_next_pos_y) + (next_h01 * (1 - fraction_next_pos_x) + next_h11 * fraction_next_pos_x) * fraction_next_pos_y

            delta_height = current_height - next_height
            slope = delta_height

            # WE SHOULD NOT BE GETTING NEGATIVES HERE FOR DELTA HEIGHT

            sediment_capacity = max(slope, min_slope) * flow_speed * water * sediment_capacity_factor

            print(delta_height)

            if carried_sediment > sediment_capacity:
                deposition = deposition_rate * (carried_sediment - sediment_capacity)

                change_map(heightmap, ipos_x, ipos_y, deposition)

                carried_sediment -= deposition * sediment_factor
            else:
                erosion = erosion_rate * (sediment_capacity - carried_sediment)

                # don't remove more than the difference in height (minus a little)
                erosion = min(erosion, delta_height * 0.99) * water

                #print(erosion, delta_height, ipos_x, ipos_y)

                change_map(heightmap, ipos_x, ipos_y, -erosion)

                # make sure to update the delta height after erosion
                delta_height -= erosion

                # add the eroded soil to the droplet
                carried_sediment += erosion * sediment_factor

            flow_speed += delta_height #math.sqrt(flow_speed * flow_speed + gravity * delta_height)
            water *= (1 - evaporation_rate)

            if water < 0.01:
                break

            fraction_pos_x = fraction_next_pos_x
            fraction_pos_y = fraction_next_pos_y

            ipos_x = next_ipos_x
            ipos_y = next_ipos_y

            h00 = next_h00
            h10 = next_h10
            h01 = next_h01
            h11 = next_h11

    return heightmap
