import numpy as np
import numba
from PIL import Image

# Color based on slope and elevation
# Could do it seperate, but probably best to do it in the loop we already have
# we could have a main loop where get call is_boundary_point and get_color_from_slope or something

# So what i have learned is that numba can make python code stupid fast
# like really

# so today i have learned that computers are stupid fast
# like dang

fp : str = "./heightmaps/render(2).png"
step : int = 5
max_pixel_value : int = 255
render_scale : float = 1

line_color : tuple[int, int, int] = (31, 16, 0)
start_color : tuple[int, int, int] = (43, 23, 0)
end_color  : tuple[int, int, int] = (211, 197, 182)

heightmap = Image.open(fp).convert("L")
heightmap = heightmap.resize((int(heightmap.width * render_scale), int(heightmap.height * render_scale)), Image.BILINEAR)

width, height = heightmap.width, heightmap.height

@numba.njit
def interpolate(a : float, b : float, t : float):
    return a * (1.0 - t) + (b * t)

@numba.njit
def color_ramp(a : tuple[int, int, int], b : tuple[int, int, int], t : float) -> tuple[int, int, int]:
    return (
        int(interpolate(a[0], b[0], t)),
        int(interpolate(a[1], b[1], t)),
        int(interpolate(a[2], b[2], t)),
    )

@numba.njit
def get_point_in_bounds(img : np.ndarray, x : int, y : int) -> int:
    img_width = img.shape[0]
    img_height = img.shape[1]
    x = max(0, min(x, img_width-1)) 
    y = max(0, min(y, img_height-1))
    return int(img[x, y])

@numba.njit
def get_neighbors(img : np.ndarray, x : int, y : int) -> list[int]:
    return [
        get_point_in_bounds(img, x-1, y-1),
        get_point_in_bounds(img, x-1, y  ),
        get_point_in_bounds(img, x-1, y+1),

        get_point_in_bounds(img, x  , y-1),
        get_point_in_bounds(img, x  , y+1),
        
        get_point_in_bounds(img, x+1, y-1),
        get_point_in_bounds(img, x+1, y  ),
        get_point_in_bounds(img, x+1, y+1),
    ]

@numba.njit
def any_true(a : list[bool]) -> bool:
    for i in a:
        if i:
            return True
    return False

@numba.njit
def is_boundary_point(img : np.ndarray, x : int, y : int) -> bool:
        pix = get_point_in_bounds(img, x, y)
        neighbors = get_neighbors(img, x, y)
        return any_true([pix < neighbor for neighbor in neighbors])

@numba.njit
def get_slope(img : np.ndarray, x : int, y : int) -> int:
    up    = get_point_in_bounds(img, x  , y+1)
    down  = get_point_in_bounds(img, x  , y-1)
    left  = get_point_in_bounds(img, x+1, y  )
    right = get_point_in_bounds(img, x-1, y  )

    vertical   = up - down
    horizontal = left - right

    return vertical

@numba.njit
def march_image(output : np.ndarray, heightmap : np.ndarray, stepped_heightmap : np.ndarray) -> None:
    boundary_points : list[tuple[int, int]] = []


    # it would make sense that x is in range of width, and y is in range of height
    # it would make sense, wouldn't it
    # i don't understand it
    for x in range(height):
        for y in range(width):
            #slope = get_slope(heightmap, x, y) * 10
            #color = (slope,slope,slope) #step_color(slope) #+ get_point_in_bounds(heightmap, x, y)) # 
            #output.putpixel((x,y), color)

            color = color_ramp(start_color, end_color, get_point_in_bounds(stepped_heightmap, x, y)/max_pixel_value)

            output[x ,y] = color

            if is_boundary_point(stepped_heightmap, x, y):
                boundary_points.append((x, y))

    for b in boundary_points:
        output[b] = line_color

def step_image(img : Image.Image) -> Image.Image:
    arr = np.array(img)

    nums = list(range(0, max_pixel_value, step))
    nums.append(max_pixel_value)

    for idx in range(len(nums)):
        i = max(0,idx-1)
        arr = np.where(np.logical_and(arr <= nums[idx], arr > nums[i]), nums[idx], arr)

    return Image.fromarray(arr)

print("Creating Stepped Image...")
stepped_heightmap = step_image(heightmap)

print("Marching Image...")
output = np.array(Image.new("RGB", (width, height)))

march_image(output, np.array(heightmap), np.array(stepped_heightmap))
Image.fromarray(output).show()
