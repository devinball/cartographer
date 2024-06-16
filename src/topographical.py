import numpy as np
import numba
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

fp : str = "heightmaps/maps/terrain.png"

step : int = 5
max_pixel_value : int = 255
min_pixel_value : int = 0
render_scale : float = 3

line_color : tuple[int, int, int] = (0, 0, 0)
start_color : tuple[int, int, int] = (43, 23, 0)
end_color  : tuple[int, int, int] = (211, 197, 182)

contour_line_width : int = 0 # TODO: Implement contour line width
index_line_width : int = 1
index_line_distance : int = 5

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
def set_point_in_bounds(img : np.ndarray, x : int, y : int, col : tuple[int, int, int]) -> None:
    img_width = img.shape[0]
    img_height = img.shape[1]
    x = max(0, min(x, img_width-1)) 
    y = max(0, min(y, img_height-1))
    img[x, y] = col

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
    # for some reason when it's not cast to a float it ends up with very wrong values later
    up_left    = float(get_point_in_bounds(img, x-1, y-1))
    up_right   = float(get_point_in_bounds(img, x+1, y+1))
    down_left  = float(get_point_in_bounds(img, x-1, y+1))
    down_right = float(get_point_in_bounds(img, x+1, y+1))

    vertical = (up_left - down_left) + (up_right - down_right)
    horizontal = (up_right - up_left) + (down_right - down_left)

    result = int(vertical + horizontal)

    return result

@numba.njit
def draw_boundaries(img : np.ndarray, boundary_points : list[tuple[int, int, int]]) -> None:
    for b in boundary_points:
        if b[2] % index_line_distance == 0:
            for x in range(-index_line_width, index_line_width):
                for y in range(-index_line_width, index_line_width):
                    set_point_in_bounds(img, b[0] + x, b[1] + y, line_color)
        else:
            img[(b[0], b[1])] = line_color

@numba.njit()
def march_image(output : np.ndarray, heightmap : np.ndarray, stepped_heightmap : np.ndarray):
    # list of x, y, value
    boundary_points : list[tuple[int, int, int]] = []

    for x in range(height):
        for y in range(width):
            color = color_ramp(start_color, end_color, get_point_in_bounds(heightmap, x, y)/max_pixel_value)

            #slope = abs(get_slope(heightmap, x, y))s
            #color = color_ramp(start_color, end_color, slope)

            output[x ,y] = color
        
            if is_boundary_point(stepped_heightmap, x, y):
                boundary_points.append((x, y, get_point_in_bounds(stepped_heightmap, x, y)))

    draw_boundaries(output, boundary_points)

def step_heightmap(heightmap : np.ndarray) -> np.ndarray:
    arr = heightmap.copy()

    nums = list(range(min_pixel_value, max_pixel_value, step))
    nums.append(max_pixel_value)

    for idx in range(len(nums)):
        i = max(0,idx-1)
        arr = np.where(np.logical_and(arr <= nums[idx], arr > nums[i]), idx, arr)

    return arr

# Loads the image, converts it to grayscale, and scales it
print("Loading Heightmap...")
heightmap_img = Image.open(fp).convert("L")
heightmap_img = heightmap_img.resize((int(heightmap_img.width * render_scale), int(heightmap_img.height * render_scale)), Image.BILINEAR)

width, height = heightmap_img.width, heightmap_img.height

heightmap = np.array(heightmap_img)
max_pixel_value = heightmap.max()
min_pixel_value = heightmap.min()

# Steps the image into distinct layers
print("Creating Stepped Image...")
stepped_heightmap = step_heightmap(heightmap)

# Iterated over the image to find the boundaries between layers
print("Finding Boundary Points...")
output = np.array(Image.new("RGB", (width, height)))
march_image(output, heightmap, stepped_heightmap)

Image.fromarray(output).show()
