from PIL import Image
import numpy as np

# Color based on slope and elevation
# Could do it seperate, but probably best to do it in the loop we already have
# we could have a main loop where get call is_boundary_point and get_color_from_slope or something

fp : str = "./heightmaps"
step : int = 10
max_pixel_value : int = 100
render_scale : float = 0.5

line_color : tuple[int, int, int] = (31, 16, 0)
start_color : tuple[int, int, int] = (43, 23, 0)
end_color  : tuple[int, int, int] = (211, 197, 182)

heightmap = Image.open(fp).convert("L")
heightmap = heightmap.resize((int(heightmap.width * render_scale), int(heightmap.height * render_scale)), Image.BILINEAR)

width, height = heightmap.width, heightmap.height

def interpolate(a : float, b : float, t : float):
    return a * (1.0 - t) + (b * t)

def color_ramp(a : tuple[int, int, int], b : tuple[int, int, int], t : float) -> tuple[int, int, int]:
    return (
        int(interpolate(a[0], b[0], t)),
        int(interpolate(a[1], b[1], t)),
        int(interpolate(a[2], b[2], t)),
    )

def get_point_in_bounds(img : Image.Image, x : int, y : int) -> int:
    x = max(0, min(x, img.width-1)) 
    y = max(0, min(y, img.height-1))
    return img.getpixel((x, y))

def get_neighbors(img : Image.Image, x : int, y : int) -> list[int]:
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

def is_boundary_point(img : Image.Image, x : int, y : int) -> bool:
        pix = get_point_in_bounds(img, x, y)
        neighbors = get_neighbors(img, x, y)
        return any([pix < neighbor for neighbor in neighbors])

def get_slope(img : Image.Image, x : int, y : int) -> int:
    up    = get_point_in_bounds(img, x  , y+1)
    down  = get_point_in_bounds(img, x  , y-1)
    left  = get_point_in_bounds(img, x+1, y  )
    right = get_point_in_bounds(img, x-1, y  )

    vertical   = up - down
    horizontal = left - right

    return vertical

def march_image(output : Image.Image, heightmap : Image.Image, stepped_heightmap : Image.Image):
    boundary_points : list[tuple[int, int]] = []

    for x in range(width):
        for y in range(height):
            #slope = get_slope(heightmap, x, y) * 10
            #color = (slope,slope,slope) #step_color(slope) #+ get_point_in_bounds(heightmap, x, y)) # 
            #output.putpixel((x,y), color)

            color = color_ramp(start_color, end_color, get_point_in_bounds(stepped_heightmap, x, y)/max_pixel_value)

            output.putpixel((x,y), color)

            if is_boundary_point(stepped_heightmap, x, y):
                boundary_points.append((x, y))

    for b in boundary_points:
        output.putpixel(b, line_color)

def step_image(img : Image.Image) -> Image.Image:
    arr = np.array(img)

    nums = list(range(0, max_pixel_value, step))
    arr = np.minimum(arr, nums[-1])

    for idx in range(len(nums)):
        i = max(0,idx-1)
        arr = np.where(np.logical_and(arr <= nums[idx], arr > nums[i]), nums[idx], arr)

    return Image.fromarray(arr)


print("Creating Stepped Image...")
stepped_heightmap = step_image(heightmap)

print("Marching Image...")
output = Image.new("RGB", (width, height))
march_image(output, heightmap, stepped_heightmap)

output.show()