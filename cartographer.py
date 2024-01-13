# The Balkan Border Builder
# The idea is we ask everyone in the balkans their neighbors are from
# If someone has neighbors from somewhere else than they are, then they live on the border

from PIL import Image
import numpy as np

fp : str = "./render.png"
step : int = 30
render_scale : float = 4

start_color : tuple[int, int, int] = (211, 197, 182)
end_color   : tuple[int, int, int] = (43 , 23 , 0  )
line_color  : tuple[int, int, int] = (31 , 16 , 0  )

heightmap = Image.open(fp).convert(mode="L")
heightmap = heightmap.resize(size=(int(heightmap.width * render_scale), int(heightmap.height * render_scale)), resample=Image.BILINEAR)

width, height = heightmap.width, heightmap.height
output = Image.new(mode="L", size=(width, height))

def get_pixel_in_range(img : Image.Image, x : int, y : int) -> int:
    if x > 0 and x < img.width and y > 0 and y < img.height:
        return img.getpixel((x,y))
    else:
        return 0

def get_neighbors(img : Image.Image, x : int, y : int) -> list[int]:
    return [
        get_pixel_in_range(img, x-1, y+1),
        get_pixel_in_range(img, x-1, y  ),
        get_pixel_in_range(img, x-1, y-1),

        get_pixel_in_range(img, x  , y+1),
        get_pixel_in_range(img, x  , y  ),
        get_pixel_in_range(img, x  , y-1),

        get_pixel_in_range(img, x+1, y+1),
        get_pixel_in_range(img, x+1, y  ),
        get_pixel_in_range(img, x+1, y-1),
    ]

def get_boundary_points(img : Image.Image) -> list[tuple[int, int]]:
    points = []
    for x in range(width):
        for y in range(height):
            pix = get_pixel_in_range(img, x, y)
            if not all([neighbor == pix for neighbor in get_neighbors(img, x, y)]):
                points.append((x, y))

    return points

def step_image(img : Image.Image, upper_bound : int) -> Image.Image:
    arr = np.array(img)

    nums = list(range(0, 255, step))

    for idx, num in enumerate(nums):
        i = max(idx-1, 0)
        arr = np.where(np.logical_and(arr<num, arr>nums[i]), num, arr)

    """
    for coord, val in np.ndenumerate(arr):
        for i in nums:
            if val > i:
                arr[coord] = i

    """

    return Image.fromarray(arr)

print("Stepping Image...")
a = step_image(heightmap, 127)

print("Finding Boundaries...")
boundary_points : list[tuple[int, int]] = get_boundary_points(a)

print("Drawing Lines...")
output = heightmap
for b in boundary_points:
    output.putpixel(b, (0))

output.show()
