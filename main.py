from PIL import Image
import numpy as np

fp : str = "./render.png"
regions : int = 10
render_scale : float = 1

heightmap = Image.open(fp).convert(mode="RGB")
heightmap = heightmap.resize(size=(heightmap.width * render_scale, heightmap.height * render_scale), resample=Image.BILINEAR)
output = Image.new(mode="RGB", size=(heightmap.width, heightmap.height))

boundary_points : list[tuple[int, int]] = []

def put_pixel_in_range(img : Image.Image, x : int, y : int) -> int:
    if x > 0 and x < img.width and y > 0 and y < img.height:
        img.putpixel((x,y), (0))

def get_pixel_in_range(img : Image.Image, x : int, y : int) -> int:
    if x > 0 and x < img.width and y > 0 and y < img.height:
        return img.getpixel((x,y))[0]
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
    return [(x,y) for y in range(0, heightmap.height, 4) for x in range(0, heightmap.width, 4) if sum(get_neighbors(img, x, y)) != 0 and get_pixel_in_range(img, x, y) == 0]

def clip_image(img : Image.Image, upper_bound : int) -> Image.Image:
    arr = np.asarray(img)
    # cut out lower boundary
    arr = np.where(arr <= upper_bound, arr, 0)

    return Image.fromarray(arr)

def split_to_layers(img : Image.Image) -> list[Image.Image]:
    return [clip_image(img, i) for i in range(0, 255, 100)]

for idx, layer in enumerate(split_to_layers(heightmap)):
    print(f"Working on layer {idx}")
    boundary_points.extend(get_boundary_points(layer))

output = heightmap

for b in boundary_points:
    for x in range(-1, 1):
        for y in range(-1, 1):
            put_pixel_in_range(output, b[0]+x, b[1]+y)

output.show()
