import os
import math
import numpy as np
from PIL import Image

stamp_dir = "./heightmaps/stamps/"
stamps = [np.array(Image.open(stamp_dir + i).resize((512, 512))) for i in os.listdir(stamp_dir)]

falloff_mask = np.zeros(stamps[0].shape[:2])

print("Falloff mask stuff")
for x in range(falloff_mask.shape[1]):
    for y in range(falloff_mask.shape[0]):
        delta_x = (falloff_mask.shape[1] / 2) - x
        delta_y = (falloff_mask.shape[0] / 2) - y
        delta_x /= (falloff_mask.shape[1] / 512)
        delta_y /= (falloff_mask.shape[0] / 512)
        falloff_mask[y, x] = 255 - math.sqrt(delta_x**2 + delta_y**2)

Image.fromarray(falloff_mask).show()
