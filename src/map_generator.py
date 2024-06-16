import numba
import math
import random
import numpy as np
from noise import *
from PIL import Image

numba.config.THREADING_LAYER = 'omp'

WIDTH, HEIGHT, COLORS = 2048, 2048, 3

MAX_PIXEL_VALUE : int = 255

FREQUENCY = 4
AMPLITUDE = 1

USE_FALLOFF : bool = True
CIRCULAR_FALLOFF : bool = False
FALLOFF_STRENGTH = 0.4
RIVER_INITIAL_CAPACITY : int = 1000000
MIN_RIVER_PATH_LENGTH : int = 50
NUM_RIVERS : int = 100

class PathPoint():
    def __init__(self, x, y, width) -> None:
        self.x = x
        self.y = y
        self.width = width

    def __str__(self) -> str:
        return f"({self.x}, {self.y} | {self.width})"
    
    def __repr__(self) -> str:
        return self.__str__()

class Path():
    points : list[PathPoint] = []
    reverse = False

    def __init__(self, reverse : bool = False) -> None:
        self.reverse = reverse

    def raster(self):
        for i in reversed(self.points) if self.reverse else self.points:
            ... # Do something?

class River(Path):
    def __init__(self, name, capacity : int, reverse : bool = False) -> None:
        self.name = name
        self.capacity = capacity
        super().__init__(reverse)

    def get_nearest_point(self, x : int, y : int) -> tuple[PathPoint, float]:
        nearest : PathPoint
        dist = 100000
        for i in self.points:
            d = math.sqrt((i.x - x)**2 + (i.y - y)**2)
            if d < dist:
                dist = d
                nearest = i

        return nearest, dist
    
    def is_colliding(self, x : int, y : int, epsilon : float) -> bool:
        for point in self.points:
            if math.sqrt((point.x - x)**2 + (point.y - y)**2) < epsilon:
                return True
        return False

    def generate(self, max_steps : int, point_distance : int, search_distance_multipler : float, img : np.ndarray, start_x : int, start_y : int) -> None:
        pos_x = start_x
        pos_y = start_y

        flow_dir_x = 0
        flow_dir_y = 0

        for _ in range(max_steps):
            # Find direction gradient
            gradient : list[tuple[float, float]] = []
            for angle in range(0, 360, 15):
                x = search_distance_multipler * point_distance * math.cos(angle * 0.0174533) + pos_x
                y = search_distance_multipler * point_distance * math.sin(angle * 0.0174533) + pos_y

                if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
                    return

                value = img[int(x), int(y)]

                gradient.append((value, angle))

            # Break early if the river flows off the map
            if pos_x < 0 or pos_x > WIDTH or pos_y < 0 or pos_y > HEIGHT:
                return
            
            # Break if the river runs out of water
            if self.capacity <= 0:
                return

            if self.is_colliding(x, y, point_distance):
                return

            # Break early if the river gets stuck
            v = img[int(pos_x), int(pos_y)]
            #if len([d for d in gradient if d[0] >= v]) == len(gradient):
            #    return
            
            # Break if we reach to ocean
            if v == 0:
                return
            
            # Weighted average of angles, the generally lowest directions
            #angle = sum([i[0] * i[1] for i in gradient]) / len(gradient) #/ sum([i[0] for i in gradient])
            #angle = min([i[0] for i in gradient])

            # Find the lowest elevation area for the river to flow to
            angle = 0
            smallest = 1000
            for g in gradient:
                if g[0] < smallest:
                    smallest = g[0]
                    angle = g[1]

            flow_dir_x += math.cos(angle)
            flow_dir_y += math.sin(angle)

            magnitude = math.sqrt(flow_dir_x**2 + flow_dir_y**2)

            flow_dir_x = (flow_dir_x / magnitude) * point_distance
            flow_dir_y = (flow_dir_y / magnitude) * point_distance

            self.points.append(PathPoint(pos_x, pos_y, 1))

            pos_x += flow_dir_x
            pos_y += flow_dir_y

            self.capacity -= 1

def generate_rivers(heightmap : np.ndarray, map : np.ndarray, num_rivers : int):
        points = []

        river_count = num_rivers

        while len(points) < river_count:
            current_point = (random.randrange(0, WIDTH), random.randrange(0, HEIGHT))

            avaliable = True

            if heightmap[current_point[0], current_point[1]] == 0:
                avaliable = False
                

            #for p in points:
            #    if math.sqrt((p[0] - current_point[0])**2 + (p[1] - current_point[1])**2) < 100:
            #        avaliable = False

            if avaliable:
                points.append(current_point)

        rivers : list[River] = []

        for i in range(river_count):
            p = points.pop(random.randrange(0, len(points)))
            river_start_x = p[0]
            river_start_y = p[1]

            r = River(f"{i}", capacity=RIVER_INITIAL_CAPACITY)
            r.generate(512, 2, 2, heightmap, river_start_x, river_start_y)
            rivers.append(r)

        for r in rivers:
            if len(r.points) < MIN_RIVER_PATH_LENGTH:
                continue

            for p in r.points:
                if p.x > 0 and p.x < WIDTH and p.y > 0 and p.y < HEIGHT:
                    if p.width == 0:
                        map[int(p.x), int(p.y)] = (0, 0, 255)
                    else:
                        for x in range(-p.width,p.width):
                            for y in range(-p.width,p.width):
                                map[int(p.x) + x, int(p.y) + y] = (0, 0, 255)

@numba.njit(parallel=True)
def generate_base_map(img : np.ndarray, noise_function) -> None:
    land_threshold = 0.5

    length = WIDTH * HEIGHT

    for i in numba.prange(length):
        x = i % WIDTH
        y = math.floor(i / WIDTH)

        a = max(WIDTH, HEIGHT)

        sample_x = (x / a)
        sample_y = (y / a)

        noise = noise_function(sample_x, sample_y)
        noise = 0 if noise < land_threshold else noise ** 1.5

        img[x, y] = noise * AMPLITUDE

@numba.njit()
def base_noise(x : float, y : float) -> float:
    sample_x = x * FREQUENCY + x_offset
    sample_y = y * FREQUENCY + y_offset

    noise = fBm(perlin, sample_x, sample_y, 10, 2, 0.5) * 0.5 + 0.5
    if USE_FALLOFF:
        if CIRCULAR_FALLOFF:
            # lol
            raise RuntimeError("Not implemented yet")
            #noise *= (x - 0.5)**2 + (y - 0.5)**2
        else:
            k = FALLOFF_STRENGTH
            noise *= min(1 - 2 * abs(x - 0.5), k) * (1/k) * min(1 - 2 * abs(y - 0.5), k) * (1/k)
        
    #river_noise = (1 - abs(fBm(perlin, sample_x, sample_y, 10, 2, 0.5))) ** 16
    #noise = 0 if noise * falloff < land_threshold else noise
    #river_noise = 0 if river_noise * falloff < 0.95 else 1
    #results = max(0, noise - river_noise)
    
    return noise

@numba.njit(parallel=True)
def heightmap_to_map(heightmap : np.ndarray, map : np.ndarray):
    length = WIDTH * HEIGHT

    for i in numba.prange(length):
        x = i % WIDTH
        y = math.floor(i / WIDTH)

        v = heightmap[x, y] * MAX_PIXEL_VALUE
        map[x, y, 0] = v
        map[x, y, 1] = v
        map[x, y, 2] = v

def main():    
    heightmap = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    print("Generating base map...")
    generate_base_map(heightmap, base_noise)

    map = np.zeros((HEIGHT, WIDTH, COLORS), dtype=np.int8)
    heightmap_to_map(heightmap, map)

    print("Generating rivers...")
    generate_rivers(heightmap, map, NUM_RIVERS)

    img = Image.fromarray(map, mode="RGB")
    img.show()

if __name__ == "__main__":
    x_offset, y_offset = random.randint(-1000000, 1000000), random.randint(-1000000, 1000000) #493157, -617065 

    print(f"X: {x_offset}, Y: {y_offset}")

    main()
