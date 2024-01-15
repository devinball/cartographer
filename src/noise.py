import math
import random
from vectors import vec

def interpolate(a : float, b : float, t : float):
    return (b - a) * (3.0 - t * 2.0) * t * t + a

def perlin(x : float, y : float) -> float:
    x_floor = math.floor(x)
    y_floor = math.floor(y)

