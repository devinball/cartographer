import numba
import numpy as np

num_raindrops : int = 1000
raindrop_lifetime : int = 30
raindrop_capacity : float = 0.5

@numba.njit
def erode(heightmap : np.ndarray):
    pass