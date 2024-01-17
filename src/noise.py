import math
import numba
import numpy as np

@numba.njit
def fract(a) -> float:
    return a - math.floor(a)

@numba.njit
def dot(a : np.ndarray, b : np.ndarray):
    sum : float = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

@numba.njit
def rand2(co : np.ndarray) -> np.ndarray:
    co = np.array([dot(co, np.array([127.1,311.7])), dot(co, np.array([269.5,183.3]))])
    a = np.array([math.sin(co[0]), math.sin(co[1])]) * 43758.5453123
    return -1.0 + 2.0 * np.array([fract(a[0]), fract(a[1])])

@numba.njit
def interpolate(a : float, b : float, t : float):
    return (b - a) * (3.0 - t * 2.0) * t * t + a

@numba.njit
def perlin(x : float, y : float) -> float:
    pos : np.ndarray = np.array([float(math.floor(x)), float(math.floor(y))], dtype=np.float32)
    fract_pos : np.ndarray = np.array([fract(x), fract(y)], dtype=np.float32)

    c   : float = dot(rand2(pos + np.array([0, 0])), fract_pos - np.array([0,0]))
    cx  : float = dot(rand2(pos + np.array([1, 0])), fract_pos - np.array([1,0]))
    cy  : float = dot(rand2(pos + np.array([0, 1])), fract_pos - np.array([0,1]))
    cxy : float = dot(rand2(pos + np.array([1, 1])), fract_pos - np.array([1,1]))

    d : np.ndarray = np.array([x - math.floor(x), y - math.floor(y)])
    
    u : np.ndarray = d*d*(3.0-2.0*d)
    
    ic0 : float = interpolate(c, cx, u[0])
    ic1 : float = interpolate(cy, cxy, u[0])

    return interpolate(ic0, ic1, u[1]) * 0.5 + 0.5

@numba.njit
def fBm(noise_function, x : float, y : float, octaves : int, lacunarity : float, gain : float) -> float:
    sum : float = 0
    octave_frequency : float = 1.0
    octave_amplitude : float = 0.5

    for i in range(octaves):
        sum += noise_function(x * octave_frequency, y * octave_frequency) * octave_amplitude
        octave_frequency *= lacunarity
        octave_amplitude *= gain

    return sum
