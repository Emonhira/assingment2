import numpy as np
import matplotlib.pyplot as plt
import time

def build_complex_grid(width, height,
                       x_min=-2.5, x_max=1.0,
                       y_min=-1.25, y_max=1.25):
    """Return a (height × width) complex NumPy array."""
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    return x[np.newaxis, :] + 1j * y[:, np.newaxis]

def mandelbrot_numpy(width, height,
                     x_min=-2.5, x_max=1.0,
                     y_min=-1.25, y_max=1.25,
                     max_iter=100):
   
    C = build_complex_grid(width, height, x_min, x_max, y_min, y_max)
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)
 
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1
 
    return M
 