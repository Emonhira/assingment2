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
 
def mandelbrot_block(C_chunk, max_iter=100):
  
    Z = np.zeros_like(C_chunk)
    M = np.zeros(C_chunk.shape, dtype=np.int32)
 
    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        if not np.any(mask):          # Note 3: all points diverged → early exit
            break
        Z[mask] = Z[mask] ** 2 + C_chunk[mask]
        M[mask] += 1
 
    return M

def mandelbrot_chunk_worker(args):

    C_chunk, max_iter = args
    return mandelbrot_block(C_chunk, max_iter)

def timed(fn, *args, repeats=3, **kwargs):
    
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), result

def plot_mandelbrot(M, title="Mandelbrot Set", filename=None):
    plt.figure(figsize=(10, 7))
    plt.imshow(M, cmap="inferno", origin="lower")
    plt.colorbar(label="Iterations")
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Saved → {filename}")
    plt.show()
 
 
def plot_chunk_analysis(df, filename=None):
    """df must have columns: processes, chunk_size, time"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for P in sorted(df["processes"].unique()):
        sub = df[df["processes"] == P].sort_values("chunk_size")
        ax.plot(sub["chunk_size"], sub["time"], marker="o", label=f"P={P}")
    ax.set_xlabel("Chunk Size (rows)")
    ax.set_ylabel("Mean Execution Time (s)")
    ax.set_title("Effect of Chunk Size on Parallel Execution Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Saved → {filename}")
    plt.show()
 
 
def plot_speedup(df, filename=None):
    """df must have columns: processes, time, speedup"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
    ax1.plot(df["processes"], df["time"], marker="o", color="steelblue")
    ax1.set_xlabel("Number of Processes")
    ax1.set_ylabel("Mean Execution Time (s)")
    ax1.set_title("Execution Time vs Number of Processes")
    ax1.grid(True, linestyle="--", alpha=0.4)
 
    ax2.plot(df["processes"], df["speedup"], marker="o",
             color="darkorange", label="Actual speedup")
    ax2.plot(df["processes"], df["processes"], "--",
             color="gray", label="Ideal (linear) speedup")
    ax2.set_xlabel("Number of Processes")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs Number of Processes")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
 
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Saved → {filename}")
    plt.show()
 
 
def plot_dask_comparison(df, filename=None):
    """df must have columns: size, numpy_time, dask_time, speedup"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
    ax1.plot(df["size"], df["numpy_time"], marker="s",
             label="NumPy baseline", color="steelblue")
    ax1.plot(df["size"], df["dask_time"], marker="o",
             label="Dask local", color="darkorange")
    ax1.set_xlabel("Grid Size (N×N)")
    ax1.set_ylabel("Mean Execution Time (s)")
    ax1.set_title("NumPy vs Dask: Execution Time")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
 
    ax2.plot(df["size"], df["speedup"], marker="o", color="green")
    ax2.axhline(1, color="gray", linestyle="--")
    ax2.set_xlabel("Grid Size (N×N)")
    ax2.set_ylabel("Speedup (NumPy / Dask)")
    ax2.set_title("Dask Speedup over NumPy")
    ax2.grid(True, linestyle="--", alpha=0.4)
 
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Saved → {filename}")
    plt.show()   
 