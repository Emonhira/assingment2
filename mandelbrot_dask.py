import numpy as np
import pandas as pd
import time
import dask.array as da
from dask.distributed import Client, LocalCluster
 
# ── import shared helpers ──────────────────────────────────────────────────────
from mandelbrot_core import (
    build_complex_grid,
    mandelbrot_numpy,
    mandelbrot_block,   # pure NumPy — Dask maps this over blocks (Note 2, 4)
    timed,
    plot_mandelbrot,
    plot_dask_comparison,
)
 
 
# ─────────────────────────────────────────────
# Dask runner  (local OR distributed)
# ─────────────────────────────────────────────
 
def mandelbrot_dask(width, height, chunk_size=250,
                    x_min=-2.5, x_max=1.0,
                    y_min=-1.25, y_max=1.25,
                    max_iter=100):
    """
    Compute Mandelbrot with Dask using map_blocks.
 
    Key design choices (see assignment notes):
      Note 1: chunk_size is tunable — smaller often wins locally (L2 cache fit).
      Note 2: map_blocks treats mandelbrot_block as ONE atomic op per chunk,
              avoiding enormous scheduling overhead from per-iteration tracking.
      Note 3: early-exit inside mandelbrot_block when all points diverged.
      Note 4: mandelbrot_block uses plain NumPy arrays internally.
 
    Parameters
    ----------
    width, height : grid resolution
    chunk_size    : rows (and cols) per Dask chunk — tune this! (Note 1)
    max_iter      : maximum iterations per point
 
    Returns
    -------
    M : np.ndarray (int32), shape (height, width)
    """
    C = build_complex_grid(width, height, x_min, x_max, y_min, y_max)
 
    # Wrap in Dask array — chunk_size controls parallelism granularity
    C_dask = da.from_array(C, chunks=(chunk_size, chunk_size))
 
    # map_blocks applies mandelbrot_block to each chunk independently (Note 2)
    M_dask = C_dask.map_blocks(
        mandelbrot_block,
        dtype=np.int32,
        max_iter=max_iter,
    )
 
    return M_dask.compute()   # triggers computation (local or distributed)
 
 
# ─────────────────────────────────────────────
# Experiment A – chunk size sweep (local)
# ─────────────────────────────────────────────
 
def experiment_chunk_size_dask(width=1000, height=1000,
                               max_iter=100, repeats=3):
    """
    Sweep chunk sizes on a local Dask scheduler.
    Returns DataFrame: chunk_size | time
    """
    chunk_sizes = [32, 64, 128, 250, 500, 1000]
    rows = []
 
    print(f"\n{'─'*55}")
    print(f"  Dask Experiment A: Chunk-size sweep ({width}×{height})")
    print(f"{'─'*55}")
 
    for cs in chunk_sizes:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mandelbrot_dask(width, height, chunk_size=cs, max_iter=max_iter)
            times.append(time.perf_counter() - t0)
        avg = float(np.mean(times))
        rows.append({"chunk_size": cs, "time": avg})
        print(f"  chunk={cs:5d}  →  {avg:.3f}s")
 
    return pd.DataFrame(rows)
 
 
# ─────────────────────────────────────────────
# Experiment B – Dask vs NumPy across grid sizes
# ─────────────────────────────────────────────
 
def experiment_dask_vs_numpy(sizes=None, best_chunk=100,
                              max_iter=100, repeats=3):
    """
    Compare NumPy baseline vs Dask local for growing grid sizes.
    Returns DataFrame: size | numpy_time | dask_time | speedup
    """
    if sizes is None:
        sizes = [500, 1000, 2000, 3000]
 
    rows = []
 
    print(f"\n{'─'*55}")
    print(f"  Dask Experiment B: Dask vs NumPy (chunk={best_chunk})")
    print(f"{'─'*55}")
 
    for sz in sizes:
        # NumPy
        t_np, _ = timed(mandelbrot_numpy, sz, sz,
                         max_iter=max_iter, repeats=repeats)
 
        # Dask
        t_dk, _ = timed(mandelbrot_dask, sz, sz,
                         chunk_size=best_chunk, max_iter=max_iter,
                         repeats=repeats)
 
        speedup = t_np / t_dk
        rows.append({"size": sz,
                     "numpy_time": t_np,
                     "dask_time":  t_dk,
                     "speedup":    speedup})
        print(f"  {sz:4d}×{sz:<4d}  NumPy={t_np:.3f}s  "
              f"Dask={t_dk:.3f}s  speedup={speedup:.2f}×")
 
    return pd.DataFrame(rows)
 
 
# ─────────────────────────────────────────────
# Experiment C – Distributed cluster
# ─────────────────────────────────────────────
 
def experiment_distributed(width=2000, height=2000,
                            best_chunk=100, max_iter=100, repeats=3,
                            scheduler_address=None):
    """
    Run Mandelbrot on a Dask distributed cluster.
 
    Parameters
    ----------
    scheduler_address : str or None
        - None  → spin up a LocalCluster on this machine (simulates cluster)
        - 'tcp://<ip>:8786' → connect to a real Strato / remote scheduler
 
    Returns
    -------
    dict with keys: mode, time, size
    """
    print(f"\n{'─'*55}")
    print(f"  Dask Experiment C: Distributed execution ({width}×{height})")
    print(f"{'─'*55}")
 
    if scheduler_address is None:
        print("  Starting LocalCluster (4 workers, 1 thread each) …")
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
        client  = Client(cluster)
        mode    = "LocalCluster"
    else:
        print(f"  Connecting to scheduler at {scheduler_address} …")
        cluster = None
        client  = Client(scheduler_address)
        mode    = f"Cluster({scheduler_address})"
 
    print(f"  Dashboard: {client.dashboard_link}")
    print(f"  Workers  : {len(client.scheduler_info()['workers'])}")
 
    times = []
    for run in range(repeats):
        t0 = time.perf_counter()
        mandelbrot_dask(width, height, chunk_size=best_chunk, max_iter=max_iter)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {run+1}/{repeats}: {elapsed:.3f}s")
 
    avg = float(np.mean(times))
    print(f"  Mean: {avg:.3f}s  (mode={mode})")
 
    client.close()
    if cluster is not None:
        cluster.close()
 
    return {"mode": mode, "time": avg, "size": f"{width}×{height}"}
 
 
 
if __name__ == "__main__":
  
    WIDTH    = 1000
    HEIGHT   = 1000
    MAX_ITER = 100
    REPEATS  = 3
 
    
    SCHEDULER_ADDRESS = None
 
    print(f"\n{'═'*55}")
    print(f"  Mandelbrot – Dask Experiments")
    print(f"  Grid: {WIDTH}×{HEIGHT}   max_iter={MAX_ITER}")
    print(f"{'═'*55}")
 
    
    print("\nGenerating sample image (Dask, chunk=100) …")
    M_sample = mandelbrot_dask(WIDTH, HEIGHT, chunk_size=100, max_iter=MAX_ITER)
    plot_mandelbrot(M_sample,
                    title="Mandelbrot – Dask (chunk=100)",
                    filename="mandelbrot_dask_image.png")
 
   
    df_chunks = experiment_chunk_size_dask(WIDTH, HEIGHT, MAX_ITER, REPEATS)
    print("\nDask chunk-size results:")
    print(df_chunks.to_string(index=False))
    df_chunks.to_csv("dask_chunk_size_results.csv", index=False)
 
    best_cs = int(df_chunks.loc[df_chunks["time"].idxmin(), "chunk_size"])
    print(f"\nBest Dask chunk size: {best_cs}")
 
    
    df_compare = experiment_dask_vs_numpy(
        sizes=[500, 1000, 2000],
        best_chunk=best_cs,
        max_iter=MAX_ITER,
        repeats=REPEATS,
    )
    print("\nDask vs NumPy results:")
    print(df_compare.to_string(index=False))
    df_compare.to_csv("dask_vs_numpy_results.csv", index=False)
 
    plot_dask_comparison(df_compare, filename="dask_vs_numpy.png")
 
    
    dist_result = experiment_distributed(
        width=2000, height=2000,
        best_chunk=best_cs,
        max_iter=MAX_ITER,
        repeats=REPEATS,
        scheduler_address=SCHEDULER_ADDRESS,
    )
    print(f"\nDistributed result: {dist_result}")
 
    print("\n✓ All Dask experiments complete. Output files:")
    print("    mandelbrot_dask_image.png")
    print("    dask_chunk_size_results.csv")
    print("    dask_vs_numpy.png  /  dask_vs_numpy_results.csv")