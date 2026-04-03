import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
 
# ── import shared helpers ──────────────────────────────────────────────────────
from mandelbrot_core import (
    build_complex_grid,
    mandelbrot_numpy,
    mandelbrot_chunk_worker,
    timed,
    plot_mandelbrot,
    plot_chunk_analysis,
    plot_speedup,
    plot_dask_comparison
)


def mandelbrot_parallel(width, height, num_processes, chunk_size,
                        x_min=-2.5, x_max=1.0,
                        y_min=-1.25, y_max=1.25,
                        max_iter=100):
    
    C = build_complex_grid(width, height, x_min, x_max, y_min, y_max)
 
    # Split rows into chunks
    row_chunks = [C[i:i + chunk_size, :] for i in range(0, height, chunk_size)]
    args = [(chunk, max_iter) for chunk in row_chunks]
 
    with Pool(processes=num_processes) as pool:
        results = pool.map(mandelbrot_chunk_worker, args)
 
    return np.vstack(results)
 
def experiment_chunk_size(width=1000, height=1000, max_iter=100, repeats=3):
    
    chunk_sizes   = [10, 25, 50, 100, 200, 500]
    process_counts = [1, 2, 4, min(8, cpu_count())]
 
    rows = []
    total = len(process_counts) * len(chunk_sizes)
    done  = 0
 
    print(f"\n{'─'*55}")
    print(f"  Experiment A: Chunk-size sweep  ({width}×{height}, max_iter={max_iter})")
    print(f"  {total} combinations × {repeats} repeats each")
    print(f"{'─'*55}")
 
    for P in process_counts:
        for cs in chunk_sizes:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                mandelbrot_parallel(width, height, P, cs, max_iter=max_iter)
                times.append(time.perf_counter() - t0)
            avg = float(np.mean(times))
            rows.append({"processes": P, "chunk_size": cs, "time": avg})
            done += 1
            print(f"  [{done:2d}/{total}] P={P:2d}, chunk={cs:4d}  →  {avg:.3f}s")
 
    return pd.DataFrame(rows) 
 

def experiment_speedup(width=1000, height=1000, max_iter=100,
                       best_chunk=100, repeats=3):
    
    process_counts = [1, 2, 4, min(8, cpu_count())]
 
    
    print(f"\n{'─'*55}")
    print(f"  Experiment B: Speedup analysis  ({width}×{height}, chunk={best_chunk})")
    print(f"{'─'*55}")
    print("  Measuring NumPy baseline …", end=" ", flush=True)
    t_baseline, _ = timed(mandelbrot_numpy, width, height,
                          max_iter=max_iter, repeats=repeats)
    print(f"{t_baseline:.3f}s")
 
    rows = []
    for P in process_counts:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mandelbrot_parallel(width, height, P, best_chunk, max_iter=max_iter)
            times.append(time.perf_counter() - t0)
        avg     = float(np.mean(times))
        speedup = t_baseline / avg
        rows.append({"processes": P, "time": avg, "speedup": speedup})
        print(f"  P={P:2d}  →  {avg:.3f}s   speedup={speedup:.2f}×")
 
    return pd.DataFrame(rows), t_baseline


