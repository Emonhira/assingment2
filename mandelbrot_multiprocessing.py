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