# CUDA Kernel Benchmark Results

## Environment

- Platform: Google Colab
- GPU: T4
- CUDA version: 12.8
- Compiler: nvcc
- Flags: `-O3`

## Matmul Benchmark

| M | N | K | CUDA naive ms | CUDA tiled ms | cuBLAS ms | tiled/naive speedup | cuBLAS/tiled speedup |
|---|---|---|---:|---:|---:|---:|---:|
| 256 | 256 | 256 | 0.0964 | 0.0653 | 0.0311 | 1.48x | 2.10x |
| 512 | 512 | 512 | 0.7007 | 0.4502 | 0.0800 | 1.56x | 5.63x |
| 1024 | 1024 | 1024 | 5.5575 | 3.2873 | 0.3722 | 1.69x | 8.83x |
| 2048 | 2048 | 2048 | 37.9663 | 24.4523 | 3.7908 | 1.55x | 6.45x |

## Observations

- The tiled CUDA version is consistently faster than the naive CUDA version, with speedups between **1.48x and 1.69x** in these tests.
- The improvement comes from staging A/B tiles into shared memory and reusing them within each thread block, which reduces repeated global memory accesses.
- cuBLAS remains significantly faster than the custom tiled implementation, especially for larger matrices.
- The gap to cuBLAS is expected because cuBLAS uses more advanced GEMM optimizations such as register tiling, Tensor Cores, architecture-specific tuning, better memory pipelining, and instruction scheduling.
- The custom tiled kernel is useful for learning CUDA memory hierarchy and data reuse, but it is not intended to replace vendor-optimized GEMM libraries.

## Tile Size Sweep

| TILE_SIZE | 512x512 ms | 1024x1024 ms | Notes |
|---|---:|---:|---|
| 8 | 1.07068 | 4.13004 | Lower data reuse and smaller tiles |
| 16 | 0.278318 | 2.82812 | Good baseline tile size |
| 32 | 0.264499 | 2.58758 | Best among tested sizes, with higher data reuse and larger blocks |

## Tile Size Observations

- `TILE_SIZE=8` is clearly slower, likely because each tile provides less data reuse and requires more tile iterations.
- `TILE_SIZE=16` gives a large improvement over `TILE_SIZE=8`.
- `TILE_SIZE=32` is slightly faster than `TILE_SIZE=16` for both tested matrix sizes, suggesting that the increased data reuse outweighs the extra shared memory usage in this setup.
- Larger tile sizes are not always better in general, because they can increase shared memory usage, register pressure, synchronization cost, and reduce occupancy. In this experiment, however, `TILE_SIZE=32` performed best.