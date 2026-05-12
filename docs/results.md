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

## Attention Benchmark

Implementation:

```text
K_T    = transpose(K)
scores = Q * K_T / sqrt(D)
probs  = softmax(scores)
O      = probs * V
```

| Version | Description | Main Difference |
|---|---|---|
| v0 baseline | transpose + tiled matmul + multi-kernel softmax + tiled matmul | Simple modular baseline |
| v1 workspace | Preallocated workspace | Removes cudaMalloc/free from hot path |
| v2 fused softmax | One-kernel row-wise softmax | Reduces kernel launches and global memory traffic |
| v3 cuBLAS backend | cuBLAS for GEMM stages | Uses optimized vendor GEMM |

Values are reported as `latency ms / GFLOP/s`.

| Version | S=128, D=64 | S=256, D=64 | S=512, D=64 | S=1024, D=64 | S=1000, D=80 |
|---|---:|---:|---:|---:|---:|
| CPU reference | 2.0753 / 2.07 | 8.8517 / 1.94 | 36.1791 / 1.90 | 148.9684 / 1.84 | 176.1196 / 1.85 |
| v0 baseline | 0.1066 / 40.35 | 0.3016 / 56.99 | 0.4987 / 137.79 | 1.8954 / 144.98 | 2.2619 / 144.16 |
| v1 workspace | 0.0926 / 46.46 | 0.1422 / 120.90 | 0.3668 / 187.34 | 1.5076 / 182.27 | 1.8586 / 175.45 |
| v2 fused softmax | 0.0530 / 81.21 | 0.1007 / 170.73 | 0.2703 / 254.21 | 1.1401 / 241.02 | 1.4799 / 220.34 |
| v3 cuBLAS backend | 0.0411 / 104.74 | 0.0632 / 272.03 | 0.1366 / 503.03 | 0.5404 / 508.53 | 0.5730 / 569.10 |

## Attention Benchmark Observations

- v1 improves over v0 by removing repeated `cudaMalloc`/`cudaFree` from the timed path.
- v2 improves over v1 by replacing the multi-kernel softmax with a fused one-block-per-row softmax.
- v3 is the fastest implementation in all tested cases because the GEMM stages use cuBLAS.
- Peak throughput reaches 569.10 GFLOP/s for v3 at S=1000, D=80.
- All four GPU versions pass correctness against the CPU reference for the listed cases.

## Softmax Benchmark

This compares the original multi-kernel softmax implementation with the fused one-block-per-row softmax implementation.

| Rows | Cols | Multi-kernel softmax ms | Multi-kernel GFLOP/s | Fused softmax ms | Fused GFLOP/s | Fused speedup | Multi-kernel max error | Fused max error | Correctness |
| ---: | ---: | ----------------------: | -------------------: | ---------------: | ------------: | ------------: | ---------------------: | --------------: | ----------- |
| 1024 |  128 |                  0.0786 |                 8.34 |           0.0199 |         32.99 |         3.96x |              1.043e-07 |       7.451e-08 | PASS        |
| 1024 |  256 |                  0.2056 |                 6.38 |           0.0380 |         34.47 |         5.41x |              6.706e-08 |       5.960e-08 | PASS        |
| 1024 |  512 |                  0.4296 |                 6.10 |           0.0844 |         31.07 |         5.09x |              4.470e-08 |       4.470e-08 | PASS        |
| 1024 | 1000 |                  0.7008 |                 7.31 |           0.1710 |         29.95 |         4.10x |              3.353e-08 |       3.353e-08 | PASS        |

## Softmax Observations

- The fused implementation is faster in all tested cases, with speedups from 3.96x to 5.41x.
- The fused version reaches up to 34.47 GFLOP/s, while the multi-kernel version stays around 6 to 8 GFLOP/s.
- All tested fused cases now pass correctness, with max error near the multi-kernel implementation.
