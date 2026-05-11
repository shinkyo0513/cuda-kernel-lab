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

|    S |  D |   CPU ms | CPU GFLOP/s | CUDA ms | CUDA GFLOP/s | CUDA speedup | Max error | Correctness |
| ---: | -: | -------: | ----------: | ------: | -----------: | -----------: | --------: | ----------- |
|  128 | 64 |   2.1707 |        1.98 |  0.1059 |        40.63 |       20.50x |      0.00 | PASS        |
|  256 | 64 |   9.5930 |        1.79 |  0.1562 |       110.04 |       61.41x |      0.00 | PASS        |
|  512 | 64 |  36.9625 |        1.86 |  0.4875 |       140.94 |       75.82x |      0.00 | PASS        |
| 1024 | 64 |      N/A |         N/A |  1.9253 |       142.73 |          N/A |       N/A | N/A         |
| 1000 | 80 | 186.1829 |        1.75 |  2.2117 |       147.44 |       84.18x |      0.00 | PASS        |

## Attention Observations

- CUDA attention is significantly faster than the CPU baseline for all tested cases.
- CUDA speedup increases with larger sequence length, from 20.50x at S=128, D=64 to 84.18x at S=1000, D=80.
- CUDA throughput improves with larger workloads, reaching around 140 to 147 GFLOP/s for the larger cases.
- The current implementation materializes both scores [S, S] and probs [S, S], which is simple and modular but increases global memory traffic.
- The S=1024, D=64 case was benchmarked only on CUDA, so CPU speedup and correctness are not reported for that row.

## Attention Benchmark With Workspace

This version uses `attention_cpu_with_workspace` and `launch_attention_cuda_with_workspace`, so the intermediate buffers are allocated once by the benchmark and reused across warmup and timed iterations.

|    S |  D |   CPU ms | CPU GFLOP/s | CUDA ms | CUDA GFLOP/s | CUDA speedup | Max error | Correctness |
| ---: | -: | -------: | ----------: | ------: | -----------: | -----------: | --------: | ----------- |
|  128 | 64 |   2.1224 |        2.03 |  0.0652 |        65.93 |       32.55x |      0.00 | PASS        |
|  256 | 64 |   9.0239 |        1.90 |  0.0881 |       195.14 |      102.43x |      0.00 | PASS        |
|  512 | 64 |  38.3580 |        1.79 |  0.2362 |       290.92 |      162.40x |      0.00 | PASS        |
| 1024 | 64 |      N/A |         N/A |  0.9478 |       289.93 |          N/A |       N/A | N/A         |
| 1000 | 80 | 174.3108 |        1.87 |  1.8451 |       176.73 |       94.47x |      0.00 | PASS        |

## Attention Workspace Observations

- Reusing workspace removes repeated `cudaMalloc`/`cudaFree` overhead from the timed CUDA path.
- CUDA throughput improves from roughly 40 to 147 GFLOP/s in the original benchmark to roughly 66 to 291 GFLOP/s with reusable workspace.
- The largest reported CPU/CUDA speedup increases to 162.40x for S=512, D=64.
- Correctness remains unchanged for CPU-checked cases, with max error reported as 0.00.

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
