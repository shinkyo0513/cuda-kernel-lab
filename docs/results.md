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
| 256 | 256 | 256 | 0.0964 | 0.0653 | 0.0311 | TODO | TODO |
| 512 | 512 | 512 | 0.7007 | 0.4502 | 0.0800 | TODO | TODO |
| 1024 | 1024 | 1024 | 5.5575 | 3.2873 | 0.3722 | TODO | TODO |
| 2048 | 2048 | 2048 | 37.9663 | 24.4523 | 3.7908 | TODO | TODO |

## Observations

- The tiled CUDA version improves over naive CUDA by reusing A/B tiles in shared memory.
- cuBLAS is still significantly faster due to more advanced optimizations such as register tiling, Tensor Cores, architecture-specific tuning, and better instruction scheduling.