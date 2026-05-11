# Profiling Notes

## Tools

- Nsight Compute: used to inspect individual CUDA kernel metrics.
- Nsight Systems: not available in the current Colab environment.

## Matmul Naive Kernel

### Observations

| Metric | Value | Notes |
|---|---:|---|
| Kernel duration | 14.73ms | Runtime of one profiled kernel launch |
| Compute throughput | 58.70% | Indicates how much of peak compute is utilized |
| Memory throughput | 58.70% | Overall memory system utilization |
| DRAM throughput | 3.44% | Global memory pressure |
| L1/TEX throughput | 81.26% | L1/cache path activity |
| L2 throughput | 35.41% | L2 cache activity |
| Achieved occupancy | 100.28% | Active warps relative to theoretical maximum |
| Registers per thread | 30 | Register usage per CUDA thread |
| Shared memory per block | 0 | Expected to be low or zero for naive matmul |
| Block size | 1024 | Usually 16x16 or similar |
| Grid size | 1024 | Depends on matrix shape |
| Main bottleneck | global memory traffic | Example: global memory traffic / low arithmetic intensity |

### Interpretation

The naive matmul kernel is not primarily DRAM-bandwidth bound. DRAM throughput is low, while L1/TEX throughput is high, indicating that many repeated global loads are served by the cache hierarchy. The main bottleneck is inefficient data reuse and heavy pressure on the on-chip memory/cache path. Although occupancy is high, the kernel does not explicitly reuse A/B tiles through shared memory, resulting in lower arithmetic intensity and limited compute utilization compared with tiled GEMM.

---

## Matmul Tiled Kernel

### Observations

| Metric | Value | Notes |
|---|---:|---|
| Kernel duration | 2.41ms | Runtime of one profiled kernel launch |
| Compute throughput | 76.25% | Should usually improve over naive matmul |
| Memory throughput | 76.25 | Overall memory system utilization |
| DRAM throughput | 1.40% | Should be more efficient per FLOP than naive |
| L1/TEX throughput | 90.56% | May increase due to shared/L1 activity |
| L2 throughput | 5.92% | Global memory traffic through L2 |
| Achieved occupancy | 99.94% | May be lower than naive due to shared memory usage |
| Registers per thread | 42 | May be slightly higher than naive |
| Shared memory per block | 8.19 | Expected to be nonzero due to A/B tiles |
| Block size | 1024 | Usually TILE_SIZE x TILE_SIZE |
| Grid size | 800 | Depends on matrix shape |
| TILE_SIZE | 32 | Example: 8, 16, 32 |
| Main bottleneck | On-chip memory/shared-memory throughput and instruction throughput | Example: memory bandwidth / occupancy / instruction throughput |

### Interpretation

The main bottleneck of the tiled matmul kernel is no longer global memory access, but rather on-chip memory activity and compute instruction throughput. Shared-memory tiling significantly reduces DRAM and L2 traffic, but the kernel still lacks advanced GEMM optimizations such as register tiling, Tensor Core usage, and double buffering. As a result, its compute throughput is improved over the naive version but still far from highly optimized vendor libraries.

---

## Naive vs Tiled Summary

| Metric | Naive Matmul | Tiled Matmul | Expected Difference |
|---|---:|---:|---|
| Kernel duration | 14.73 ms | 2.41 ms | Tiled is much faster due to shared-memory data reuse |
| Compute throughput | 58.70% | 76.25% | Tiled achieves higher compute throughput |
| DRAM throughput | 3.44% | 1.40% | Tiled reduces DRAM traffic by reusing A/B tiles in shared memory |
| Achieved occupancy | 100.28% | 99.94% | Both are close to full occupancy; occupancy is not the main bottleneck |
| Registers per thread | 30 | 42 | Tiled uses more registers due to tile indexing and additional intermediate variables |
| Shared memory per block | 0 KB | 8.19 KB | Tiled uses shared memory for A/B tiles |
| Main bottleneck | Heavy L1/TEX and L2 cache activity due to poor explicit data reuse | On-chip memory/shared-memory activity and compute instruction throughput | Tiled shifts the bottleneck away from global memory/L2 traffic toward on-chip memory and compute efficiency |
|

### Interpretation

Compared with the naive kernel, the tiled matmul kernel reduces runtime from 14.73 ms to 2.41 ms. DRAM throughput decreases from 3.44% to 1.40%, and L2 throughput decreases from 35.41% to 5.92%, indicating that shared-memory tiling successfully reduces global memory traffic. Compute throughput improves from 58.70% to 76.25%. The high L1/TEX throughput of 90.56% suggests that the bottleneck has shifted from global memory/cache traffic to on-chip memory activity and instruction throughput. Occupancy remains close to 100%, so occupancy is not the limiting factor.

## Comparison with cuBLAS

| Implementation | Runtime | GFLOP/s | Notes |
|---|---:|---:|---|
| CUDA naive matmul | 6.2400ms | 344.145 | One thread computes one output element |
| CUDA tiled matmul | 3.6038ms | 595.888 | Uses shared memory tiling |
| cuBLAS SGEMM | 0.5204ms | 4126.79 | Highly optimized vendor implementation |

cuBLAS is faster because it uses highly optimized GEMM kernels, including architecture-specific tiling, register blocking, better memory pipelining, instruction scheduling, and Tensor Core paths where applicable.

## Attention Profiling Summary

| Kernel Stage | Kernel Name | Duration | Compute Throughput | Memory Throughput | Occupancy | Notes |
|---|---|---:|---:|---:|---:|---|
| K transpose | transpose_shared_kernel | 6.78us | 16.22% | 16.22% | 100.70% | Creates K_T |
| QK^T matmul | matmul_tiled_kernel | 356.48us | 71.45% | 71.45% | 99.10% | Computes scores |
| Scale | scale_kernel | 38.11us | 23.85% | 72.62% | 83.31% | Elementwise scaling |
| Softmax max | max_row_wise_kernel | 7.14us | 26.23% | 26.23% | 96.99% | Row-wise max reduction |
| Softmax exp | subtract_max_and_exp_kernel | 52.61 | 27.45% | 57.55% | 91.61 | exp(x - max) |
| Softmax sum | sum_row_wise_kernel | 108.61us | 54.91% | 54.91% | 98.57% | Row-wise sum reduction |
| Softmax normalize | normalize_kernel | 51.97us | 28.01% | 55.41% | 90.05% | Normalize probabilities |
| PV matmul | matmul_tiled_kernel | 414.43us | 59.70% | 59.70% | 100.32% | Computes output |