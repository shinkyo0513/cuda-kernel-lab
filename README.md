# CUDA Kernel Lab

This repository is a small CUDA learning and benchmarking lab. It implements common GPU kernels from simple versions to more optimized versions, then compares correctness and performance against CPU references and vendor libraries where useful.

Current topics:

- Reductions: sum reduction and argmax reduction
- Matrix multiplication: naive CUDA, tiled CUDA, and cuBLAS SGEMM
- Matrix transpose: naive and shared-memory tiled transpose
- Softmax: multi-kernel row-wise softmax and fused one-block-per-row softmax
- Single-head attention: CPU reference, tiled CUDA, workspace reuse, fused softmax, and cuBLAS-backed GEMM

Benchmark results and notes are in:

- [docs/results.md](docs/results.md)
- [docs/profiling_notes.md](docs/profiling_notes.md)
- [profile/](profile/)

## Requirements

- NVIDIA GPU
- CUDA toolkit with `nvcc`
- cuBLAS
- `make`

The benchmark results in `docs/results.md` were collected on Google Colab with a T4 GPU and CUDA 12.8.

## Build

Build everything:

```bash
make all
```

Build one target:

```bash
make sum_reduction
make max_reduction
make softmax_benchmark
make transpose_benchmark
make matmul_benchmark
make attention_benchmark
make attention_profile
```

Clean generated binaries:

```bash
make clean
```

Build artifacts are written under `build/`.

## Run

Reduction examples:

```bash
./build/reduction/sum_reduction
./build/reduction/max_reduction
```

Softmax benchmark:

```bash
./build/softmax/softmax_benchmark
```

Transpose benchmark:

```bash
./build/transpose/transpose_benchmark
```

Matmul benchmark:

```bash
./build/matmul/matmul_benchmark
```

Attention benchmark:

```bash
./build/attention/attention_benchmark
```

The attention benchmark reports:

- CPU reference
- `v0 baseline`: tiled matmul + multi-kernel softmax with internal temporary allocation
- `v1 workspace`: same kernels with preallocated intermediate buffers
- `v2 fused softmax`: tiled matmul + fused softmax
- `v3 cuBLAS backend`: cuBLAS GEMM + fused softmax

## Profiling

Build the attention profiling binary:

```bash
make attention_profile
```

Run with Nsight Compute:

```bash
ncu ./build/attention/attention_profile
```

You can run correctness mode for the profiling binary with:

```bash
ATTENTION_CHECK=1 ./build/attention/attention_profile
```

Optional dimensions:

```bash
ATTENTION_S=512 ATTENTION_D=64 ./build/attention/attention_profile
ATTENTION_CHECK=1 ATTENTION_S=512 ATTENTION_D=64 ./build/attention/attention_profile
```

## Project Layout

```text
common/      CUDA/cuBLAS checking helpers and timers
kernels/     Reusable CUDA kernels and CPU helpers
reduction/   Standalone reduction examples
softmax/     Softmax benchmark driver
transpose/   Transpose benchmark driver
matmul/      Matmul benchmark driver
attention/   CPU/CUDA attention implementations and benchmarks
docs/        Benchmark results and profiling notes
profile/     Saved Nsight Compute outputs
```

## Notes

- Matrices are stored as row-major 1D arrays. Kernels map CUDA thread/block indices to logical 2D matrix coordinates manually.
- The fused softmax implementation currently supports rows with `ncols <= 1024`.
- The custom kernels are intended for learning and comparison. cuBLAS is expected to be faster for production GEMM workloads.
