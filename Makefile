.PHONY: all reduction transpose matmul clean

NVCC=nvcc
CFLAGS=-O3 -Icommon

BUILD_DIR=build

all: reduction transpose matmul

reduction:
	mkdir -p $(BUILD_DIR)/reduction
	$(NVCC) $(CFLAGS) reduction/sum_reduction.cu -o $(BUILD_DIR)/reduction/sum_reduction

transpose:
	mkdir -p $(BUILD_DIR)/transpose
	$(NVCC) $(CFLAGS) transpose/transpose_naive.cu -o $(BUILD_DIR)/transpose/transpose_naive
	$(NVCC) $(CFLAGS) transpose/transpose_shared.cu -o $(BUILD_DIR)/transpose/transpose_shared

matmul:
	mkdir -p $(BUILD_DIR)/matmul
	$(NVCC) $(CFLAGS) matmul/matmul_naive.cu -o $(BUILD_DIR)/matmul/matmul_naive
	$(NVCC) $(CFLAGS) matmul/matmul_tiled.cu -o $(BUILD_DIR)/matmul/matmul_tiled

clean:
	rm -rf $(BUILD_DIR)
