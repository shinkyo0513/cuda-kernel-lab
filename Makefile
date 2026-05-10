.PHONY: all clean \
	sum_reduction \
	max_reduction \
	softmax_benchmark \
	transpose_benchmark \
	matmul_benchmark \
	attention_benchmark \
	attention_profile

NVCC=nvcc

BUILD_DIR=build

CFLAGS=-O3 -Icommon -Ikernels
LDFLAGS_CUBLAS=-lcublas

# Kernel configuration
REDUCTION_BLOCK_SIZE=256
SOFTMAX_BLOCK_SIZE=32
MATMUL_BLOCK_SIZE=16
MATMUL_TILE_SIZE=32
TRANSPOSE_TILE_SIZE=32

DEFINES=\
	-DREDUCTION_BLOCK_SIZE=$(REDUCTION_BLOCK_SIZE) \
	-DSOFTMAX_BLOCK_SIZE=$(SOFTMAX_BLOCK_SIZE) \
	-DMATMUL_BLOCK_SIZE=$(MATMUL_BLOCK_SIZE) \
	-DMATMUL_TILE_SIZE=$(MATMUL_TILE_SIZE) \
	-DTRANSPOSE_TILE_SIZE=$(TRANSPOSE_TILE_SIZE)

all: sum_reduction max_reduction softmax_benchmark transpose_benchmark matmul_benchmark attention_benchmark

# ------------------------------------------------------------
# Reduction exercises
# ------------------------------------------------------------
sum_reduction:
	mkdir -p $(BUILD_DIR)/reduction
	$(NVCC) $(CFLAGS) $(DEFINES) \
		reduction/sum_reduction.cu \
		-o $(BUILD_DIR)/reduction/sum_reduction

max_reduction:
	mkdir -p $(BUILD_DIR)/reduction
	$(NVCC) $(CFLAGS) $(DEFINES) \
		reduction/max_reduction.cu \
		-o $(BUILD_DIR)/reduction/max_reduction

# ------------------------------------------------------------
# Softmax benchmark
# ------------------------------------------------------------
softmax_benchmark:
	mkdir -p $(BUILD_DIR)/softmax
	$(NVCC) $(CFLAGS) $(DEFINES) \
		kernels/softmax.cu \
		softmax/softmax_benchmark.cu \
		-o $(BUILD_DIR)/softmax/softmax_benchmark

# ------------------------------------------------------------
# Transpose benchmark
# ------------------------------------------------------------
transpose_benchmark:
	mkdir -p $(BUILD_DIR)/transpose
	$(NVCC) $(CFLAGS) $(DEFINES) \
		kernels/transpose.cu \
		transpose/transpose_benchmark.cu \
		-o $(BUILD_DIR)/transpose/transpose_benchmark

# ------------------------------------------------------------
# Matmul benchmark
# ------------------------------------------------------------
matmul_benchmark:
	mkdir -p $(BUILD_DIR)/matmul
	$(NVCC) $(CFLAGS) $(DEFINES) \
		kernels/matmul_cpu.cpp \
		kernels/matmul_naive.cu \
		kernels/matmul_tiled.cu \
		kernels/matmul_cublas.cu \
		matmul/matmul_benchmark.cu \
		-o $(BUILD_DIR)/matmul/matmul_benchmark \
		$(LDFLAGS_CUBLAS)

# ------------------------------------------------------------
# Attention
# ------------------------------------------------------------
attention_benchmark:
	mkdir -p $(BUILD_DIR)/attention
	$(NVCC) $(CFLAGS) $(DEFINES) \
		kernels/matmul_tiled.cu \
		kernels/softmax.cu \
		kernels/transpose.cu \
		attention/attention_cpu.cpp \
		attention/attention_cuda.cu \
		attention/attention_benchmark.cu \
		-o $(BUILD_DIR)/attention/attention_benchmark

attention_profile:
	mkdir -p $(BUILD_DIR)/attention
	$(NVCC) $(CFLAGS) $(DEFINES) \
		kernels/matmul_tiled.cu \
		kernels/softmax.cu \
		kernels/transpose.cu \
		attention/attention_cpu.cpp \
		attention/attention_cuda.cu \
		attention/attention_profile.cu \
		-o $(BUILD_DIR)/attention/attention_profile

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
