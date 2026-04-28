# ============================================================
# Makefile — CUDA DIY Exercises 1-5
# ============================================================
# Usage:
#   make all        — build all 5 exercises
#   make ex01       — build only Exercise 1
#   make clean      — remove all binaries
#   make run_ex01   — build and run Exercise 1
#
# Adjust SM (compute capability) to match your GPU:
#   RTX 2060/2070/2080 : sm_75
#   RTX 3060/3070/3080 : sm_86
#   RTX 4080/4090      : sm_89
#   A100               : sm_80
#   H100               : sm_90
# ============================================================

SM        := sm_86
NVCC      := nvcc
NVCCFLAGS := -O2 -arch=$(SM) -lineinfo

# Libraries
LIBS_BASE  := -lm
LIBS_BLAS  := -lcublas $(LIBS_BASE)
LIBS_FULL  := -lcudnn -lcublas $(LIBS_BASE)

.PHONY: all clean ex01 ex02 ex03 ex04 ex05 2 3 4 5 \
	run_ex01 run_ex02 run_ex03 run_ex04 run_2 run_3 run_4 run_5

all: ex01 ex02 ex03 ex04 ex05 2 3 4 5

ex01: ex01_cuda_basics.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex02: ex02_memory_hierarchy.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex03: ex03_ml_primitives.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex04: ex04_cnn_layers.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BLAS)
	@echo "[✓] Built $@"

ex05: ex05_mnist_cnn.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_FULL)
	@echo "[✓] Built $@"

2: 2.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

3: 3.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

4: 4.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BLAS)
	@echo "[✓] Built $@"

5: 5.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_FULL)
	@echo "[✓] Built $@"

run_ex01: ex01
	./ex01

run_ex02: ex02
	./ex02

run_ex03: ex03
	./ex03

run_ex04: ex04
	./ex04

run_2: 2
	./2

run_3: 3
	./3

run_4: 4
	./4

run_5: 5
	./5

clean:
	rm -f ex01 ex02 ex03 ex04 ex05 2 3 4 5
	@echo "[✓] Cleaned"
