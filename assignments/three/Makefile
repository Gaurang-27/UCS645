# Makefile – Correlation Calculator
# UCS645: Parallel & Distributed Computing

CXX := g++
BASE_FLAGS := -std=c++11 -Wall -Wextra -O3

FLAGS_SEQ := $(BASE_FLAGS)
FLAGS_OMP := $(BASE_FLAGS) -fopenmp
FLAGS_OPT := $(BASE_FLAGS) -fopenmp -march=native -mavx2 -mfma

SRC_MAIN := main.cpp
SRC_CORE := correlate.cpp
HDR      := correlate.h

BIN_SEQ := correlate_seq
BIN_OMP := correlate_omp
BIN_OPT := correlate_opt

.DEFAULT_GOAL := all

# ============================================================
# Build Targets
# ============================================================

all: $(BIN_SEQ) $(BIN_OMP) $(BIN_OPT)
	@echo "Build complete:"
	@echo "  $(BIN_SEQ)"
	@echo "  $(BIN_OMP)"
	@echo "  $(BIN_OPT)"
	@echo ""

v1: $(BIN_SEQ)
v2: $(BIN_OMP)
v3: $(BIN_OPT)

$(BIN_SEQ): $(SRC_MAIN) $(SRC_CORE) $(HDR)
	@echo "Compiling sequential version..."
	$(CXX) $(FLAGS_SEQ) -DVERSION=1 -o $@ $(SRC_MAIN) $(SRC_CORE)

$(BIN_OMP): $(SRC_MAIN) $(SRC_CORE) $(HDR)
	@echo "Compiling OpenMP version..."
	$(CXX) $(FLAGS_OMP) -DVERSION=2 -o $@ $(SRC_MAIN) $(SRC_CORE)

$(BIN_OPT): $(SRC_MAIN) $(SRC_CORE) $(HDR)
	@echo "Compiling optimized version..."
	$(CXX) $(FLAGS_OPT) -DVERSION=3 -o $@ $(SRC_MAIN) $(SRC_CORE)

# ============================================================
# Testing
# ============================================================

test: $(BIN_SEQ) $(BIN_OMP) $(BIN_OPT)
	@echo "Running correctness test (100x100)"
	@./$(BIN_SEQ) 100 100 | tail -6
	@OMP_NUM_THREADS=4 ./$(BIN_OMP) 100 100 | tail -6
	@OMP_NUM_THREADS=4 ./$(BIN_OPT) 100 100 | tail -6

compare: $(BIN_SEQ) $(BIN_OMP) $(BIN_OPT)
	@echo "Performance comparison (500x500)"
	@./$(BIN_SEQ) 500 500 | grep "Time"
	@OMP_NUM_THREADS=4 ./$(BIN_OMP) 500 500 | grep "Time"
	@OMP_NUM_THREADS=4 ./$(BIN_OPT) 500 500 | grep "Time"

# ============================================================
# Analysis
# ============================================================

test-sizes: $(BIN_OPT)
	@for s in 100 500 1000 2000; do \
		echo "Size: $$s x $$s"; \
		OMP_NUM_THREADS=4 ./$(BIN_OPT) $$s $$s | grep "Time"; \
	done

test-threads: $(BIN_OMP)
	@for t in 1 2 4 8; do \
		echo "Threads: $$t"; \
		OMP_NUM_THREADS=$$t ./$(BIN_OMP) 1000 1000 | grep "Time"; \
	done

perf-seq: $(BIN_SEQ)
	perf stat -d ./$(BIN_SEQ) 1000 1000

perf-omp: $(BIN_OMP)
	OMP_NUM_THREADS=4 perf stat -d ./$(BIN_OMP) 1000 1000

perf-opt: $(BIN_OPT)
	OMP_NUM_THREADS=4 perf stat -d ./$(BIN_OPT) 1000 1000

check-avx: $(BIN_OPT)
	objdump -d $(BIN_OPT) | grep -E "vfmadd|vmulpd|vaddpd" | head -10 || true

# ============================================================
# Assignment Shortcuts
# ============================================================

q1: $(BIN_SEQ)
	./$(BIN_SEQ) 1000 1000

q2: $(BIN_OMP)
	OMP_NUM_THREADS=4 ./$(BIN_OMP) 1000 1000

q3: $(BIN_OPT)
	OMP_NUM_THREADS=4 ./$(BIN_OPT) 1000 1000

q4: $(BIN_OMP) $(BIN_OPT)
	@for s in 500 1000 2000; do \
		echo "Size $$s"; \
		OMP_NUM_THREADS=4 ./$(BIN_OPT) $$s $$s | grep "Time"; \
	done
	@for t in 1 2 4 8; do \
		echo "Threads $$t"; \
		OMP_NUM_THREADS=$$t ./$(BIN_OMP) 1000 1000 | grep "Time"; \
	done

# ============================================================
# Utilities
# ============================================================

clean:
	rm -f $(BIN_SEQ) $(BIN_OMP) $(BIN_OPT) *.o

rebuild: clean all

help:
	@echo "Targets:"
	@echo "  all, v1, v2, v3"
	@echo "  test, compare"
	@echo "  test-sizes, test-threads"
	@echo "  perf-seq, perf-omp, perf-opt"
	@echo "  q1, q2, q3, q4"
	@echo "  clean, rebuild"

.PHONY: all v1 v2 v3 test compare test-sizes test-threads \
        perf-seq perf-omp perf-opt check-avx \
        q1 q2 q3 q4 clean rebuild help
