#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 4096   // Large matrix (4096 x 4096)

// ================= GPU KERNEL =================
__global__ void matrixAdd(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

// ================= CPU FUNCTION =================
void matrixAddCPU(int *A, int *B, int *C, int n) {
    for (int i = 0; i < n * n; i++) {
        C[i] = A[i] + B[i];
    }
}

// ================= MAIN =================
int main() {

    size_t size = N * N * sizeof(int);

    // Allocate host memory
    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C_cpu = new int[N * N];
    int *h_C_gpu = new int[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // ================= CPU EXECUTION =================
    auto start_cpu = std::chrono::high_resolution_clock::now();

    matrixAddCPU(h_A, h_B, h_C_cpu, N);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // ================= GPU SETUP =================
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    // ================= GPU EXECUTION =================
    auto start_gpu = std::chrono::high_resolution_clock::now();

    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    // Copy back result
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // ================= VERIFY =================
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            correct = false;
            break;
        }
    }

    // ================= OUTPUT =================
    std::cout << "Result Correct: " << (correct ? "YES" : "NO") << std::endl;

    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time.count() << " ms" << std::endl;

    std::cout << "Sample Output: " << h_C_gpu[0] << std::endl;

    // ================= CLEANUP =================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}