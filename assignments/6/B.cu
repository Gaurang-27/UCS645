#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N (1 << 24)   // ~16 million elements

// ================= GPU KERNEL =================
__global__ void sumKernel(float *input, float *output, int n) {
    __shared__ float sharedData[256];

    int tid = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sharedData[tid] = (globalId < n) ? input[globalId] : 0.0f;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Store result per block
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// ================= CPU FUNCTION =================
float cpuSum(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// ================= MAIN =================
int main() {

    // Allocate host memory
    float *h_input = new float[N];

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // ================= CPU EXECUTION =================
    auto start_cpu = std::chrono::high_resolution_clock::now();

    float cpu_result = cpuSum(h_input, N);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // ================= GPU SETUP =================
    float *d_input, *d_output;

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_output = new float[blocks];

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // ================= GPU EXECUTION =================
    auto start_gpu = std::chrono::high_resolution_clock::now();

    sumKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();  // IMPORTANT for timing

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    // Copy result back
    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float gpu_result = 0.0f;
    for (int i = 0; i < blocks; i++) {
        gpu_result += h_output[i];
    }

    // ================= OUTPUT =================
    std::cout << "CPU Sum: " << cpu_result << std::endl;
    std::cout << "GPU Sum: " << gpu_result << std::endl;

    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time.count() << " ms" << std::endl;

    // ================= CLEANUP =================
    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}