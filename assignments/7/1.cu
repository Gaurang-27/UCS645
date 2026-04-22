#include <iostream>
#include <cuda.h>
#include <chrono>

using namespace std;

#define N 1024

// ---------------- GPU Kernels ----------------

// Task A: Iterative sum
__global__ void sum_iterative(int *result) {
    int tid = threadIdx.x;

    if (tid == 0) {
        int sum = 0;
        for (int i = 1; i <= N; i++) {
            sum += i;
        }
        result[0] = sum;
    }
}

// Task B: Direct formula
__global__ void sum_formula(int *result) {
    int tid = threadIdx.x;

    if (tid == 1) {
        result[1] = (N * (N + 1)) / 2;
    }
}

// ---------------- CPU Functions ----------------

int cpu_iterative() {
    int sum = 0;
    for (int i = 1; i <= N; i++) {
        sum += i;
    }
    return sum;
}

int cpu_formula() {
    return (N * (N + 1)) / 2;
}

// ---------------- MAIN ----------------

int main() {

    int h_result[2];   // host output
    int *d_result;     // device output

    // Step 3: Allocate memory on GPU
    cudaMalloc((void**)&d_result, 2 * sizeof(int));

    // ---------------- CPU Timing ----------------
    auto start_cpu = chrono::high_resolution_clock::now();

    int cpu_sum1 = cpu_iterative();
    int cpu_sum2 = cpu_formula();

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_time = end_cpu - start_cpu;

    // ---------------- GPU Timing ----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Step 6: Launch kernel with 1 block, 2 threads
    sum_iterative<<<1, 2>>>(d_result);
    sum_formula<<<1, 2>>>(d_result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Step 5: Copy result back
    cudaMemcpy(h_result, d_result, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------- OUTPUT ----------------

    cout << "CPU Iterative Sum: " << cpu_sum1 << endl;
    cout << "CPU Formula Sum:   " << cpu_sum2 << endl;
    cout << "CPU Time (ms):     " << cpu_time.count() << endl;

    cout << "GPU Iterative Sum: " << h_result[0] << endl;
    cout << "GPU Formula Sum:   " << h_result[1] << endl;
    cout << "GPU Time (ms):     " << gpu_time << endl;

    // Cleanup
    cudaFree(d_result);

    return 0;
}