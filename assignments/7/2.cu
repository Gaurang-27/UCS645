#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <chrono>

using namespace std;
#define N 1000

// ================= CPU MERGE SORT (PIPELINED USING OMP) =================

void merge_cpu(int arr[], int l, int m, int r) {
    int i = l, j = m + 1, k = 0;
    int temp[r - l + 1];

    while (i <= m && j <= r) {
        if (arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];

    for (i = l, k = 0; i <= r; i++, k++)
        arr[i] = temp[k];
}

void mergeSort_cpu(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort_cpu(arr, l, m);

            #pragma omp section
            mergeSort_cpu(arr, m + 1, r);
        }

        merge_cpu(arr, l, m, r);
    }
}

// ================= CUDA MERGE SORT =================

__device__ void merge_gpu(int *arr, int left, int mid, int right) {
    int i = left, j = mid + 1, k = 0;

    int temp[1024];  // enough for N=1000

    while (i <= mid && j <= right) {
        if (arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (i = left, k = 0; i <= right; i++, k++)
        arr[i] = temp[k];
}

__global__ void mergeSortKernel(int *arr, int size, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int left = tid * 2 * width;

    if (left < size) {
        int mid = min(left + width - 1, size - 1);
        int right = min(left + 2 * width - 1, size - 1);

        merge_gpu(arr, left, mid, right);
    }
}

// ================= MAIN =================

int main() {
    int arr[N], arr_cpu[N];

    // Step 4: Initialize array
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000;
        arr_cpu[i] = arr[i];
    }

    // ================= CPU TIMING =================
    auto start_cpu = chrono::high_resolution_clock::now();

    mergeSort_cpu(arr_cpu, 0, N - 1);

    auto end_cpu = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(end_cpu - start_cpu).count();

    // ================= GPU SETUP =================
    int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ================= GPU TIMING =================
    cudaEventRecord(start);

    for (int width = 1; width < N; width *= 2) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        mergeSortKernel<<<blocks, threads>>>(d_arr, N, width);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // ================= OUTPUT =================

    cout << "CPU Time (ms): " << cpu_time << endl;
    cout << "GPU Time (ms): " << gpu_time << endl;

    // (Optional) verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (arr[i] != arr_cpu[i]) {
            correct = false;
            break;
        }
    }

    if (correct) cout << "Sorting Correct ✅" << endl;
    else cout << "Sorting Incorrect ❌" << endl;

    cudaFree(d_arr);

    return 0;
}