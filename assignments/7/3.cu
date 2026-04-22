#include <iostream>
#include <cuda.h>

using namespace std;

#define N 1024  // compile-time size

// ================= STATIC GLOBAL DEVICE MEMORY =================
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

// ================= KERNEL =================
__global__ void vectorAddKernel() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

// ================= MAIN =================
int main() {

    float h_A[N], h_B[N], h_C[N];

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Copy to device symbols
    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    // ================= TIMING =================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(start);

    vectorAddKernel<<<blocks, threads>>>();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    // Copy result back
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    // ================= PRINT RESULT =================
    cout << "Sample Output: ";
    for (int i = 0; i < 5; i++) {
        cout << h_C[i] << " ";
    }
    cout << endl;

    cout << "Kernel Time (ms): " << time_ms << endl;

    // ================= DEVICE PROPERTIES =================
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float memoryClock = prop.memoryClockRate;   // KHz
    float memoryBusWidth = prop.memoryBusWidth; // bits

    // Theoretical BW calculation
    float theoreticalBW = 2.0f * memoryClock * (memoryBusWidth / 8.0f); 
    // now in KB/s

    theoreticalBW /= 1e6; // convert to GB/s

    cout << "Theoretical Bandwidth (GB/s): " << theoreticalBW << endl;

    // ================= MEASURED BANDWIDTH =================
    float RBytes = 2 * N * sizeof(float); // read A & B
    float WBytes = N * sizeof(float);     // write C

    float totalBytes = RBytes + WBytes;

    float time_sec = time_ms / 1000.0f;

    float measuredBW = totalBytes / time_sec; // Bytes/sec
    measuredBW /= 1e9; // convert to GB/s

    cout << "Measured Bandwidth (GB/s): " << measuredBW << endl;

    return 0;
}