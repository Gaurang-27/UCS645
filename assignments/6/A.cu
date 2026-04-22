#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 0;
    }

    std::cout << "Number of CUDA Devices: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "===== Device " << i << " =====\n";
        std::cout << "Name: " << prop.name << "\n";

        std::cout << "Compute Capability: "
                  << prop.major << "." << prop.minor << "\n";

        std::cout << "Total Global Memory: "
                  << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

        std::cout << "Shared Memory per Block: "
                  << prop.sharedMemPerBlock / 1024 << " KB\n";

        std::cout << "Constant Memory: "
                  << prop.totalConstMem / 1024 << " KB\n";

        std::cout << "Registers per Block: "
                  << prop.regsPerBlock << "\n";

        std::cout << "Warp Size: "
                  << prop.warpSize << "\n";

        std::cout << "Max Threads per Block: "
                  << prop.maxThreadsPerBlock << "\n";

        std::cout << "Max Threads per Multiprocessor: "
                  << prop.maxThreadsPerMultiProcessor << "\n";

        std::cout << "Max Block Dimensions: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";

        std::cout << "Max Grid Dimensions: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";

        std::cout << "Multiprocessor Count: "
                  << prop.multiProcessorCount << "\n";

        std::cout << "Clock Rate: "
                  << prop.clockRate / 1000 << " MHz\n";

        std::cout << "Memory Clock Rate: "
                  << prop.memoryClockRate / 1000 << " MHz\n";

        std::cout << "Memory Bus Width: "
                  << prop.memoryBusWidth << " bits\n";

        std::cout << "Concurrent Kernels: "
                  << prop.concurrentKernels << "\n";

        std::cout << "Unified Addressing: "
                  << prop.unifiedAddressing << "\n";

        std::cout << "Double Precision Support: "
                  << (prop.major >= 1 ? "Yes" : "No") << "\n";

        std::cout << "\n";
    }

    return 0;
}