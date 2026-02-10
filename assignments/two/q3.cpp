/* 
 * 2D Heat Equation Solver using Finite Difference Method
 * Implements parallel computation with OpenMP
 */

#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Simulation constants
#define THERMAL_DIFF 0.01
#define STEP_X 0.1
#define STEP_Y 0.1
#define STEP_T 0.001
#define GRID_DIM 512
#define TIME_ITERATIONS 100

typedef std::vector<std::vector<double>> Grid2D;

class HeatSimulator {
private:
    int gridSize;
    int iterations;
    Grid2D currentState;
    Grid2D nextState;
    
    void initializeGrid() {
        int mid = gridSize / 2;
        int hotZone = gridSize / 10;
        
        for(int row = 0; row < gridSize; row++) {
            for(int col = 0; col < gridSize; col++) {
                double distance = std::sqrt((row - mid) * (row - mid) + 
                                           (col - mid) * (col - mid));
                currentState[row][col] = (distance < hotZone) ? 100.0 : 0.0;
                nextState[row][col] = 0.0;
            }
        }
    }
    
    double calculateAverageTemperature(const Grid2D& grid) {
        double sum = 0.0;
        for(int i = 0; i < gridSize; i++) {
            for(int j = 0; j < gridSize; j++) {
                sum += grid[i][j];
            }
        }
        return sum / (gridSize * gridSize);
    }
    
public:
    HeatSimulator(int size, int steps) : gridSize(size), iterations(steps) {
        currentState.resize(size, std::vector<double>(size, 0.0));
        nextState.resize(size, std::vector<double>(size, 0.0));
        initializeGrid();
    }
    
    double runSimulation(int numThreads, std::string schedType, double* avgTemp) {
        double startTimer = omp_get_wtime();
        double heatSum;
        
        for(int step = 0; step < iterations; step++) {
            heatSum = 0.0;
            
            if(schedType == "static") {
                #pragma omp parallel for collapse(2) schedule(static) \
                        num_threads(numThreads) reduction(+:heatSum)
                for(int i = 1; i < gridSize - 1; i++) {
                    for(int j = 1; j < gridSize - 1; j++) {
                        double laplacian = (currentState[i+1][j] - 2*currentState[i][j] + 
                                          currentState[i-1][j]) / (STEP_X * STEP_X) +
                                         (currentState[i][j+1] - 2*currentState[i][j] + 
                                          currentState[i][j-1]) / (STEP_Y * STEP_Y);
                        nextState[i][j] = currentState[i][j] + THERMAL_DIFF * STEP_T * laplacian;
                        heatSum += nextState[i][j];
                    }
                }
            }
            else if(schedType == "dynamic") {
                #pragma omp parallel for collapse(2) schedule(dynamic, 16) \
                        num_threads(numThreads) reduction(+:heatSum)
                for(int i = 1; i < gridSize - 1; i++) {
                    for(int j = 1; j < gridSize - 1; j++) {
                        double laplacian = (currentState[i+1][j] - 2*currentState[i][j] + 
                                          currentState[i-1][j]) / (STEP_X * STEP_X) +
                                         (currentState[i][j+1] - 2*currentState[i][j] + 
                                          currentState[i][j-1]) / (STEP_Y * STEP_Y);
                        nextState[i][j] = currentState[i][j] + THERMAL_DIFF * STEP_T * laplacian;
                        heatSum += nextState[i][j];
                    }
                }
            }
            else {
                #pragma omp parallel for collapse(2) schedule(guided) \
                        num_threads(numThreads) reduction(+:heatSum)
                for(int i = 1; i < gridSize - 1; i++) {
                    for(int j = 1; j < gridSize - 1; j++) {
                        double laplacian = (currentState[i+1][j] - 2*currentState[i][j] + 
                                          currentState[i-1][j]) / (STEP_X * STEP_X) +
                                         (currentState[i][j+1] - 2*currentState[i][j] + 
                                          currentState[i][j-1]) / (STEP_Y * STEP_Y);
                        nextState[i][j] = currentState[i][j] + THERMAL_DIFF * STEP_T * laplacian;
                        heatSum += nextState[i][j];
                    }
                }
            }
            
            std::swap(currentState, nextState);
        }
        
        *avgTemp = calculateAverageTemperature(currentState);
        return omp_get_wtime() - startTimer;
    }
    
    double runBlockedSimulation(int numThreads, double* avgTemp) {
        const int blockDim = 32;
        double startTimer = omp_get_wtime();
        
        for(int step = 0; step < iterations; step++) {
            #pragma omp parallel for collapse(2) schedule(static) num_threads(numThreads)
            for(int blockI = 1; blockI < gridSize - 1; blockI += blockDim) {
                for(int blockJ = 1; blockJ < gridSize - 1; blockJ += blockDim) {
                    int endI = std::min(blockI + blockDim, gridSize - 1);
                    int endJ = std::min(blockJ + blockDim, gridSize - 1);
                    
                    for(int i = blockI; i < endI; i++) {
                        for(int j = blockJ; j < endJ; j++) {
                            double laplacian = (currentState[i+1][j] - 2*currentState[i][j] + 
                                              currentState[i-1][j]) / (STEP_X * STEP_X) +
                                             (currentState[i][j+1] - 2*currentState[i][j] + 
                                              currentState[i][j-1]) / (STEP_Y * STEP_Y);
                            nextState[i][j] = currentState[i][j] + THERMAL_DIFF * STEP_T * laplacian;
                        }
                    }
                }
            }
            std::swap(currentState, nextState);
        }
        
        *avgTemp = calculateAverageTemperature(currentState);
        return omp_get_wtime() - startTimer;
    }
};

void printHeader() {
    std::cout << "Heat Diffusion - 2D Finite Difference Simulation\n";
    std::cout << "Grid: " << GRID_DIM << "x" << GRID_DIM << "\n";
    std::cout << "Timesteps: " << TIME_ITERATIONS << "\n";
    std::cout << "Stability: " << (THERMAL_DIFF * STEP_T / (STEP_X * STEP_X)) 
              << " (threshold: 0.25)\n\n";
}

void benchmarkScheduling() {
    std::string schedules[] = {"static", "dynamic", "guided"};
    int threadOptions[] = {1, 2, 4, 8,12,16};
    
    for(auto& sched : schedules) {
        std::cout << "[ " << sched << " scheduling ]\n";
        std::cout << std::left << std::setw(10) << "Threads" 
                  << std::setw(15) << "Runtime(s)"
                  << std::setw(15) << "Speedup" 
                  << std::setw(15) << "Efficiency"
                  << "AvgTemp\n";
        std::cout << std::string(70, '-') << "\n";
        
        double baselineTime = 0.0;
        
        for(auto threads : threadOptions) {
            if(threads > omp_get_max_threads()) continue;
            
            HeatSimulator sim(GRID_DIM, TIME_ITERATIONS);
            double temp;
            double runtime = sim.runSimulation(threads, sched, &temp);
            
            if(threads == 1) baselineTime = runtime;
            
            double speedup = baselineTime / runtime;
            double efficiency = (speedup / threads) * 100.0;
            
            std::cout << std::left << std::setw(10) << threads
                     << std::setw(15) << std::fixed << std::setprecision(6) << runtime
                     << std::setw(15) << std::setprecision(2) << speedup << "x"
                     << std::setw(15) << std::setprecision(1) << efficiency << "%"
                     << std::setprecision(2) << temp << "°C\n";
        }
        std::cout << "\n";
    }
}

void benchmarkBlocking() {
    int threadOptions[] = {1, 2, 4, 8,12,16};
    
    std::cout << "[ Cache-optimized blocking (32x32) ]\n";
    std::cout << std::left << std::setw(10) << "Threads" 
              << std::setw(15) << "Runtime(s)"
              << std::setw(15) << "Speedup" 
              << "Efficiency\n";
    std::cout << std::string(55, '-') << "\n";
    
    double baselineTime = 0.0;
    
    for(auto threads : threadOptions) {
        if(threads > omp_get_max_threads()) continue;
        
        HeatSimulator sim(GRID_DIM, TIME_ITERATIONS);
        double temp;
        double runtime = sim.runBlockedSimulation(threads, &temp);
        
        if(threads == 1) baselineTime = runtime;
        
        double speedup = baselineTime / runtime;
        double efficiency = (speedup / threads) * 100.0;
        
        std::cout << std::left << std::setw(10) << threads
                 << std::setw(15) << std::fixed << std::setprecision(6) << runtime
                 << std::setw(15) << std::setprecision(2) << speedup << "x"
                 << std::setprecision(1) << efficiency << "%\n";
    }
}


int main() {
    printHeader();
    benchmarkScheduling();
    benchmarkBlocking();
    
    return 0;
}