// DNA Sequence Local Alignment using Smith-Waterman Algorithm
// Parallel Computing Implementation with OpenMP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#include <ctime>

using namespace std;

// Alignment scoring scheme
const int SCORE_MATCH = 2;
const int SCORE_MISMATCH = -1;
const int PENALTY_GAP = -2;

// Function to create random DNA sequences
string createRandomSequence(int len, unsigned int randomSeed) {
    srand(randomSeed);
    string sequence = "";
    char nucleotides[] = {'A', 'T', 'G', 'C'};
    
    for (int idx = 0; idx < len; idx++) {
        int randIndex = rand() % 4;
        sequence += nucleotides[randIndex];
    }
    return sequence;
}

// Compute alignment score between two nucleotides
int computeScore(char nucleotide1, char nucleotide2) {
    if (nucleotide1 == nucleotide2)
        return SCORE_MATCH;
    else
        return SCORE_MISMATCH;
}

// Anti-diagonal wavefront approach for parallel alignment
void alignSequencesWavefront(const string& dna1, const string& dna2, 
                             int threadCount, double& executionTime) {
    int len1 = dna1.size();
    int len2 = dna2.size();
    
    // Allocate scoring matrix
    vector<vector<int>> scoreMatrix(len1 + 1, vector<int>(len2 + 1, 0));
    
    double startClock = omp_get_wtime();
    
    // Process matrix using diagonal wavefront
    int totalDiagonals = len1 + len2 - 1;
    for (int diagonal = 1; diagonal <= totalDiagonals; diagonal++) {
        int rowStart = max(1, diagonal - len2 + 1);
        int rowEnd = min(len1, diagonal);
        
        #pragma omp parallel for num_threads(threadCount) schedule(dynamic, 4)
        for (int row = rowStart; row <= rowEnd; row++) {
            int col = diagonal - row + 1;
            if (col >= 1 && col <= len2) {
                int scoreMatch = scoreMatrix[row-1][col-1] + 
                                computeScore(dna1[row-1], dna2[col-1]);
                int scoreDeletion = scoreMatrix[row-1][col] + PENALTY_GAP;
                int scoreInsertion = scoreMatrix[row][col-1] + PENALTY_GAP;
                
                int maxScore = max({0, scoreMatch, scoreDeletion, scoreInsertion});
                scoreMatrix[row][col] = maxScore;
            }
        }
    }
    
    double endClock = omp_get_wtime();
    executionTime = endClock - startClock;
    
    // Locate best alignment score
    int optimalScore = 0;
    for (int r = 0; r <= len1; r++) {
        for (int c = 0; c <= len2; c++) {
            if (scoreMatrix[r][c] > optimalScore)
                optimalScore = scoreMatrix[r][c];
        }
    }
}

// Row-based parallelization approach
void alignSequencesRowBased(const string& dna1, const string& dna2,
                            int threadCount, double& executionTime) {
    int len1 = dna1.size();
    int len2 = dna2.size();
    
    vector<vector<int>> dpTable(len1 + 1, vector<int>(len2 + 1, 0));
    
    double t0 = omp_get_wtime();
    
    // Fill matrix row by row
    for (int r = 1; r <= len1; r++) {
        #pragma omp parallel for num_threads(threadCount) schedule(static)
        for (int c = 1; c <= len2; c++) {
            int alignScore = dpTable[r-1][c-1] + computeScore(dna1[r-1], dna2[c-1]);
            int delScore = dpTable[r-1][c] + PENALTY_GAP;
            int insScore = dpTable[r][c-1] + PENALTY_GAP;
            
            dpTable[r][c] = max({0, alignScore, delScore, insScore});
        }
    }
    
    double t1 = omp_get_wtime();
    executionTime = t1 - t0;
}

void printResults(vector<int>& threads, vector<double>& times, string method) {
    cout << "\n=== " << method << " ===" << endl;
    cout << left << setw(12) << "Threads" << setw(18) << "Execution Time"
         << setw(18) << "Speedup Factor" << "Parallel Efficiency" << endl;
    cout << string(60, '-') << endl;
    
    double baseTime = times[0];
    
    for (size_t i = 0; i < threads.size(); i++) {
        double speedupRatio = baseTime / times[i];
        double parallelEfficiency = (speedupRatio / threads[i]) * 100.0;
        
        cout << left << setw(12) << threads[i]
             << setw(18) << fixed << setprecision(6) << times[i]
             << setw(18) << setprecision(3) << speedupRatio << "x"
             << setprecision(2) << parallelEfficiency << "%" << endl;
    }
}

int main() {
    // Create DNA sequences for testing
    const int LENGTH = 500;
    string sequence1 = createRandomSequence(LENGTH, 42);
    string sequence2 = createRandomSequence(LENGTH, 123);
    
    // Insert common region to simulate biological similarity
    int commonStart = LENGTH / 3;
    int commonLength = 50;
    for (int i = 0; i < commonLength; i++) {
        sequence2[commonStart + i] = sequence1[commonStart + i];
    }
    
    cout << "Smith-Waterman Local Alignment Algorithm" << endl;
    cout << "Length of Sequence 1: " << sequence1.length() << endl;
    cout << "Length of Sequence 2: " << sequence2.length() << endl;
    cout << "Match Score: " << SCORE_MATCH << " | Mismatch Penalty: " 
         << SCORE_MISMATCH << " | Gap Penalty: " << PENALTY_GAP << "\n" << endl;
    
    vector<int> numThreads = {1, 2, 4, 6,8,10,12,14,16};
    int maxAvailableThreads = omp_get_max_threads();
    
    // Test wavefront method
    vector<double> wavefrontTimes;
    for (int t : numThreads) {
        if (t > maxAvailableThreads) continue;
        
        double execTime;
        alignSequencesWavefront(sequence1, sequence2, t, execTime);
        wavefrontTimes.push_back(execTime);
    }
    printResults(numThreads, wavefrontTimes, "Wavefront Diagonal Method");
    
    // Test row-based method
    vector<double> rowTimes;
    for (int t : numThreads) {
        if (t > maxAvailableThreads) continue;
        
        double execTime;
        alignSequencesRowBased(sequence1, sequence2, t, execTime);
        rowTimes.push_back(execTime);
    }
    printResults(numThreads, rowTimes, "Row-Based Parallelization");
    
 
    return 0;
}