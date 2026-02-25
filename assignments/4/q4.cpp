// Clean Version: Parallel Dot Product
// Compile: mpicxx -O2 -o dot_clean dot_clean.cpp
// Run:     mpirun -np 4 ./dot_clean

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int VECTOR_SIZE = 8;

    if (VECTOR_SIZE % nprocs != 0) {
        if (pid == 0)
            printf("Vector size must be divisible by number of processes.\n");
        MPI_Finalize();
        return 0;
    }

    int portion = VECTOR_SIZE / nprocs;

    int vector_X[VECTOR_SIZE], vector_Y[VECTOR_SIZE];
    int sub_X[portion], sub_Y[portion];

    // Initialize vectors in root
    if (pid == 0) {
        int temp1[VECTOR_SIZE] = {1,2,3,4,5,6,7,8};
        int temp2[VECTOR_SIZE] = {8,7,6,5,4,3,2,1};

        for (int i = 0; i < VECTOR_SIZE; i++) {
            vector_X[i] = temp1[i];
            vector_Y[i] = temp2[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double time_start = MPI_Wtime();

    // Scatter data
    MPI_Scatter(vector_X, portion, MPI_INT,
                sub_X, portion, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(vector_Y, portion, MPI_INT,
                sub_Y, portion, MPI_INT,
                0, MPI_COMM_WORLD);

    // Local dot product
    int partial_result = 0;
    for (int i = 0; i < portion; i++)
        partial_result += sub_X[i] * sub_Y[i];

    int final_result = 0;
    MPI_Reduce(&partial_result, &final_result,
               1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double time_end = MPI_Wtime();

    double runtime = time_end - time_start;
    double Tp;

    MPI_Reduce(&runtime, &Tp,
               1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("Dot Product: %d\n", final_result);
        printf("Execution Time (Tp): %.8f seconds\n", Tp);
    }

    MPI_Finalize();
    return 0;
}