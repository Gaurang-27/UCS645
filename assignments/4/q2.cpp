// Minimal Version: Distributed Array Summation
// Compile: mpicxx -O2 -o distributed_sum distributed_sum.cpp
// Run:     mpirun -np 4 ./distributed_sum

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int TOTAL_ELEMENTS = 100;

    int *global_array = NULL;

    int elements_per_process = TOTAL_ELEMENTS / nprocs;
    int extra_elements = TOTAL_ELEMENTS % nprocs;

    int chunk_size = elements_per_process +
                     (pid < extra_elements ? 1 : 0);

    int *local_chunk = (int*)malloc(chunk_size * sizeof(int));

    int *send_counts = NULL;
    int *displs = NULL;

    if (pid == 0) {

        global_array = (int*)malloc(TOTAL_ELEMENTS * sizeof(int));

        for (int i = 0; i < TOTAL_ELEMENTS; i++)
            global_array[i] = i + 1;

        send_counts = (int*)malloc(nprocs * sizeof(int));
        displs = (int*)malloc(nprocs * sizeof(int));

        int offset = 0;
        for (int i = 0; i < nprocs; i++) {
            send_counts[i] = elements_per_process +
                             (i < extra_elements ? 1 : 0);
            displs[i] = offset;
            offset += send_counts[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(global_array, send_counts, displs,
                 MPI_INT,
                 local_chunk, chunk_size,
                 MPI_INT,
                 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    int final_sum = 0;
    MPI_Reduce(&local_sum, &final_sum,
               1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    double exec_time = end - start;
    double Tp;

    MPI_Reduce(&exec_time, &Tp,
               1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("Computed Sum: %d\n", final_sum);
        printf("Execution Time (Tp): %.8f seconds\n", Tp);
    }

    MPI_Finalize();
    return 0;
}