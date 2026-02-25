// Exercise 3: Distributed Min-Max (Clean Version)
// Compile: mpicxx -O2 -o ex3_clean ex3_clean.cpp
// Run:     mpirun -np 4 ./ex3_clean

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int VALUES_PER_PROC = 10;
    int data_block[VALUES_PER_PROC];

    srand(time(NULL) + pid);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_begin = MPI_Wtime();

    // Generate random values
    for (int i = 0; i < VALUES_PER_PROC; i++)
        data_block[i] = rand() % 1001;

    // Local min & max
    int local_high = data_block[0];
    int local_low  = data_block[0];

    for (int i = 1; i < VALUES_PER_PROC; i++) {
        if (data_block[i] > local_high) local_high = data_block[i];
        if (data_block[i] < local_low)  local_low  = data_block[i];
    }

    struct {
        int value;
        int owner;
    } send_max, recv_max,
      send_min, recv_min;

    send_max.value = local_high;
    send_max.owner = pid;

    send_min.value = local_low;
    send_min.owner = pid;

    // Global reductions
    MPI_Reduce(&send_max, &recv_max,
               1, MPI_2INT, MPI_MAXLOC,
               0, MPI_COMM_WORLD);

    MPI_Reduce(&send_min, &recv_min,
               1, MPI_2INT, MPI_MINLOC,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double local_runtime = t_end - t_begin;
    double Tp;

    MPI_Reduce(&local_runtime, &Tp,
               1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("Global Maximum: %d (Process %d)\n",
               recv_max.value, recv_max.owner);

        printf("Global Minimum: %d (Process %d)\n",
               recv_min.value, recv_min.owner);

        printf("Execution Time (Tp): %.8f seconds\n", Tp);
    }

    MPI_Finalize();
    return 0;
}