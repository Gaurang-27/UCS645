// Compile: mpicxx -O2 -o ring_simple ring_simple.cpp
// Run:     mpirun -np 4 ./ring_simple

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int next = (pid + 1) % nprocs;
    int prev = (pid - 1 + nprocs) % nprocs;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Heavy computation
    long long work = 200000000;
    long long sum = 0;

    for (long long i = 0; i < work; i++) {
        sum += (i % 7) * (i % 5);
    }

    int data;

    if (pid == 0) {
        data = 100 + (sum % 1000);
        MPI_Send(&data, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&data, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += (sum % 1000);
        MPI_Send(&data, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double exec_time = t_end - t_start;

    // Get maximum execution time (Tp)
    double Tp;
    MPI_Reduce(&exec_time, &Tp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("Execution Time (Tp) with %d processes: %.6f seconds\n", 
               nprocs, Tp);
    }

    MPI_Finalize();
    return 0;
}