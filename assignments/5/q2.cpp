#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

static size_t parse_size(int argc, char** argv) {
    if (argc > 1) {
        return static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    return 10000000ULL;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t n = parse_size(argc, argv);
    std::vector<double> buffer(n, 0.0);

    if (rank == 0) {
        std::fill(buffer.begin(), buffer.end(), 1.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    if (rank == 0) {
        for (int dest = 1; dest < size; ++dest) {
            MPI_Send(buffer.data(), static_cast<int>(n), MPI_DOUBLE, dest, 0,
                     MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(buffer.data(), static_cast<int>(n), MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double mybcast_time = t1 - t0;

    if (rank == 0) {
        std::fill(buffer.begin(), buffer.end(), 2.0);
    } else {
        std::fill(buffer.begin(), buffer.end(), 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t2 = MPI_Wtime();
    MPI_Bcast(buffer.data(), static_cast<int>(n), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t3 = MPI_Wtime();
    const double bcast_time = t3 - t2;

    if (rank == 0) {
        std::cout << "Array size: " << n << " doubles\n";
        std::cout << "MyBcast time: " << mybcast_time << " s\n";
        std::cout << "MPI_Bcast time: " << bcast_time << " s\n";
    }

    MPI_Finalize();
    return 0;
}
