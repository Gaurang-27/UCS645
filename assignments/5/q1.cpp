#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

static size_t parse_size(int argc, char** argv) {
    if (argc > 1) {
        return static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    return static_cast<size_t>(1) << 16;
}

static double parse_scalar(int argc, char** argv) {
    if (argc > 2) {
        return std::strtod(argv[2], nullptr);
    }
    return 2.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t n = parse_size(argc, argv);
    const double a = parse_scalar(argc, argv);

    double serial_time = 0.0;
    double serial_checksum = 0.0;

    if (rank == 0) {
        std::vector<double> x(n, 1.0);
        std::vector<double> y(n, 2.0);

        const double t0 = MPI_Wtime();
        for (size_t i = 0; i < n; ++i) {
            x[i] = a * x[i] + y[i];
        }
        const double t1 = MPI_Wtime();

        serial_time = t1 - t0;
        serial_checksum = std::accumulate(x.begin(), x.end(), 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t local_n = base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    std::vector<double> local_x(local_n, 1.0);
    std::vector<double> local_y(local_n, 2.0);

    const double p0 = MPI_Wtime();
    for (size_t i = 0; i < local_n; ++i) {
        local_x[i] = a * local_x[i] + local_y[i];
    }
    const double p1 = MPI_Wtime();

    const double local_time = p1 - p0;
    const double local_checksum =
        std::accumulate(local_x.begin(), local_x.end(), 0.0);

    double parallel_time = 0.0;
    double parallel_checksum = 0.0;
    MPI_Reduce(&local_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_checksum, &parallel_checksum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const double speedup = serial_time / parallel_time;
        std::cout << "DAXPY size: " << n << "\n";
        std::cout << "Serial time: " << serial_time << " s, checksum: "
                  << serial_checksum << "\n";
        std::cout << "Parallel time (max): " << parallel_time
                  << " s, checksum: " << parallel_checksum << "\n";
        std::cout << "Speedup: " << speedup << "\n";
    }

    MPI_Finalize();
    return 0;
}
