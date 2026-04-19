#include <mpi.h>

#include <cstdlib>
#include <iostream>

static long long parse_total(int argc, char** argv) {
    if (argc > 1) {
        return std::strtoll(argv[1], nullptr, 10);
    }
    return 500000000LL;
}

static double parse_multiplier(int argc, char** argv) {
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

    const long long total_n = parse_total(argc, argv);
    double multiplier = 0.0;
    if (rank == 0) {
        multiplier = parse_multiplier(argc, argv);
    }
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const long long base = total_n / size;
    const long long rem = total_n % size;
    const long long local_n = base + (rank < rem ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    double local_sum = 0.0;
    const double a_value = 1.0;
    const double b_value = 2.0 * multiplier;
    for (long long i = 0; i < local_n; ++i) {
        local_sum += a_value * b_value;
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Total elements: " << total_n << "\n";
        std::cout << "Multiplier: " << multiplier << "\n";
        std::cout << "Dot product: " << global_sum << "\n";
        std::cout << "Total time: " << (t1 - t0) << " s\n";
    }

    MPI_Finalize();
    return 0;
}
