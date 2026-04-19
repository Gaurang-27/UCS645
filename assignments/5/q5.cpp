#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static int parse_max(int argc, char** argv) {
    if (argc > 1) {
        return std::atoi(argv[1]);
    }
    return 10000;
}

static bool is_perfect(int n) {
    if (n < 2) {
        return false;
    }
    int sum = 1;
    const int limit = static_cast<int>(std::sqrt(n));
    for (int i = 2; i <= limit; ++i) {
        if (n % i == 0) {
            sum += i;
            const int other = n / i;
            if (other != i) {
                sum += other;
            }
        }
    }
    return sum == n;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cout << "Run with at least 2 processes.\n";
        }
        MPI_Finalize();
        return 0;
    }

    const int max_n = parse_max(argc, argv);

    if (rank == 0) {
        int next = 2;
        int finished = 0;
        std::vector<int> perfects;

        while (finished < size - 1) {
            int msg = 0;
            MPI_Status status;
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                     &status);

            if (msg > 0) {
                perfects.push_back(msg);
            }

            const int src = status.MPI_SOURCE;
            if (next <= max_n) {
                MPI_Send(&next, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                ++next;
            } else {
                const int stop = 0;
                MPI_Send(&stop, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                ++finished;
            }
        }

        std::sort(perfects.begin(), perfects.end());
        std::cout << "Perfect numbers up to " << max_n << ":\n";
        for (size_t i = 0; i < perfects.size(); ++i) {
            std::cout << perfects[i]
                      << (i + 1 == perfects.size() ? "\n" : " ");
        }
        std::cout << "Total perfect numbers: " << perfects.size() << "\n";
    } else {
        int request = 0;
        MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (true) {
            int value = 0;
            MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            if (value == 0) {
                break;
            }

            const int result = is_perfect(value) ? value : -value;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
