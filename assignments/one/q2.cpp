#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

const int N = 1000;

int main()
{
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    vector<vector<double>> B(N, vector<double>(N, 2.0));
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    double serial_time;

    /* ---------------- SERIAL EXECUTION ---------------- */
    double start = omp_get_wtime();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    double end = omp_get_wtime();
    serial_time = end - start;

    cout << "Serial time: " << serial_time << " seconds\n\n";

    /* ---------------- 1D PARALLEL ---------------- */
    cout << "1D Parallel (parallel for)\n";
    cout << left
         << setw(10) << "Threads"
         << setw(15) << "Time(s)"
         << setw(15) << "Speedup"
         << endl;

    for (int threads = 1; threads <= 16; threads++)
    {
        // reset C to avoid cache bias
        for (auto &row : C)
            fill(row.begin(), row.end(), 0.0);

        start = omp_get_wtime();

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < N; k++)
                {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        end = omp_get_wtime();
        double time = end - start;

        cout << left
             << setw(10) << threads
             << setw(15) << fixed << setprecision(6) << time
             << setw(15) << fixed << setprecision(3) << (serial_time / time)
             << endl;
    }

    cout << "\n";

    /* ---------------- 2D PARALLEL ---------------- */
    cout << "2D Parallel (collapse(2))\n";
    cout << "Threads\tTime\t\tSpeedup\n";

    for (int threads = 1; threads <= 16; threads++)
    {
        for (auto &row : C)
            fill(row.begin(), row.end(), 0.0);

        start = omp_get_wtime();

        #pragma omp parallel for collapse(2) num_threads(threads)
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < N; k++)
                {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        end = omp_get_wtime();
        double time = end - start;

        cout << left
             << setw(10) << threads
             << setw(15) << fixed << setprecision(6) << time
             << setw(15) << fixed << setprecision(3) << (serial_time / time)
             << endl;
    }
}
