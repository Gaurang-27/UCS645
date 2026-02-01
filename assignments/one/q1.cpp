#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main() {
    const int N = 1 << 16;
    double a = 2.5;

    vector<double> X(N, 1.0), Y(N, 2.0);

    double serial_time;
    vector<pair<int,double>> time_taken;

    /* ---------------- SERIAL EXECUTION ---------------- */
    vector<double> X_serial = X;

    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        X_serial[i] = a * X_serial[i] + Y[i];
    }
    double end = omp_get_wtime();

    serial_time = end - start;

    cout << "Serial time: " << serial_time << "\n\n";

    /* ---------------- PARALLEL EXECUTION ---------------- */
    for (int threads = 1; threads <= 16; threads++) {

        vector<double> X_copy = X;

        start = omp_get_wtime();

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; i++) {
            X_copy[i] = a * X_copy[i] + Y[i];
        }

        end = omp_get_wtime();
        time_taken.push_back({threads, end - start});
    }
    
    /* ---------------- RESULTS ---------------- */
    cout << "Threads\tTime\t\tSpeedup\n";
    for (auto it : time_taken) {
        double speedup = serial_time / it.second;
        cout << it.first << "\t"
             << it.second << "\t"
             << speedup << endl;
    }
}
