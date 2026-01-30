#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main() {
    const int N = 1 << 16;
    double a = 2.5;

    vector<double> X(N, 1.0), Y(N, 2.0);
    vector<pair<int,double>> time_taken;

    for (int threads = 2; threads <= 20; threads += 2) {

        vector<double> X_copy = X;   

        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; i++) {
            X_copy[i] = a * X_copy[i] + Y[i];
        }

        double end = omp_get_wtime();
        time_taken.push_back({threads,end-start});
    }

    for (auto it : time_taken)
        cout <<"threads : "<<it.first<<"  time: "<<it.second << endl;
}
