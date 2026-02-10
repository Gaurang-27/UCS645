#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main()
{
    static long long num_steps = 1000000000;
    double step = 1.0 / (double)num_steps;

    double start, end;
    double serial_time;

    /* -------- SERIAL EXECUTION -------- */
    double sum = 0.0;
    start = omp_get_wtime();

    for (long i = 0; i < num_steps; i++)
    {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;   
    end = omp_get_wtime();
    serial_time = end - start;

    cout << fixed << setprecision(9)
     << "Serial time: " << serial_time << " s\n\n";

    cout << left
         << setw(10) << "Threads"
         << setw(15) << "Time(s)"
         << setw(15) << "Speedup"
         << endl;

    /* -------- PARALLEL EXECUTION -------- */
    for (int threads = 1; threads <= 16; threads++)
    {
        double sum = 0.0;   // reset for each run

        omp_set_num_threads(threads);
        start = omp_get_wtime();

        #pragma omp parallel for reduction(+ : sum)
        for (long i = 0; i < num_steps; i++)
        {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        end = omp_get_wtime();
        double time = end - start;

        cout << left
             << setw(10) << threads
             << setw(15) << fixed << setprecision(6) << time
             << setw(15) << fixed << setprecision(9) << (serial_time / time)
             << endl;
    }

    cout<<endl<<"Pi : "<<pi<<endl;

    

    return 0;
}
