// Parallel Molecular Dynamics using Lennard-Jones Interaction
// Clean-room implementation with OpenMP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <iomanip>

using std::cout;
using std::endl;
using std::vector;

struct Atom {
    double px, py, pz;   // Position
    double fx, fy, fz;   // Net force
};

// Lennard-Jones constants
constexpr double LJ_EPS = 1.0;
constexpr double LJ_SIG = 1.0;
constexpr double LJ_CUT = 2.5 * LJ_SIG;
constexpr double LJ_CUT2 = LJ_CUT * LJ_CUT;

// Compute Lennard-Jones force contribution
inline void compute_lj(
    const Atom& a,
    const Atom& b,
    double& fx,
    double& fy,
    double& fz,
    double& pe
) {
    double dx = a.px - b.px;
    double dy = a.py - b.py;
    double dz = a.pz - b.pz;

    double dist2 = dx*dx + dy*dy + dz*dz;

    if (dist2 > 0.0 && dist2 < LJ_CUT2) {
        double inv_r2 = 1.0 / dist2;
        double sig2 = LJ_SIG * LJ_SIG * inv_r2;
        double sig6 = sig2 * sig2 * sig2;
        double sig12 = sig6 * sig6;

        pe = 4.0 * LJ_EPS * (sig12 - sig6);

        double force_scale = 24.0 * LJ_EPS * inv_r2 * (2.0 * sig12 - sig6);

        fx = force_scale * dx;
        fy = force_scale * dy;
        fz = force_scale * dz;
    } else {
        fx = fy = fz = pe = 0.0;
    }
}

int main() {
    const int num_atoms = 1000;
    vector<Atom> atoms(num_atoms);

    // Random initialization
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> uni(0.0, 100.0);

    for (auto& a : atoms) {
        a.px = uni(rng);
        a.py = uni(rng);
        a.pz = uni(rng);
        a.fx = a.fy = a.fz = 0.0;
    }

    vector<int> threads_to_test = {1, 2, 4, 6, 8, 12, 14,16};

    cout << "\nParallel Lennard-Jones Simulation\n";
    cout << "Atoms: " << num_atoms << "\n";
    cout << "Cutoff radius: " << LJ_CUT << "\n\n";

    cout << std::left
         << std::setw(10) << "Threads"
         << std::setw(14) << "Time(s)"
         << std::setw(14) << "Speedup"
         << std::setw(14) << "Efficiency"
         << "Potential\n";

    cout << std::string(70, '-') << endl;

    double baseline_time = 0.0;

    for (int tcount : threads_to_test) {
        if (tcount > omp_get_max_threads()) continue;

        for (auto& a : atoms) {
            a.fx = a.fy = a.fz = 0.0;
        }

        double total_energy = 0.0;
        double t0 = omp_get_wtime();

        #pragma omp parallel num_threads(tcount)
        {
            double local_energy = 0.0;

            #pragma omp for schedule(dynamic, 32)
            for (int i = 0; i < num_atoms; ++i) {
                double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;

                for (int j = 0; j < num_atoms; ++j) {
                    if (i == j) continue;

                    double fx, fy, fz, pe;
                    compute_lj(atoms[i], atoms[j], fx, fy, fz, pe);

                    fx_sum += fx;
                    fy_sum += fy;
                    fz_sum += fz;
                    local_energy += pe;
                }

                atoms[i].fx = fx_sum;
                atoms[i].fy = fy_sum;
                atoms[i].fz = fz_sum;
            }

            #pragma omp atomic
            total_energy += local_energy;
        }

        double t1 = omp_get_wtime();
        double elapsed = t1 - t0;

        total_energy *= 0.5;  // Correct double counting

        if (tcount == 1) baseline_time = elapsed;

        double speedup = baseline_time / elapsed;
        double efficiency = (speedup / tcount) * 100.0;

        cout << std::setw(10) << tcount
             << std::setw(14) << std::fixed << std::setprecision(6) << elapsed
             << std::setw(14) << std::setprecision(2) << speedup
             << std::setw(13) << std::setprecision(1) << efficiency << "%"
             << std::setprecision(2) << total_energy << endl;
    }

    cout << std::string(70, '-') << endl;


    return 0;
}
