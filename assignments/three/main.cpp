/*
 * main.cpp - UCS645 Assignment 3
 * Performance Analysis (Refactored)
 * Usage: ./correlate <ny> <nx>
 */

#include "correlate.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <sys/resource.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using HighClock = std::chrono::high_resolution_clock;
using TimeSpan  = std::chrono::duration<double>;

// ──────────────────────────────────────────────────────────────────────────
// Data Generation
// ──────────────────────────────────────────────────────────────────────────
static std::vector<float> create_input_data(int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> buffer((size_t)rows * cols);
    for (auto &x : buffer) x = dist(gen);
    return buffer;
}

// ──────────────────────────────────────────────────────────────────────────
// Performance Container
// ──────────────────────────────────────────────────────────────────────────
struct PerfStats {
    int threads_used;
    double wall_time;
    double proc_time;
    double utilization;
    double cpi;
    long page_faults;
};

// ──────────────────────────────────────────────────────────────────────────
// Benchmark Runner
// ──────────────────────────────────────────────────────────────────────────
static PerfStats run_benchmark(
    int ny, int nx,
    const std::vector<float>& input,
    std::vector<float>& output,
    int threads,
    int hw_threads)
{
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    std::fill(output.begin(), output.end(), 0.0f);

    struct rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);

    auto t0 = HighClock::now();
    correlate(ny, nx, input.data(), output.data());
    auto t1 = HighClock::now();

    getrusage(RUSAGE_SELF, &after);

    double elapsed = TimeSpan(t1 - t0).count();

    double cpu =
        (after.ru_utime.tv_sec  - before.ru_utime.tv_sec) +
        (after.ru_utime.tv_usec - before.ru_utime.tv_usec) * 1e-6 +
        (after.ru_stime.tv_sec  - before.ru_stime.tv_sec) +
        (after.ru_stime.tv_usec - before.ru_stime.tv_usec) * 1e-6;

    long faults =
        (after.ru_minflt - before.ru_minflt) +
        (after.ru_majflt - before.ru_majflt);

    int active = std::min(threads, hw_threads);
    double util = (elapsed > 0) ? (cpu / (elapsed * active)) * 100.0 : 100.0;

    double cycles = elapsed * 3.0e9;
    double ops = (double)ny * (ny + 1) / 2.0 * nx * 4.0;
    double cpi_est = cycles / ops;

    return {threads, elapsed, cpu, util, cpi_est, faults};
}

// ──────────────────────────────────────────────────────────────────────────
// Formatting Helpers
// ──────────────────────────────────────────────────────────────────────────
static std::string right_pad(const std::string& s, int w) {
    return (int)s.size() >= w ? s : s + std::string(w - s.size(), ' ');
}

static void draw_line(const std::vector<int>& w) {
    std::cout << "+";
    for (int x : w) std::cout << std::string(x + 2, '-') << "+";
    std::cout << "\n";
}

static void draw_row(const std::vector<std::string>& c,
                     const std::vector<int>& w) {
    std::cout << "|";
    for (size_t i = 0; i < c.size(); ++i)
        std::cout << " " << right_pad(c[i], w[i]) << " |";
    std::cout << "\n";
}

static std::string fmt(double v, int p = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(p) << v;
    return ss.str();
}

// ──────────────────────────────────────────────────────────────────────────
// Report Printer (Simplified)
// ──────────────────────────────────────────────────────────────────────────
static void print_report(
    const std::string& title,
    const std::string& color,
    int ny, int nx,
    const PerfStats& base,
    const std::vector<int>& threads,
    const std::vector<PerfStats>& stats)
{
    std::cout << color
              << "  ══════════════════════════════════════════\n"
              << "  " << title << " (" << ny << " x " << nx << ")\n"
              << "  ══════════════════════════════════════════\033[0m\n\n";

    std::vector<int> w = {13, 20, 13, 20};
    draw_line(w);
    draw_row({"Threads", "Time (s)", "Speedup", "Efficiency"}, w);
    draw_line(w);

    draw_row({"1", fmt(base.wall_time), "1.00", "100%"}, w);
    draw_line(w);

    for (size_t i = 0; i < stats.size(); ++i) {
        double speedup = base.wall_time / stats[i].wall_time;
        double eff = (speedup / threads[i]) * 100.0;

        draw_row({
            std::to_string(threads[i]),
            fmt(stats[i].wall_time),
            fmt(speedup, 2),
            fmt(eff, 2) + "%"
        }, w);

        draw_line(w);
    }
    std::cout << "\n";
}

// ──────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ny> <nx>\n";
        return 1;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);

#ifdef _OPENMP
    int hw_threads = omp_get_max_threads();
#else
    int hw_threads = 1;
#endif

    auto data = create_input_data(ny, nx);
    std::vector<float> result((size_t)ny * ny, 0.0f);

    const std::vector<int> test_threads = {2, 4, 8, 10,12,14,16};

    correlate(ny, nx, data.data(), result.data()); // warm-up

    PerfStats baseline = run_benchmark(ny, nx, data, result, 1, hw_threads);

    std::vector<PerfStats> results;
    for (int t : test_threads)
        results.push_back(run_benchmark(ny, nx, data, result, t, hw_threads));

    std::string title, color;

#if VERSION == 1
    title = "SEQUENTIAL BASELINE";
    color = "\033[1;37m";
#elif VERSION == 2
    title = "PARALLEL (OpenMP)";
    color = "\033[1;33m";
#elif VERSION == 3
    title = "OPTIMIZED VERSION";
    color = "\033[1;32m";
#else
    title = "CORRELATION RUN";
    color = "\033[1;36m";
#endif

    print_report(title, color, ny, nx, baseline, test_threads, results);

    // Verification
    std::cout << "\nVerification (diagonal ≈ 1.0):\n";
    for (int i = 0; i < std::min(5, ny); ++i) {
        float v = result[i + i * ny];
        std::cout << "result[" << i << "][" << i << "] = "
                  << std::fixed << std::setprecision(6) << v
                  << ((std::abs(v - 1.0f) < 0.01f) ? " [OK]\n" : " [FAIL]\n");
    }

    return 0;
}
