/*
 * correlate.cpp
 * Correlation Coefficient Calculator
 */

#include "correlate.h"
#include <cmath>
#include <vector>

#ifndef VERSION
#define VERSION 1
#endif

#if VERSION >= 2
#include <omp.h>
#endif

#if VERSION == 3
    #if defined(__AVX2__) && defined(__FMA__)
        #include <immintrin.h>
        #define HAS_AVX2 1
    #else
        #define HAS_AVX2 0
    #endif
#endif

//==============================================================================
// VERSION 1: Sequential
//==============================================================================
#if VERSION == 1

void correlate(int ny, int nx, const float* data, float* result) {

    std::vector<double> z_buffer(ny * nx);

    for (int r = 0; r < ny; ++r) {
        int base = r * nx;

        double sum_val = 0.0;
        for (int i = 0; i < nx; ++i)
            sum_val += data[base + i];

        double mean = sum_val / nx;

        double var_sum = 0.0;
        for (int i = 0; i < nx; ++i) {
            double diff = data[base + i] - mean;
            z_buffer[base + i] = diff;
            var_sum += diff * diff;
        }

        double scale = (var_sum > 1e-10)
                     ? std::sqrt(nx / var_sum)
                     : 0.0;

        for (int i = 0; i < nx; ++i)
            z_buffer[base + i] *= scale;
    }

    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c <= r; ++c) {
            double dot_sum = 0.0;

            for (int k = 0; k < nx; ++k)
                dot_sum += z_buffer[r * nx + k] *
                           z_buffer[c * nx + k];

            double corr = dot_sum / nx;
            corr = std::max(-1.0, std::min(1.0, corr));

            result[r + c * ny] = static_cast<float>(corr);
        }
    }
}

//==============================================================================
// VERSION 2: OpenMP Parallel
//==============================================================================
#elif VERSION == 2

void correlate(int ny, int nx, const float* data, float* result) {

    std::vector<double> norm_rows(ny * nx);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ny; ++r) {
        int base = r * nx;

        double mean = 0.0;
        for (int i = 0; i < nx; ++i)
            mean += data[base + i];
        mean /= nx;

        double var_sum = 0.0;
        for (int i = 0; i < nx; ++i) {
            double centered = data[base + i] - mean;
            norm_rows[base + i] = centered;
            var_sum += centered * centered;
        }

        double inv_std = (var_sum > 1e-10)
                       ? (std::sqrt(nx) / std::sqrt(var_sum))
                       : 0.0;

        for (int i = 0; i < nx; ++i)
            norm_rows[base + i] *= inv_std;
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (int r = 0; r < ny; ++r) {
        const double* row_a = &norm_rows[r * nx];

        for (int c = 0; c <= r; ++c) {
            const double* row_b = &norm_rows[c * nx];

            double dot_sum = 0.0;
            for (int k = 0; k < nx; ++k)
                dot_sum += row_a[k] * row_b[k];

            double corr = dot_sum / nx;
            corr = std::max(-1.0, std::min(1.0, corr));

            result[r + c * ny] = static_cast<float>(corr);
        }
    }
}

//==============================================================================
// VERSION 3: OpenMP + SIMD
//==============================================================================
#elif VERSION == 3

void correlate(int ny, int nx, const float* data, float* result) {

    const double inv_n = 1.0 / nx;
    std::vector<double> work_buffer(ny * nx);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ny; ++r) {
        const float* in = &data[r * nx];
        double* out = &work_buffer[r * nx];

        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < nx; ++i)
            sum += in[i];

        double mean = sum * inv_n;

        double var_sum = 0.0;
        #pragma omp simd reduction(+:var_sum)
        for (int i = 0; i < nx; ++i) {
            double diff = in[i] - mean;
            out[i] = diff;
            var_sum += diff * diff;
        }

        double scale = (var_sum > 1e-10)
                     ? std::sqrt(nx / var_sum)
                     : 0.0;

        #pragma omp simd
        for (int i = 0; i < nx; ++i)
            out[i] *= scale;
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (int r = 0; r < ny; ++r) {
        const double* __restrict__ a = &work_buffer[r * nx];

        for (int c = 0; c <= r; ++c) {
            const double* __restrict__ b = &work_buffer[c * nx];

            double dot_sum = 0.0;

            #if HAS_AVX2
            __m256d acc = _mm256_setzero_pd();
            int k = 0;

            for (; k + 3 < nx; k += 4) {
                __m256d va = _mm256_loadu_pd(a + k);
                __m256d vb = _mm256_loadu_pd(b + k);
                acc = _mm256_fmadd_pd(va, vb, acc);
            }

            __m128d hi = _mm256_extractf128_pd(acc, 1);
            __m128d lo = _mm256_castpd256_pd128(acc);
            __m128d sum2 = _mm_add_pd(hi, lo);
            __m128d sum1 = _mm_hadd_pd(sum2, sum2);
            dot_sum = _mm_cvtsd_f64(sum1);

            for (; k < nx; ++k)
                dot_sum += a[k] * b[k];
            #else
            #pragma omp simd reduction(+:dot_sum)
            for (int k = 0; k < nx; ++k)
                dot_sum += a[k] * b[k];
            #endif

            double corr = dot_sum * inv_n;
            corr = std::max(-1.0, std::min(1.0, corr));

            result[r + c * ny] = static_cast<float>(corr);
        }
    }
}

#else
#error "VERSION must be 1, 2, or 3"
#endif
