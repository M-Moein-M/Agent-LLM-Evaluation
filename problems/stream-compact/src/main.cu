// ===========================================================================
//  VERIFICATION HARNESS — DO NOT MODIFY.
//
//  The evaluation replays `./compact -c`. The input, the CPU reference, the
//  number of repetitions, and the error metric below are fixed and are how your
//  solution is graded. Implement `compact()` in src/compact.cu, not this file.
//
//  Why it repeats: an order-preserving compaction that uses a racing atomic
//  append produces the right *set* of elements in a nondeterministic *order*.
//  Running the same input many times and reporting the WORST-case error turns
//  that runtime nondeterminism into a single, reproducible number.
// ===========================================================================

#include "compact.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(_e), __FILE__, __LINE__);              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// A random permutation of [0, n): all values are distinct (so any reordering of
// the kept elements is detectable) and the kept set is scattered across the
// input (so output positions can't be guessed with a closed form).
static std::vector<int> makePermutation(int n, unsigned seed) {
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    std::mt19937 rng(seed);
    for (int i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(p[i], p[dist(rng)]);
    }
    return p;
}

static std::vector<int> cpuCompact(const std::vector<int>& in, int threshold) {
    std::vector<int> out;
    out.reserve(in.size());
    for (int v : in)
        if (v > threshold) out.push_back(v);
    return out;
}

static void verify() {
    const int n = 1 << 20;
    const int threshold = n / 2;
    const int REPS = 32;

    const std::vector<int> host = makePermutation(n, 12345u);
    const std::vector<int> expected = cpuCompact(host, threshold);
    const int expCount = (int)expected.size();

    int *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dIn, host.data(), (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));

    double worst = 0.0;
    int worstRun = -1, worstCount = -1;
    std::vector<int> out(n);

    for (int r = 0; r < REPS; ++r) {
        CUDA_CHECK(cudaMemset(dOut, 0, (size_t)n * sizeof(int)));

        const int count = compact(dIn, n, threshold, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        const int copyN = (count < 0) ? 0 : (count > n ? n : count);
        if (copyN > 0)
            CUDA_CHECK(cudaMemcpy(out.data(), dOut, (size_t)copyN * sizeof(int),
                                  cudaMemcpyDeviceToHost));

        // Compare element-by-element against the order-preserving reference.
        // Positions past one side's length count as a (large) mismatch, so a
        // wrong count is penalised too.
        const long m = std::max((long)expCount, (long)copyN);
        double sum = 0.0;
        for (long k = 0; k < m; ++k) {
            const double a = (k < copyN) ? (double)out[k] : -1.0;
            const double b = (k < expCount) ? (double)expected[k] : -1.0;
            const double d = a - b;
            sum += d * d;
        }
        const double e = (m > 0) ? sum / (double)m : 0.0;
        if (e > worst) {
            worst = e;
            worstRun = r;
            worstCount = count;
        }
    }

    printf("Verifying order-preserving compaction (n=%d, kept=%d, reps=%d):\n",
           n, expCount, REPS);
    printf("  worst run=%d  returned_count=%d (expected %d)\n",
           worstRun, worstCount, expCount);
    printf("Mean Squared Error: %.6e\n", worst);

    cudaFree(dIn);
    cudaFree(dOut);
}

// Sustained GPU work so the harness sampler observes utilization (and a
// CPU-only implementation stands out as ~0% GPU). Runs until MIN_SECONDS of
// wall time elapse, independent of GPU speed.
static void perf() {
    const int n = 1 << 24;
    const int threshold = n / 2;
    const double MIN_SECONDS = 3.0;

    const std::vector<int> host = makePermutation(n, 777u);
    int *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dIn, host.data(), (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));

    const auto t0 = std::chrono::steady_clock::now();
    long long iters = 0;
    double elapsed = 0.0;
    while (elapsed < MIN_SECONDS) {
        for (int i = 0; i < 10; ++i, ++iters) compact(dIn, n, threshold, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        elapsed = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
    }
    printf("Perf: n=%d iters=%lld  total time [s]: %.2f\n", n, iters, elapsed);

    cudaFree(dIn);
    cudaFree(dOut);
}

int main(int argc, char** argv) {
    bool check = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "-c") check = true;

    // Compute the authoritative correctness result first so it always lands in
    // run.log, then generate sustained GPU load for the harness sampler.
    if (check) {
        verify();
        perf();
    } else {
        perf();
        printf("Run with -c to verify correctness.\n");
    }
    return 0;
}
