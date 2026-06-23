// ===========================================================================
//  Your iteration harness for `./gemm -c`: verifies correctness against a CPU
//  reference over a fixed size sweep, then prints a self-measured throughput
//  line so you can see progress while you optimize.
//
//  NOTE: this printed number is for YOUR convenience only — it is NOT how you
//  are graded. The evaluation ignores it: it links your launchGemm() into its
//  own benchmark and scores your throughput as a fraction of cuBLAS on a fixed
//  workload (timed by Nsight Systems). You may edit this file freely; the only
//  fixed contract is the launchGemm() signature in gemm.cuh.
// ===========================================================================

#include "gemm.cuh"

#include <cuda_runtime.h>

#include <cmath>
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

static void fillRandom(std::vector<float>& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : m) x = dist(rng);
}

static void cpuReference(const std::vector<float>& A, const std::vector<float>& B,
                         std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k)
                acc += (double)A[i * N + k] * (double)B[k * N + j];
            C[i * N + j] = (float)acc;
        }
}

static void runGpu(const std::vector<float>& hA, const std::vector<float>& hB,
                   std::vector<float>& hC, int N) {
    const size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));  // unwritten entries read back as 0

    launchGemm(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

static double mse(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s / (double)a.size();
}

// Run a fixed sweep of sizes — deliberately mixing multiples and non-multiples
// of the kernel's tile size — and report the worst MSE against the CPU
// reference. A correct implementation produces a near-zero error for every N.
static void verify() {
    const int sizes[] = {128, 200, 256, 257, 384};
    const int n = sizeof(sizes) / sizeof(sizes[0]);
    double worst = 0.0;

    printf("Verifying tiled GEMM against CPU reference:\n");
    for (int s = 0; s < n; ++s) {
        const int N = sizes[s];
        std::vector<float> A((size_t)N * N), B((size_t)N * N);
        std::vector<float> C((size_t)N * N), R((size_t)N * N);
        fillRandom(A, 1234u + (unsigned)N);
        fillRandom(B, 5678u + (unsigned)N);

        runGpu(A, B, C, N);
        cpuReference(A, B, R, N);

        const double e = mse(C, R);
        if (e > worst) worst = e;
        printf("  N=%-4d  MSE=%.6e  %s\n", N, e, e < 1e-2 ? "ok" : "MISMATCH");
    }
    printf("Mean Squared Error: %.6e\n", worst);
}

// Sustained GPU work at a tile-aligned size so the harness GPU sampler observes
// real device utilization (and a CPU-only re-implementation stands out). The
// loop keeps launching until at least MIN_SECONDS of wall time elapse, so the
// 250 ms sampler gets enough samples on any GPU, fast or slow.
static void perf() {
    const int N = 2048;
    const double MIN_SECONDS = 3.0;
    std::vector<float> A((size_t)N * N), B((size_t)N * N), C((size_t)N * N);
    fillRandom(A, 11u);
    fillRandom(B, 22u);

    const size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    double elapsed = 0.0;
    long long iters = 0;
    while (elapsed < MIN_SECONDS) {
        cudaEventRecord(a);
        for (int i = 0; i < 50; ++i, ++iters) launchGemm(dA, dB, dC, N);
        cudaEventRecord(b);
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        elapsed += ms / 1000.0;
    }
    printf("Perf: N=%d iters=%lld  total time [s]: %.2f\n", N, iters, elapsed);
    // 2*N^3 FLOPs per GEMM, summed over all launches — the conventional GEMM
    // throughput figure, printed for your own feedback. NOT the graded number
    // (grading is external, vs cuBLAS; see the file header).
    const double gflops = 2.0 * (double)N * N * N * (double)iters / elapsed / 1e9;
    printf("Performance: %.2f GFLOP/s\n", gflops);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
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
