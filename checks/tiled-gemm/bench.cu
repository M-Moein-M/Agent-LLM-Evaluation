// ===========================================================================
//  HARNESS-OWNED GRADING BENCH for tiled-gemm — NOT part of the agent's tree.
//
//  This file lives under checks/ and is injected read-only at grade time
//  (bound to /grade/bench.cu by host.sh, compiled against the agent's kernel
//  sources by measure.sh). The agent never sees or edits it, so the graded
//  measurement cannot be faked, omitted, or miscomputed by the agent — the
//  agent supplies only the kernel behind the fixed entry point:
//
//      void launchGemm(const float* dA, const float* dB, float* dC, int N);
//
//  What it does, for a single mode given on argv ("agent" | "cublas"):
//    * builds one fixed, harness-controlled NxN workload (same seeds for both
//      modes, so the two are bit-for-bit comparable),
//    * (agent mode only) verifies the agent kernel's output against cuBLAS and
//      prints "BENCH mse=<x>" — correctness at the GRADED size, independent of
//      the agent's own Makefile flags (e.g. -use_fast_math),
//    * warms up, then times ITERS launches inside a cudaProfiler start/stop
//      window so nsys (run with --capture-range=cudaProfilerApi) reports ONLY
//      the timed kernels — not warmup, not allocation, not the H2D copies,
//    * prints a cudaEvent cross-check line "BENCH <mode> N=.. iters=.. ms_total=..".
//
//  The grade is a RATIO of cuBLAS kernel time to agent kernel time on this same
//  workload (see checks/tiled-gemm/check.py): it never assumes an operation
//  count, and it self-calibrates to whatever GPU runs it. nsys supplies the
//  authoritative timing; the ms_total here is a fallback / sanity cross-check.
// ===========================================================================

#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Fixed, harness-controlled workload. Large enough to give the A40 real work
// (and enough nsys samples), small enough to fit any modern GPU: 3 x N^2 floats
// = ~768 MB at N=8192? no — at N=4096 it is 3*4096^2*4 = 201 MB.
static const int N = 4096;
static const int ITERS = 50;
static const int WARMUP = 5;

#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(_e), __FILE__, __LINE__);              \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s,            \
                    __FILE__, __LINE__);                                      \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

// The agent's deliverable. Declared here (not via the agent's gemm.cuh) so the
// graded interface is fixed by US: if the agent changed this signature, the
// link fails and the run forfeits perf credit — correctness still scores.
extern void launchGemm(const float* dA, const float* dB, float* dC, int N);

static void fillRandom(std::vector<float>& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : m) x = dist(rng);
}

// Row-major C = A*B via cuBLAS (which is column-major): reading our row-major
// buffers column-major transposes them, so computing Bcm*Acm = (A*B)^T column-
// major == A*B row-major in the same C buffer. Hence the swapped B,A order.
static void cublasGemm(cublasHandle_t h, const float* dA, const float* dB,
                       float* dC, int n) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                             dB, n, dA, n, &beta, dC, n));
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "agent";
    const bool is_agent = std::strcmp(mode, "agent") == 0;
    const bool is_cublas = std::strcmp(mode, "cublas") == 0;
    if (!is_agent && !is_cublas) {
        fprintf(stderr, "usage: %s agent|cublas\n", argv[0]);
        return 2;
    }

    const size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA((size_t)N * N), hB((size_t)N * N);
    fillRandom(hA, 11u);
    fillRandom(hB, 22u);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- correctness at the graded size (agent mode only) -----------------
    // Compare the agent kernel against cuBLAS on identical inputs so a kernel
    // that is "fast" only because it is wrong (or loses accuracy under the
    // agent's own build flags) is caught here, not rewarded.
    if (is_agent) {
        float* dRef;
        CUDA_CHECK(cudaMalloc(&dRef, bytes));
        cublasGemm(handle, dA, dB, dRef, N);
        launchGemm(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> hRef((size_t)N * N), hOut((size_t)N * N);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double s = 0.0;
        for (size_t i = 0; i < hRef.size(); ++i) {
            const double d = (double)hOut[i] - (double)hRef[i];
            s += d * d;
        }
        printf("BENCH mse=%.6e\n", s / (double)hRef.size());
        cudaFree(dRef);
    }

    // ---- warmup (outside the profiler window) ------------------------------
    for (int i = 0; i < WARMUP; ++i) {
        if (is_agent) launchGemm(dA, dB, dC, N);
        else          cublasGemm(handle, dA, dB, dC, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- timed window: only these kernels land in the nsys report ----------
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    cudaProfilerStart();
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < ITERS; ++i) {
        if (is_agent) launchGemm(dA, dB, dC, N);
        else          cublasGemm(handle, dA, dB, dC, N);
    }
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    cudaProfilerStop();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    printf("BENCH %s N=%d iters=%d ms_total=%.3f ms_per=%.4f\n",
           mode, N, ITERS, ms, ms / ITERS);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}