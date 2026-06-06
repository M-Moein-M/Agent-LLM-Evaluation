#include "gemm.cuh"

#include <cuda_runtime.h>

// Edge length of a square thread-block tile. Each block cooperatively loads a
// TILE x TILE sub-block of A and B into shared memory, multiplies them, and
// accumulates into the corresponding tile of C.
#define TILE 16

// Classic shared-memory tiled matrix multiply. The kernel is written to handle
// matrices whose dimensions are not a multiple of TILE: every global load is
// guarded so out-of-range elements read as zero, and the final store is masked
// by `row < N && col < N`.
__global__ void gemmKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int numTiles = (N + TILE - 1) / TILE;

    float acc = 0.0f;
    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + threadIdx.x;
        const int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

void launchGemm(const float* dA, const float* dB, float* dC, int N) {
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);  // one block per output tile
    gemmKernel<<<grid, block>>>(dA, dB, dC, N);
}
