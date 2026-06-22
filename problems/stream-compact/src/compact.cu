#include "compact.cuh"

#include <cuda_runtime.h>

// TODO: implement order-preserving stream compaction on the GPU.
//
//   Keep every element of d_in with value > threshold, preserving the input
//   order, write them contiguously into d_out[0..count), and return count.
//
// Constraints (see PROMPT.md):
//   - Implement this yourself with your own CUDA kernels.
//   - Do NOT call Thrust or CUB (e.g. thrust::copy_if, cub::DeviceSelect).
//   - Do not modify the verification harness in src/main.cu.
int compact(const int* d_in, int n, int threshold, int* d_out) {
    (void)d_in;
    (void)n;
    (void)threshold;
    (void)d_out;
    return 0;  // not implemented
}
