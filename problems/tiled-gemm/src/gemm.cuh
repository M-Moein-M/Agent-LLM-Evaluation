#pragma once

// Tiled single-precision GEMM: C = A * B for NxN row-major matrices that
// already live in device memory. C must be allocated by the caller and is
// overwritten with the product.
void launchGemm(const float* dA, const float* dB, float* dC, int N);
