/* Fixed verifier. The agent must not modify this file. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "solve.h"

#define N 20000000L

int main(int argc, char** argv) {
    int check = argc > 1 && strcmp(argv[1], "-c") == 0;
    double* x = malloc(N * sizeof(double));
    unsigned long long s = 88172645463325252ULL;
    for (long i = 0; i < N; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        x[i] = (double)(s % 2000) / 1000.0 - 1.0;
    }
    double ref = 0.0;
    for (long i = 0; i < N; i++) ref += x[i] * x[i];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double got = sum_squares(x, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("Time: %.3f ms\n", ms);
    if (check) {
        double rel = (got - ref) / ref;
        printf("Mean Squared Error: %g\n", rel * rel);
    }
    free(x);
    return 0;
}
