#include "solve.h"

/* Correct but slow starting point. */
double sum_squares(const double* x, long n) {
    double s = 0.0;
    for (long i = 0; i < n; i++) s += x[i] * x[i];
    return s;
}
