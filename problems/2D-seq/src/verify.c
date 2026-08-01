/* Harness-owned verification. DO NOT MODIFY.
 *
 * Reduces the interior of the pressure field to a handful of scalars and
 * prints them on a single DIGEST line. The evaluation harness compares that
 * line against a reference produced by the unmodified solver and rejects any
 * run whose values differ by more than a small relative tolerance.
 *
 * The reduction is deliberately kept serial and in a fixed order so that it
 * contributes no floating-point variation of its own. It runs once, after the
 * timed region, so it does not affect the reported runtime. */
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "verify.h"

#define P(i, j) p[(j) * (imax + 2) + (i)]

void printDigest(Solver *solver) {
  int imax = solver->imax;
  int jmax = solver->jmax;
  double *p = solver->p;

  double sumsq = 0.0, sumabs = 0.0, osc = 0.0;
  double mn = DBL_MAX, mx = -DBL_MAX;

  for (int j = 1; j < jmax + 1; j++) {
    for (int i = 1; i < imax + 1; i++) {
      double v = P(i, j);
      sumsq += v * v;
      sumabs += fabs(v);
      osc += ((i + j) & 1) ? -v : v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }

  double n = (double)imax * (double)jmax;
  printf("DIGEST l2=%.15e l1=%.15e osc=%.15e min=%.15e max=%.15e\n",
         sqrt(sumsq / n), sumabs / n, osc / n, mn, mx);
}
