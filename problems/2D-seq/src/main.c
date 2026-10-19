/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.ke
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.    */

/* Harness-owned driver. DO NOT MODIFY.
 *
 * This file defines what is being measured: the timed region is exactly the
 * call to solve(), and the SOLVE_TIME line it prints is the number the
 * evaluation harness ranks on. It is frozen for the same reason verify.c is —
 * a solution that edits the timer is not a faster solution. Setup, allocation
 * and the solver itself all live in files you are free to change. */
#include <stdio.h>
#include <stdlib.h>

#include "likwid-marker.h"
#include "parameter.h"
#include "solver.h"
#include "timing.h"
#include "verify.h"

#define LIKWID_PROFILE(tag, call)                                              \
  startTime = getTimeStamp();                                                  \
  LIKWID_MARKER_START(#tag);                                                   \
  call(&solver);                                                               \
  LIKWID_MARKER_STOP(#tag);                                                    \
  endTime = getTimeStamp();

int main(int argc, char **argv) {
  double startTime, endTime;
  Parameter params;
  Solver solver;
  initParameter(&params);
  LIKWID_MARKER_INIT;

  if (argc < 2) {
    printf("Usage: %s <configFile>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  readParameter(&params, argv[1]);
  // printParameter(&params);

  initSolver(&solver, &params, 2);
  if (getenv("POISSON_WRITE_FIELD")) writeResult(&solver, "p-0.dat");

  LIKWID_PROFILE("SOR", solve);
  /* Tagged and at microsecond resolution on purpose: the harness matches this
     line by its SOLVE_TIME prefix, so no other output can be mistaken for it,
     and a solver that gets from ~100s down to well under a second is still
     resolved. */
  printf("SOLVE_TIME %.6fs\n", endTime - startTime);
  printDigest(&solver);
  if (getenv("POISSON_WRITE_FIELD")) writeResult(&solver, "p-final.dat");

  LIKWID_MARKER_CLOSE;
  return EXIT_SUCCESS;
}
