/* Copyright (C) the cluster, University Erlangen-Nuremberg.ke
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.    */
#include <stdio.h>
#include <stdlib.h>

#include "likwid-marker.h"
#include "parameter.h"
#include "solver.h"
#include "timing.h"

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
  writeResult(&solver, "p-0.dat");

  LIKWID_PROFILE("SOR", solve);
  printf(" %.2fs\n", endTime - startTime);
  writeResult(&solver, "p-final.dat");

  LIKWID_MARKER_CLOSE;
  return EXIT_SUCCESS;
}
