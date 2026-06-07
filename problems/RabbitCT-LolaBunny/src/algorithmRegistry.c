/* Copyright (C) the cluster, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <string.h>

#include "algorithmRegistry.h"

/* ---- forward declarations for each compiled-in algorithm ---- */
extern int lolaBunnyPrepare(RabbitCtGlobalData *);
extern int lolaBunnyBackprojection(RabbitCtGlobalData *);
extern int lolaBunnyFinish(RabbitCtGlobalData *);

/* ---- global function pointer variables ---- */
FncPrepareAlgorithmType FncPrepareAlgorithm;
FncAlgorithmIterationType FncAlgorithmIteration;
FncFinishAlgorithmType FncFinishAlgorithm;

/* ---- static registry: add new algorithms here ---- */
static const AlgorithmEntryType S_ALGORITHMS[] = {
  { "LolaBunny", lolaBunnyPrepare, lolaBunnyBackprojection, lolaBunnyFinish },
  { NULL,        NULL,             NULL,                    NULL            }  /* sentinel */
};

int algorithmRegistryFind(const char *name)
{
  for (int i = 0; S_ALGORITHMS[i].name != NULL; i++) {
    if (strcmp(S_ALGORITHMS[i].name, name) == 0) {
      FncPrepareAlgorithm   = S_ALGORITHMS[i].prepare;
      FncAlgorithmIteration = S_ALGORITHMS[i].iterate;
      FncFinishAlgorithm    = S_ALGORITHMS[i].finish;
      return 1;
    }
  }
  printf("Unknown algorithm: %s\n", name);
  algorithmRegistryList();
  return 0;
}

void algorithmRegistryList(void)
{
  printf("Available algorithms:\n");
  for (int i = 0; S_ALGORITHMS[i].name != NULL; i++) {
    printf("  %s\n", S_ALGORITHMS[i].name);
  }
}
