/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef ALGORITHMREGISTRY_H
#define ALGORITHMREGISTRY_H

#include "moduleLoader_types.h"

typedef struct {
  const char *name;
  FncPrepareAlgorithmType prepare;
  FncAlgorithmIterationType iterate;
  FncFinishAlgorithmType finish;
} AlgorithmEntryType;

extern FncPrepareAlgorithmType FncPrepareAlgorithm;
extern FncAlgorithmIterationType FncAlgorithmIteration;
extern FncFinishAlgorithmType FncFinishAlgorithm;

extern int algorithmRegistryFind(const char *name);
extern void algorithmRegistryList(void);

#endif
