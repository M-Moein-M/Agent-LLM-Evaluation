/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef NUMA_TYPES_H
#define NUMA_TYPES_H

#include <stdint.h>

typedef struct {
  int id;
  uint32_t totalMemory;
  uint32_t freeMemory;
  int numberOfProcessors;
  uint32_t *processors;
} RabbitNumaNode;

typedef struct {
  uint32_t numberOfNodes;
  RabbitNumaNode *nodes;
} RabbitNumaTopology;

#endif /*NUMA_TYPES_H*/
