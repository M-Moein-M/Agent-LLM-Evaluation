/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#ifndef AFFINITY_TYPES_H
#define AFFINITY_TYPES_H

#include <stdint.h>

typedef struct {
  char *tag;
  uint32_t numberOfProcessors;
  int *processorList;
} AffinityDomainType;

#endif /*AFFINITY_TYPES_H*/
