/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef MEMORYUTILS_TYPES_H
#define MEMORYUTILS_TYPES_H

#include <stdint.h>

#include "analyseGeometry_types.h"

typedef struct {
  float ***buffern;
  float **buffer;
  OutShadowType extend;
  uint64_t paddedSize;
  int startOffset;
  int lineOffset;
  int lineSize;
  float **savePtr;
  int *masterProcessor;
} ZeroPaddingType;

#endif /*MEMORYUTILS_TYPES_H*/
