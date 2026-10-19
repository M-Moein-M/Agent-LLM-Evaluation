/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef CT_FILEREADER_TYPES_H
#define CT_FILEREADER_TYPES_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint32_t imageDimension[2]; /* projection image dimension */
  uint32_t numberOfImages; /* number of projection images */
  float HUScalingFactors[2]; /* HU scaling factors */
} RabbitCtHeaderType;

typedef struct {
  RabbitCtHeaderType header;
  int imageSize;
  FILE *file;
} RabbitCtFileType;

#endif
