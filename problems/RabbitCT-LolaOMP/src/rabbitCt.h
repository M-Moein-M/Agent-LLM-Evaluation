/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef RABBITRUNNER_TYPES_H
#define RABBITRUNNER_TYPES_H
#include <stdint.h>

typedef struct {
  int id;
  double *matrix;
  float *image;
} Projection;

typedef struct {
  uint32_t problemSize;
  uint32_t imageWidth;
  uint32_t imageHeight;
  float voxelSize;
  float O_Index;
  float *volumeData;
  uint32_t numberOfProjections;
  double *globalGeometry;
  Projection *projectionBuffer;
  // not part of the original rabbitct
  char *clipFilename;
} RabbitCtGlobalData;

#define FNCSIGN extern "C"

#define RCT_FNCN_PREPAREALGORITHM "RCTPrepareAlgorithm"
#define RCT_FNCN_ALGORITHMBACKPROJ "RCTAlgorithmBackprojection"
#define RCT_FNCN_FINISHALGORITHM "RCTFinishAlgorithm"

#endif
