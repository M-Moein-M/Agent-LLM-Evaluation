/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef ANALYSEGEOMETRY_TYPES_H
#define ANALYSEGEOMETRY_TYPES_H

#include <stdint.h>

typedef struct {
  int x;
  int y;
  int z;
} Point3DType;

typedef struct {
  int Umin;
  int Umax;
  int Vmin;
  int Vmax;
} OutShadowType;

typedef struct {
  int16_t start;
  int16_t stop;
} LineRangeType;

#endif /*ANALYSEGEOMETRY_TYPES_H*/
