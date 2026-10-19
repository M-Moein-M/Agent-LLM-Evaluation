/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>

#include "rabbitCt.h"

static RabbitCtGlobalData *SRcgd = NULL;

static inline double pN(const float *image, int i, int j)
{
  if (i >= 0 && i < (int)SRcgd->imageWidth && j >= 0 && j < (int)SRcgd->imageHeight)
    return image[j * SRcgd->imageWidth + i];

  return 0.0;
}

static inline double pHatN(const float *image, double x, double y)
{
  int i        = (int)x;
  int j        = (int)y;
  double alpha = x - (int)x;
  double beta  = y - (int)y;

  return (1.0 - alpha) * (1.0 - beta) * pN(image, i, j) +
         alpha * (1.0 - beta) * pN(image, i + 1, j) +
         (1.0 - alpha) * beta * pN(image, i, j + 1) +
         alpha * beta * pN(image, i + 1, j + 1);
}

int lolaBunnyPrepare(RabbitCtGlobalData *rcgd)
{
  (void)rcgd;
  return 1;
}

int lolaBunnyFinish(RabbitCtGlobalData *rcgd)
{
  (void)rcgd;
  return 1;
}

int lolaBunnyBackprojection(RabbitCtGlobalData *r)
{
  unsigned int l = r->problemSize;
  float oL       = r->O_Index;
  float rL       = r->voxelSize;
  float *fL      = r->volumeData;

  SRcgd          = r;

  for (int p = 0; p < (int)r->numberOfProjections; p++) {
    double *aN         = r->projectionBuffer[p].matrix;
    const float *image = r->projectionBuffer[p].image;

    for (unsigned int k = 0; k < l; k++) {
      double z = oL + (double)k * rL;
      for (unsigned int j = 0; j < l; j++) {
        double y = oL + (double)j * rL;
        for (unsigned int i = 0; i < l; i++) {
          double x  = oL + (double)i * rL;

          double wN = aN[2] * x + aN[5] * y + aN[8] * z + aN[11];
          double uN = (aN[0] * x + aN[3] * y + aN[6] * z + aN[9]) / wN;
          double vN = (aN[1] * x + aN[4] * y + aN[7] * z + aN[10]) / wN;

          fL[k * l * l + j * l + i] += (float)(1.0 / (wN * wN) * pHatN(image, uN, vN));
        }
      }
    }
  }

  return 1;
}
