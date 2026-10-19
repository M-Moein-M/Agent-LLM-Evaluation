/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>

#include "analyseGeometry.h"
#include "memoryUtils_types.h"
#include "rabbitCt.h"
#ifdef KERNEL_CYCLES
#include "rabbitTimer.h"
#endif

static OutShadowType Shadow;
static LineRangeType **Range;
static ZeroPaddingType Padding;
static float *PaddedImg;

void lolaOmpFinish(RabbitCtGlobalData *rcgd)
{
  (void)rcgd;
}

int lolaOmpPrepare(RabbitCtGlobalData *rcgd)
{
  /* numberOfProjections is set to N in main if -a was given */
  if (rcgd->numberOfProjections == 0) {
    fprintf(stderr,
        "This module needs global geometry information.\n"
        "Please add -a switch!\n");
    exit(EXIT_FAILURE);
  }

  /* Allocate N * L * L * sizeof(LineRange) bytes of memory:
   * For each projection we need L * L LineRanges */
  if ((Range = (LineRangeType **)malloc(
           rcgd->numberOfProjections * sizeof(LineRangeType *))) == NULL) {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < rcgd->numberOfProjections; i++) {
    if ((Range[i] = (LineRangeType *)malloc(
             rcgd->problemSize * rcgd->problemSize * sizeof(LineRangeType))) == NULL) {
      perror("malloc");
      exit(EXIT_FAILURE);
    }
  }

  /* Calculate LineRanges
   *
   * Some rays through our volume might not hit the detector pane.
   * For each projection image, we calculate a x-voxelline-wise clipping
   * mask for our volume.
   */
  computeLineRanges(rcgd, Range);

  return 1;
}

int lolaOmpBackprojection(RabbitCtGlobalData *rcgd)
{
  const int l             = rcgd->problemSize;
  const int np            = rcgd->numberOfProjections; // projections in this run
  const float mm          = rcgd->voxelSize;
  const float o           = rcgd->O_Index;
  const float imageWidth  = rcgd->imageWidth;
  const float imageHeight = rcgd->imageHeight;

  float *vol              = rcgd->volumeData;

  for (int p = 0; p < np; ++p) {
    const double *const a = rcgd->projectionBuffer[p].matrix;
    const float *const i  = rcgd->projectionBuffer[p].image;
    const int id          = rcgd->projectionBuffer[p].id;

#pragma omp parallel for
    for (int z = 0; z < l; ++z) {
      for (int y = 0; y < l; ++y) {
        /* Select starting voxel and voxel count in x direction for this
         * projection image */
        int start = Range[id - 1][z * l + y].start;
        int stop  = Range[id - 1][z * l + y].stop;
        if (stop - start == 0)
          continue;

        for (int x = start; x < stop; ++x) {
          /* Convert from VCS to WCS */
          float wz = z * mm + o;
          float wy = y * mm + o;
          float wx = x * mm + o;

          float u, v, w;
          u = a[0] * wx + a[3] * wy + a[6] * wz + a[9];
          v = a[1] * wx + a[4] * wy + a[7] * wz + a[10];
          w = a[2] * wx + a[5] * wy + a[8] * wz + a[11];

          float ix, iy;
          int iix, iiy;
          ix  = u / w;
          iy  = v / w;
          iix = (int)ix;
          iiy = (int)iy;

          // continue with next voxel if we missed the projection
          if ((iix + 1 < 0) || (iix > imageWidth - 1) || (iiy + 1 < 0) ||
              (iiy > imageHeight - 1))
            continue;

          // bilinear interpolation
          float alpha  = ix - iix;
          float beta   = iy - iiy;
          const int iw = (int)imageWidth;

          float val    = 0.0f;
          if (iix >= 0 && iix < iw && iiy >= 0 && iiy < (int)imageHeight)
            val += (1.0f - alpha) * (1.0f - beta) * i[iiy * iw + iix];
          if (iix + 1 < iw && iix + 1 >= 0 && iiy >= 0 && iiy < (int)imageHeight)
            val += alpha * (1.0f - beta) * i[iiy * iw + iix + 1];
          if (iix >= 0 && iix < iw && iiy + 1 >= 0 && iiy + 1 < (int)imageHeight)
            val += (1.0f - alpha) * beta * i[(iiy + 1) * iw + iix];
          if (iix + 1 < iw && iix + 1 >= 0 && iiy + 1 >= 0 && iiy + 1 < (int)imageHeight)
            val += alpha * beta * i[(iiy + 1) * iw + iix + 1];

          vol[z * l * l + y * l + x] += val / (w * w);
        } // x-loop
      } // y-loop
    } // z-loop
  } // projection loop
  return 1;
}
