/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "analyseGeometry.h"
#include "analyseGeometry_types.h"
#include "rabbitCt.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void initCuboid(Point3DType *s, Point3DType corners[], Point3DType *p)
{
  corners[0].x = p->x;
  corners[0].y = p->y;
  corners[0].z = p->z;

  corners[1].x = p->x;
  corners[1].y = p->y + s->y - 1;
  corners[1].z = p->z;

  corners[2].x = p->x + s->x - 1;
  corners[2].y = p->y;
  corners[2].z = p->z;

  corners[3].x = p->x + s->x - 1;
  corners[3].y = p->y + s->y - 1;
  corners[3].z = p->z;

  corners[4].x = p->x;
  corners[4].y = p->y;
  corners[4].z = p->z + s->z - 1;

  corners[5].x = p->x;
  corners[5].y = p->y + s->y - 1;
  corners[5].z = p->z + s->z - 1;

  corners[6].x = p->x + s->x - 1;
  corners[6].y = p->y;
  corners[6].z = p->z + s->z - 1;

  corners[7].x = p->x + s->x - 1;
  corners[7].y = p->y + s->y - 1;
  corners[7].z = p->z + s->z - 1;
}

void computeShadowOfProjection(RabbitCtGlobalData *data, OutShadowType *shadow)
{
  Point3DType size = { data->problemSize, data->problemSize, data->problemSize };
  Point3DType corners[8];
  Point3DType principalCorner = { 0, 0, 0 };
  float rL                    = data->voxelSize;
  float oL                    = data->O_Index;
  double minU                 = data->imageWidth + 1;
  double minV                 = data->imageHeight + 1;
  double maxU                 = -2;
  double maxV                 = -2;

  initCuboid(&size, corners, &principalCorner);

  for (uint32_t view = 0; view < data->numberOfProjections; view++) {
    double *aN = data->globalGeometry + (view * 12);

    for (int co = 0; co < 8; co++) {
      Point3DType p = corners[co];

      double x      = oL + (double)p.x * rL;
      double y      = oL + (double)p.y * rL;
      double z      = oL + (double)p.z * rL;

      double wN     = aN[2] * x + aN[5] * y + aN[8] * z + aN[11];
      double uN     = (aN[0] * x + aN[3] * y + aN[6] * z + aN[9]) / wN;
      double vN     = (aN[1] * x + aN[4] * y + aN[7] * z + aN[10]) / wN;

      if (uN < minU)
        minU = uN;
      if (vN < minV)
        minV = vN;
      if (uN > maxU)
        maxU = uN;
      if (vN > maxV)
        maxV = vN;
    }
  }

  shadow->Umin = (int)floor(minU);
  shadow->Vmin = (int)floor(minV);
  shadow->Umax = (int)ceil(maxU);
  shadow->Vmax = (int)ceil(maxV);
}

#define COMPUTE_GEOMETRY                                                                 \
  w_n = A_n[2] * wx + A_n[5] * wy + A_n[8] * wz + A_n[11];                               \
  u_n = (A_n[0] * wx + A_n[3] * wy + A_n[6] * wz + A_n[9]) / w_n;                        \
  v_n = (A_n[1] * wx + A_n[4] * wy + A_n[7] * wz + A_n[10]) / w_n;                       \
  iix = (int)u_n;                                                                        \
  iiy = (int)v_n

void computeLineRanges(RabbitCtGlobalData *data, LineRangeType **range)
{
  unsigned int l   = data->problemSize;
  unsigned int isx = data->imageWidth;
  unsigned int isy = data->imageHeight;
  unsigned int n   = data->numberOfProjections;
  const float mm   = data->voxelSize;

  /* Check whether a filename was specified using the -C option,
   * if not fall back to the default filename "LR_PREFIXxxx.rct",
   * where LR_PREFIX is "RabbitInput/LineRange" and xxx is the
   * problem size. */
  char *clipFile = data->clipFilename;
  if (clipFile == NULL) {
    int nchars;
    switch (l) {
    case 128:
    case 256:
    case 512:
      nchars = strlen(LR_PREFIX) + 3 + strlen(".rct") + 1;
      break;
    case 1024:
      nchars = strlen(LR_PREFIX) + 4 + strlen(".rct") + 1;
      break;
    default:
      printf("Unsupported problem size: %d\n", l);
      exit(EXIT_FAILURE);
    }

    switch (VECTORSIZE) {
    case 4:
      // fall through
    case 8:
      nchars += strlen("-x");
      break;
    case 16:
      nchars += strlen("-xx");
      break;
    default:
      printf("Unsupported VECTORSIZE size: %d\n", VECTORSIZE);
      exit(EXIT_FAILURE);
    }

    if ((clipFile = (char *)malloc(nchars * sizeof(char))) == NULL) {
      perror("malloc");
      exit(EXIT_FAILURE);
    }
    snprintf(clipFile, nchars, "%s%d-%d.rct", LR_PREFIX, l, VECTORSIZE);
    printf("No clipping file specified, falling back to %s\n", clipFile);
  }

  /* If filename exists, verify the size and load contents into range */
  FILE *fClipFile;
  if ((fClipFile = fopen(clipFile, "re")) != NULL) {
    /* verify file size */
    fseek(fClipFile, 0L, SEEK_END);
    int fsize = ftell(fClipFile);
    if (fsize != n * l * l * sizeof(LineRangeType)) {
      printf("corrupted file %s: size is %lu (should be %lu)\n",
          clipFile,
          (unsigned long)fsize,
          n * l * l * sizeof(LineRangeType));
      exit(EXIT_FAILURE);
    }
    fseek(fClipFile, 0L, SEEK_SET);

    /* Read file NUMA-aware into buffer. */
    int nbytes;
    for (int i = 0; i < n; ++i) {
#pragma omp parallel for schedule(static) ordered
      for (int j = 0; j < l; j++) {
#pragma omp ordered
        {
          nbytes = fread(range[i] + j * l, sizeof(LineRangeType), l, fClipFile);
          if (nbytes != l) {
            fprintf(stderr, "error reading from %s: ", clipFile);
            perror("");
            exit(EXIT_FAILURE);
          }
        }
      }
    }

    printf("Read %d bytes of clipping data from %s\n", fsize, clipFile);
    fclose(fClipFile);
    /* done */
    return;
  } else {
    /* Can't open clipFile for reading. This means that the file either
     * doesn't exist (in which case we create it, calculate the LineRange
     * and store it in the file) or there was some other problem opening
     * the file (in which case we exit). */
    if (errno != ENOENT) {
      fprintf(stderr, "fopen %s: ", clipFile);
      perror("");
      exit(EXIT_FAILURE);
    }

    /* Open file for writing. */
    if ((fClipFile = fopen(clipFile, "we")) == NULL) {
      fprintf(stderr, "fopen %s: ", clipFile);
      perror("");
      exit(EXIT_FAILURE);
    }

    /* Calculate LineRange */
#pragma omp parallel for
    for (int view = 0; view < data->numberOfProjections; view++) {
      double w_n;
      double u_n;
      double v_n;
      int iix, iiy;
      float wz = data->O_Index;
      for (int z = 0; z < l; z++, wz += mm) {
        double *A_n = data->globalGeometry + (view * 12);
        float wy    = data->O_Index;
        float wx;

        for (int y = 0; y < l; y++, wy += mm) {
          int xstart = -1;
          int xstop  = -1;

          // idea: we scan from left to right and set xstart to the first
          // voxel thats ray hits the projection image and break the loop
          // next: we scan from right to left and set xstop to the first
          // voxel thats raw hits the projection image and break the loop

          // scan from left to right
          for (int x = 0; x < l; x++) {
            wx = x * mm + data->O_Index;
            COMPUTE_GEOMETRY;

            // continue with next voxel if we missed the projection
            if ((iix + 1 < 0) || (iix > isx - 1) || (iiy + 1 < 0) || (iiy > isy - 1))
              continue;

            // we hit the projection image, this means that we have
            // to include voxels starting from here
            xstart = x;
            break;
          } // x-loop scan from left to right

          // if we scanned from left to right and didn't set xstart in
          // the process this means that we scanned the whole x-line
          // without ever hitting the projection image
          if (xstart == -1) {
            range[view][z * l + y].start = 0;
            range[view][z * l + y].stop  = 0;
            continue; // jump to next iteration in y-loop
          }

          // since we hit a voxel while scanning from left to right
          // we now scan from right to left to find out where to stop
          // (if at all)
          for (int x = l - 1; x >= 0; --x) {
            wx = x * mm + data->O_Index;
            COMPUTE_GEOMETRY;

            // continue with next voxel if we missed the projection
            if ((iix + 1 < 0) || (iix > isx - 1) || (iiy + 1 < 0) || (iiy > isy - 1))
              continue;

            // we hit the projection image, this means that we have
            // to include voxels up to here
            xstop = x + 1;
            break;
          } // x-loop scan from right to left

          // fix alignment
          int align = 0;
          if ((align = (xstop % VECTORSIZE)))
            xstop += (VECTORSIZE - align);
          if ((align = (xstart % VECTORSIZE)))
            xstart -= align;

          range[view][z * l + y].start = xstart;
          range[view][z * l + y].stop  = xstop;
        } // y-loop
      } // z-loop
    } // view-loop

    /* Write LineRange into file. */
    int nbytes;
    for (int i = 0; i < n; ++i) {
      nbytes = fwrite(range[i], sizeof(LineRangeType), l * l, fClipFile);
      if (nbytes != l * l) {
        fprintf(stderr, "error writing to %s: ", clipFile);
        perror("");
        exit(EXIT_FAILURE);
      }
    }
    printf("Wrote %lu bytes of clipping data to %s\n",
        n * l * l * sizeof(LineRangeType),
        clipFile);
    fclose(fClipFile);
  }
}
