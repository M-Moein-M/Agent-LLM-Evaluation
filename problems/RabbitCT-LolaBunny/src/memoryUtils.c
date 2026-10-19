/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#include "memoryUtils.h"
#include "rabbitAffinity.h"
#include "rabbitNuma.h"

void memoryUtilsInit()
{
  rabbitNuma_init();
}

static void memoryUtilsNoInitAllocate(float **ptr, uint64_t size)
{
  int errorCode;

  errorCode = posix_memalign((void **)ptr, 64, size * sizeof(float));

  if (errorCode) {
    if (errorCode == EINVAL) {
      fprintf(stderr, "Alignment parameter is not a power of two\n");
      exit(EXIT_FAILURE);
    }
    if (errorCode == ENOMEM) {
      fprintf(stderr, "Insufficient memory to fulfill the request\n");
      exit(EXIT_FAILURE);
    }
  }
  memset(*ptr, 0, size * sizeof(float));
}

void memoryUtilsAllocate(float **ptr, uint64_t size)
{
  int errorCode;

  errorCode = posix_memalign((void **)ptr, 64, size * sizeof(float));

  if (errorCode) {
    if (errorCode == EINVAL) {
      fprintf(stderr, "Alignment parameter is not a power of two\n");
      exit(EXIT_FAILURE);
    }
    if (errorCode == ENOMEM) {
      fprintf(stderr, "Insufficient memory to fulfill the request\n");
      exit(EXIT_FAILURE);
    }
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1 * 512 * 512)
  for (int i = 0; i < size; ++i) {
    (*ptr)[i] = 0.0f;
  }
#else
  memset(*ptr, 0, size * sizeof(float));
#endif
}

void memoryUtilsZeroPadInit(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding)
{
  int align = 0;

  int padXl = padding->extend.Umin;
  int padXr = padding->extend.Umax;
  int padYb = padding->extend.Vmin;
  int padYt = padding->extend.Vmax;

  if ((align = (padYb % VECTORSIZE)))
    padYb += (VECTORSIZE - align);
  if ((align = (padXl % VECTORSIZE)))
    padXl += (VECTORSIZE - align);
  if ((align = (padXr % VECTORSIZE)))
    padXr += (VECTORSIZE - align);

  int xSize            = padXl + rcgd->imageWidth + padXr;
  int ySize            = padYb + rcgd->imageHeight + padYt;

  padding->paddedSize  = xSize * ySize;

  padding->startOffset = padYb * xSize + padXl;
  padding->lineOffset  = xSize;

  printf("Padded Size = %d X %d \n", xSize, ySize);
  printf("startOffset = %d \n", padding->startOffset);
  printf("lineOffset = %d \n", padding->lineOffset);
}

#if 0
void
memoryUtils_zeroPadAllocate(RabbitCtGlobalData * rcgd, ZeroPadding* padding )
{
    padding->savePtr = (float**) malloc(rcgd->numberOfProjections * sizeof(float*));
    padding->buffer = (float**) malloc(rcgd->numberOfProjections * sizeof(float*));

    for (int i=0; i<rcgd->numberOfProjections; i++)
    {
        memoryUtils_allocate(padding->buffer+i,padding->paddedSize*2);
    }
}

#endif

void memoryUtilsZeroPadAllocate(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding)
{
#ifdef __linux__
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  sched_getaffinity(0, sizeof(cpu_set_t), &cpuSet);
#endif

  padding->buffern         = (float ***)malloc(numa_info.numberOfNodes * sizeof(float **));
  padding->masterProcessor = (int *)malloc(numa_info.numberOfNodes * sizeof(int));

  for (int i = 0; i < numa_info.numberOfNodes; i++) {
    int masterCpu = -1;

#ifdef __linux__
    for (int j = 0; j < numa_info.nodes[i].numberOfProcessors; j++) {
      int cpu = numa_info.nodes[i].processors[j];
      if (CPU_ISSET(cpu, &cpuSet)) {
        masterCpu = cpu;
        break;
      }
    }
#else
    if (numa_info.nodes[i].numberOfProcessors > 0) {
      masterCpu = numa_info.nodes[i].processors[0];
    }
#endif

    if (masterCpu < 0) {
      padding->buffern[i]         = NULL;
      padding->masterProcessor[i] = -1;
      printf("Skipping NUMA node %d (no processors in affinity mask)\n", i);
      continue;
    }

    padding->masterProcessor[i] = masterCpu;
    rabbitAffinity_pinProcess(masterCpu);
    printf("Allocating Node %d on processor %d \n", i, masterCpu);

    padding->buffern[i] = (float **)malloc(rcgd->numberOfProjections * sizeof(float *));

    for (int j = 0; j < rcgd->numberOfProjections; j++) {
      memoryUtilsNoInitAllocate(padding->buffern[i] + j, padding->paddedSize * 2);
    }
  }

  /* restore affinity mask of process */
#ifdef __linux__
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet);
#endif

#if 0
    padding->savePtr = (float**) malloc(rcgd->numberOfProjections * sizeof(float*));
    padding->buffer = (float**) malloc(rcgd->numberOfProjections * sizeof(float*));

    for (int i=0; i<rcgd->numberOfProjections; i++)
    {
        memoryUtils_noInitAllocate(padding->buffer+i,padding->paddedSize * 2);
    }
#endif
}

void memoryUtilsZeroPadEnterExp(
    RabbitCtGlobalData *rcgd, ZeroPaddingType *padding, Projection *projectionBuffer)
{
  int myNode        = -1;
  int myWorkerNode  = -1;

  int myProcessorId = rabbitAffinity_threadGetProcessorId();

  for (int i = 0; i < numa_info.numberOfNodes; i++) {
    if (padding->masterProcessor[i] < 0) {
      continue;
    }
    if (myProcessorId == padding->masterProcessor[i]) {
      myNode = i;
      break;
    } else {
      for (int j = 0; j < numa_info.nodes[i].numberOfProcessors; j++) {
        if (myProcessorId == numa_info.nodes[i].processors[j]) {
          myWorkerNode = i;
          break;
        }
      }
    }
  }

  //    printf("Core %d on node %d %d \n",myProcessorId, myNode, myWorkerNode);

  if (myNode >= 0) {
    for (int j = 0; j < rcgd->numberOfProjections; j++) {
      float *cursor = padding->buffern[myNode][j] + padding->startOffset;

      for (int i = 0; i < rcgd->imageHeight; i++) {
        memcpy(cursor,
            rcgd->projectionBuffer[j].image + (i * rcgd->imageWidth),
            rcgd->imageWidth * sizeof(float));
        cursor += padding->lineOffset;
      }

      //            printf(" Master Core %d on node %d \n",myProcessorId, myNode);
      projectionBuffer[j].image  = padding->buffern[myNode][j] + padding->startOffset;
      projectionBuffer[j].matrix = rcgd->projectionBuffer[j].matrix;
      projectionBuffer[j].id     = rcgd->projectionBuffer[j].id;
    }
  } else {
    for (int j = 0; j < rcgd->numberOfProjections; j++) {
      //            printf("Core %d on node %d \n",myProcessorId, myWorkerNode);
      projectionBuffer[j].image =
          padding->buffern[myWorkerNode][j] + padding->startOffset;
      projectionBuffer[j].matrix = rcgd->projectionBuffer[j].matrix;
      projectionBuffer[j].id     = rcgd->projectionBuffer[j].id;
    }
  }
}

void memoryUtilsZeroPadEnter(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding)
{
  padding->lineSize = rcgd->imageWidth;

  for (int j = 0; j < rcgd->numberOfProjections; j++) {
    float *cursor       = padding->buffer[j] + padding->startOffset;
    padding->savePtr[j] = rcgd->projectionBuffer[j].image;

    for (int i = 0; i < rcgd->imageHeight; i++) {
      memcpy(cursor,
          rcgd->projectionBuffer[j].image + (i * padding->lineSize),
          padding->lineSize * sizeof(float));
      cursor += padding->lineOffset;
    }

    rcgd->projectionBuffer[j].image = padding->buffer[j] + padding->startOffset;
  }

  rcgd->imageWidth = padding->lineOffset;
}

void memoryUtilsZeroPadLeave(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding)
{
  rcgd->imageWidth = padding->lineSize;

  for (int i = 0; i < rcgd->numberOfProjections; i++) {
    rcgd->projectionBuffer[i].image = padding->savePtr[i];
  }
}
