/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "algorithmRegistry.h"
#include "ctFileReader.h"
#include "ctFileReader_types.h"
#include "likwid-marker.h"
#include "memoryUtils.h"
#include "rabbitProgress.h"
#include "rabbitTimer.h"
#include "types.h"

#define US_ROUND(x) (floorf((x) + 0.5f))

#define HELP_MSG                                                                         \
  printf("RabbitRunner --  Version 0.1 \n\n");                                           \
  printf("Supported Options:\n");                                                        \
  printf("Example usage: ./RabbitRunner -i filename \n");                                \
  printf("-h\t Help message\n");                                                         \
  printf("-v\t verbose output\n");                                                       \
  printf("-i\t Specify input filename\n");                                               \
  printf("-a\t Specify geometry filename\n");                                            \
  printf("-s\t problem size: 128, 256, 512, 1024\n");                                    \
  printf("-b\t buffer size of projection images\n");                                     \
  printf("-c\t check error\n");                                                          \
  printf("-C\t Specify line clipping filename\n");                                       \
  printf("-o\t output volume image\n");                                                  \
  printf("-m\t algorithm name\n\n");                                                     \
  algorithmRegistryList()

#ifdef LIKWID_PERFMON
#define PROFILE(call)                                                                    \
  _Pragma("omp parallel")                                                                \
  {                                                                                      \
    LIKWID_MARKER_START("LOLA");                                                         \
  }                                                                                      \
  rabbitTimer_startCycles(&cycleData);                                                   \
  call;                                                                                  \
  rabbitTimer_stopCycles(&cycleData);                                                    \
  _Pragma("omp parallel")                                                                \
  {                                                                                      \
    LIKWID_MARKER_STOP("LOLA");                                                          \
  }
#else /* LIKWID_PERFMON */
#define PROFILE(call)                                                                    \
  rabbitTimer_startCycles(&cycleData);                                                   \
  call;                                                                                  \
  rabbitTimer_stopCycles(&cycleData);
#endif /* LIKWID_PERFMON */

static float volumeResolution(const int problemSize)
{
  float rv = 0.0f;

  switch (problemSize) {
  case 128:
    rv = 2.0f;
    break;
  case 256:
    rv = 1.0f;
    break;
  case 512:
    rv = 0.5f;
    break;
  case 1024:
    rv = 0.25f;
    break;
  default:
    printf("Wrong problem size, check again!");
    exit(EXIT_FAILURE);
  }

  return rv;
}

int main(int argc, char **argv)
{
  int c;
  int size                      = 0;
  int bufferSize                = 1;
  uint64_t time                 = 0.0;
  int optVerbose                = 0;
  char *inFilename              = NULL;
  char *outFilename             = NULL;
  char *pgmFilename             = NULL;
  char *refFilename             = NULL;
  char *clipFilename            = NULL;
  char *analysisFile            = NULL;
  char *modulePath              = NULL;
  int globalNumberOfProjections = 0;
  RabbitCtGlobalData data;
  double error           = 0.0;
  double maxError        = 0.0;
  uint64_t numberOfVoxel = 0;
  RabbitCtFileType inputFile;

  if (argc == 1) {
    HELP_MSG;
    exit(EXIT_SUCCESS);
  }

  while ((c = getopt(argc, argv, "+o:p:b:a:C:c:m:s:i:hv")) != -1) {
    switch (c) {
    case 'h':
      HELP_MSG;
      exit(EXIT_SUCCESS);
    case 'v':
      optVerbose = 1;
      break;
    case 'i':
      inFilename = (char *)malloc(strlen(optarg) + 1);
      strcpy(inFilename, optarg);
      break;
    case 'a':
      analysisFile = (char *)malloc(strlen(optarg) + 1);
      strcpy(analysisFile, optarg);
      break;
    case 's':
      size = atoi(optarg);
      break;
    case 'b':
      bufferSize = atoi(optarg);
      break;
    case 'C':
      clipFilename = (char *)malloc(strlen(optarg) + 1);
      strcpy(clipFilename, optarg);
      break;
    case 'c':
      refFilename = (char *)malloc(strlen(optarg) + 1);
      strcpy(refFilename, optarg);
      break;
    case 'o':
      outFilename = (char *)malloc(strlen(optarg) + 1);
      strcpy(outFilename, optarg);
      break;
    case 'p':
      pgmFilename = (char *)malloc(strlen(optarg) + 1);
      strcpy(pgmFilename, optarg);
      break;
    case 'm':
      modulePath = (char *)malloc(strlen(optarg) + 1);
      strcpy(modulePath, optarg);
      break;
    default:
      HELP_MSG;
      exit(EXIT_FAILURE);
    }
  }

  if (!inFilename || !modulePath) {
    HELP_MSG;
    exit(EXIT_FAILURE);
  }

  printf("\n");
  printf("  (\\(\\         ____       _     _     _ _    ____ _____\n");
  printf("  ( -.-)      |  _ \\ __ _| |__ | |__ (_) |_ / ___|_   _|\n");
  printf("  o_(\")(\")    | |_) / _` | '_ \\| '_ \\| | __| |     | |  \n");
  printf("              |  _ < (_| | |_) | |_) | | |_| |___  | |  \n");
  printf("              |_| \\_\\__,_|_.__/|_.__/|_|\\__|\\____| |_|  \n");
  printf("\n");

  LIKWID_MARKER_INIT;
  _Pragma("omp parallel")
  {
    LIKWID_MARKER_REGISTER("LOLA");
  }
  rabbitTimer_init();

  /********************************************************
   * LOAD ALGORITHM
   * *****************************************************/
  if (!algorithmRegistryFind(modulePath)) {
    exit(EXIT_FAILURE);
  }

  data.voxelSize   = volumeResolution(size);
  data.problemSize = size;
  data.O_Index     = -0.5f * data.voxelSize * ((float)(data.problemSize) - 1.0f);

  numberOfVoxel    = size * size * size;
  memoryUtilsAllocate(&data.volumeData, numberOfVoxel);

  ctFileReaderOpenFile(inFilename, &inputFile);

  globalNumberOfProjections = inputFile.header.numberOfImages;
  data.imageWidth           = inputFile.header.imageDimension[0];
  data.imageHeight          = inputFile.header.imageDimension[1];
  data.numberOfProjections  = 0;

  rabbitProgress_init(globalNumberOfProjections, size);

  /* NULL if -C was not used */
  data.clipFilename = clipFilename;

  /********************************************************
   * OPTIONAL: LOAD GLOBAL GEOMETRY
   * *****************************************************/
  if (analysisFile != NULL) {
    data.numberOfProjections = globalNumberOfProjections;
    data.globalGeometry =
        (double *)malloc(globalNumberOfProjections * 12 * sizeof(double));
    RabbitCtFileType geometryFile;
    ctFileReaderOpenFile(analysisFile, &geometryFile);
    ctFileReaderReadGeometry(&geometryFile, data.globalGeometry);
    ctFileReaderClose(&geometryFile);
  }

  /********************************************************
   * PREPARE ALGORITHM
   * *****************************************************/
  if (FncPrepareAlgorithm != NULL) {
    CyclesDataType cycleData;
    rabbitTimer_startCycles(&cycleData);
    FncPrepareAlgorithm(&data);
    rabbitTimer_stopCycles(&cycleData);

    if (optVerbose) {
      printf("Prepare Algorithm Timing: %g sec\n",
          (double)rabbitTimer_printCyclesTime(&cycleData) / (double)1000000.0);
    }
  }

  /********************************************************
   * BACKPROJECTION LOOP
   * *****************************************************/
  {
#ifdef _OPENMP
    printf("Running threaded with %d threads ... this may take some time.\n",
        omp_get_max_threads());
#else
    printf("Running sequential ... this may take some time.\n");
#endif
    if (optVerbose) {
      printf("Processing %d projections with size %d X %d\n",
          globalNumberOfProjections,
          data.imageWidth,
          data.imageHeight);

      printf("Buffering %d projections per call.\n", bufferSize);
      printf("Parameters:\n");
      printf("Problem size: %d\n", data.problemSize);
      printf("Voxel size: %g\n", data.voxelSize);
    }

    int counter = globalNumberOfProjections;
    CyclesDataType cycleData;
    data.numberOfProjections = bufferSize;
    data.projectionBuffer    = (Projection *)malloc(bufferSize * sizeof(Projection));

    for (int i = 0; i < bufferSize; i++) {
      data.projectionBuffer[i].matrix = (double *)malloc(12 * sizeof(double));
      memoryUtilsAllocate(
          &data.projectionBuffer[i].image, data.imageWidth * data.imageHeight);
    }

    while (counter > 0) {
      if (optVerbose) {
        printf("Processing buffered run - %d left\n", counter);
      }

      /********************************************************
       * BUFFER PROJECTIONS
       * *****************************************************/
      for (int i = 0; i < bufferSize; i++) {
        data.projectionBuffer[i].id = ctFileReaderReadImage(
            &inputFile, data.projectionBuffer[i].matrix, data.projectionBuffer[i].image);
      }
      PROFILE(FncAlgorithmIteration(&data))
      time += rabbitTimer_printCyclesTime(&cycleData);

      counter -= bufferSize;
      rabbitProgress_progress(globalNumberOfProjections - counter);
    }
  }

  FncFinishAlgorithm(&data);

  /********************************************************
   * OPTIONAL WRITE PGM SLICE IMAGE
   * *****************************************************/
  if (pgmFilename != NULL) {
    FILE *pgmFile = fopen(pgmFilename, "we");
    if (pgmFile != NULL) {
      float *recVolume    = data.volumeData;
      float *hus          = inputFile.header.HUScalingFactors;
      const int sliceSize = data.problemSize * data.problemSize;
      float *slice        = recVolume + (data.problemSize / 2 - 1) * sliceSize;
      uint16_t *row       = (uint16_t *)malloc(sliceSize * sizeof(uint16_t));

      fprintf(pgmFile, "P5\n%d %d\n4095\n", data.problemSize, data.problemSize);

      for (int i = 0; i < sliceSize; i++) {
        float huAx = (slice[i] > 0.0f) ? US_ROUND(hus[0] * slice[i] + hus[1]) : 0.0f;
        if (huAx > 4095.0f)
          huAx = 4095.0f;
        uint16_t val = (uint16_t)huAx;
        /* PGM 16-bit requires big-endian byte order */
        row[i] = (uint16_t)((val >> 8) | (val << 8));
      }
      fwrite(row, sizeof(uint16_t), sliceSize, pgmFile);
      free(row);
      fclose(pgmFile);
    }
  }

  /********************************************************
   * OPTIONAL VERIFY RESULT
   * *****************************************************/
  if (refFilename != NULL) {
    const int magicNum        = 591984;
    const int magicNumEnd     = 489195;
    uint16_t *referenceVolume = (uint16_t *)malloc(numberOfVoxel * sizeof(uint16_t));
    uint64_t *runtime = (uint64_t *)malloc(sizeof(uint64_t) * globalNumberOfProjections);
    FILE *file        = fopen(refFilename, "re");
    FILE *resFile     = fopen("result.rctd", "we");

    if (file != NULL) {
      if (!fread((void *)referenceVolume, sizeof(uint16_t), numberOfVoxel, file)) {
        printf("Failed to read reference file !\n");
      }
    } else {
      printf("Failed to read reference file!\n");
    }

    fclose(file);

    float *recVolume = data.volumeData;
    float *hus       = inputFile.header.HUScalingFactors;
    float huAx;
    const int numBins = 11;
    // 0-1, 1-2, 2-3, 3-4,  ... 9-10, >10
    uint32_t errorHist[numBins];
    memset(errorHist, 0, numBins * sizeof(uint32_t));

    for (uint32_t i = 0; i < numberOfVoxel; i++) {
      if (recVolume[i] > 0.0f) {
        huAx = US_ROUND(hus[0] * recVolume[i] + hus[1]);
      } else {
        huAx = 0.0f;
      }

      if (huAx > 4095.0f) {
        huAx = 4095.0f;
      }

      const float cerr   = (huAx - (float)(referenceVolume[i]));
      const float absErr = fabsf(cerr);

      maxError           = MAX(maxError, (double)absErr);
      error += cerr * cerr;
      recVolume[i]     = huAx;
      uint32_t histIdx = (uint32_t)floorf(absErr);

      if (histIdx > numBins - 1) {
        histIdx = numBins - 1;
      }

      ++errorHist[histIdx];
    }
    error /= (double)numberOfVoxel;

    const double psnr = 10.0 * log10(4095.0 * 4095.0 / error);
    printf("\n--------------------------------------------------------------\n");
    printf("Quality of reconstructed volume:\n");
    printf("Root Mean Squared Error: %g HU\n", sqrt(error));
    printf("Mean Squared Error: %g HU^2\n", error);
    printf("Max. Absolute Error: %g HU\n", maxError);

    if (error > 1e-8) {
      printf("PSNR: %g db\n", psnr);
    } else {
      printf("PSNR: INF db\n");
    }

    /* the reference RabbitCtRunner implementation saves the
     * runtime of every image. So we compute the average time
     * per image here since we only measure the total runtime.*/
    uint64_t meantime = time / globalNumberOfProjections;
    printf("mean time [usec]: %lld \n", LL_CAST meantime);

    for (int i = 0; i < globalNumberOfProjections; i++) {
      runtime[i] = meantime;
    }

    fwrite(&magicNum, sizeof(int), 1, resFile);
    fwrite(&size, sizeof(int), 1, resFile);
    fwrite(&inputFile.header.numberOfImages, sizeof(uint32_t), 1, resFile);
    fwrite(runtime, sizeof(uint64_t), globalNumberOfProjections, resFile);
    fwrite(recVolume + (data.problemSize / 2 - 1) * data.problemSize * data.problemSize,
        sizeof(float),
        data.problemSize * data.problemSize,
        resFile);
    fwrite(&error, sizeof(double), 1, resFile);
    fwrite(&maxError, sizeof(double), 1, resFile);
    fwrite(&errorHist, sizeof(uint32_t), numBins, resFile);
    fwrite(&magicNumEnd, sizeof(int), 1, resFile);

    fclose(resFile);

    /* check error of timing conversion */
    {
      uint64_t newtime = 0ULL;

      for (int i = 0; i < globalNumberOfProjections; i++) {
        newtime += runtime[i];
      }

      printf("Error [usec]: %lld \n", LL_CAST(int64_t) newtime - (int64_t)time);
    }
  }

  /********************************************************
   * OPTIONAL  WRITE RESULT
   * *****************************************************/
  if (outFilename != NULL) {
    FILE *file = fopen(outFilename, "we");
    fwrite(data.volumeData, sizeof(float), numberOfVoxel, file);
    fclose(file);
  }

  /********************************************************
   * OUTPUT RESULT
   * *****************************************************/

  double timeMean = (double)time / (double)(1000.0 * (double)globalNumberOfProjections);
  printf("\n--------------------------------------------------------------\n");
  printf("Runtime statistics:\n");
  printf("Total: %g s\n", (double)time / (double)1000000.0);
  printf("Average: %g ms \n\n", timeMean);

  LIKWID_MARKER_CLOSE;

  return EXIT_SUCCESS;
}
