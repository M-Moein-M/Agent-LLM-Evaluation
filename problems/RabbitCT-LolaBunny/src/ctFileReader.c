/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>

#include "ctFileReader.h"

static int CurrentImage = 0;

void ctFileReaderOpenFile(char *filename, RabbitCtFileType *ctFile)
{

  ctFile->file = fopen(filename, "re");

  if (ctFile->file != NULL) {
    if (!fread((void *)&ctFile->header, sizeof(RabbitCtHeaderType), 1, ctFile->file)) {
      printf("Failed to read file header! Exit.\n");
      exit(EXIT_FAILURE);
    }
  } else {
    printf("Failed to read file header! Exit.\n");
    exit(EXIT_FAILURE);
  }

  CurrentImage      = 0;
  ctFile->imageSize = ctFile->header.imageDimension[0] * ctFile->header.imageDimension[1];

  printf("Opened CT Data file: %d projections with size %d X %d\n",
      ctFile->header.numberOfImages,
      ctFile->header.imageDimension[0],
      ctFile->header.imageDimension[1]);
}

void ctFileReaderReadGeometry(RabbitCtFileType *ctFile, double *projectionData)
{
  if (!fread((void *)projectionData,
          12 * sizeof(double),
          ctFile->header.numberOfImages,
          ctFile->file)) {
    printf("Failed to read projection matrix! Exit.\n");
    exit(EXIT_FAILURE);
  }
}

int ctFileReaderReadImage(RabbitCtFileType *ctFile, double *matrix, float *image)
{
  CurrentImage++;

  if (!fread((void *)matrix, sizeof(double), 12, ctFile->file)) {
    printf("Failed to read projection matrix! Exit.\n");
    exit(EXIT_FAILURE);
  }
  if (!fread((void *)image, sizeof(float), ctFile->imageSize, ctFile->file)) {
    printf("Failed to read projection image! Exit.\n");
    exit(EXIT_FAILURE);
  }

  return CurrentImage;
}

#if 0
void 
ctFileReader_copyProjections(double* matrix, float* image)
{
    FILE* file =  fopen("outIn.vol", "w");

    fwrite(&gfileHeader, sizeof(RabbitCtHeader),1, file);

    for (int i=0; i<numberOfImages; i++)
    {
        if (!fread((void*) matrix,sizeof(double),
                    12, ctFileReader_file))
        {
            printf("Failed to read projection matrix! Exit.\n");
            exit(EXIT_FAILURE);
        }
        if (!fread((void*) image,sizeof(float),
                    imageSize, ctFileReader_file))
        {
            printf("Failed to read projection image! Exit.\n");
            exit(EXIT_FAILURE);
        }

        fwrite(matrix, sizeof(double),12, file);
#if 0
        fwrite(image, sizeof(float),imageSize, file);
#endif
    }

    fclose(file);
}



void 
ctFileReader_extractReference()
{
    FILE* file =  NULL;
    uint64_t numberOfVoxel = 1024*1024*1024;

    uint16_t*  referenceVolume =
        (uint16_t*) malloc(numberOfVoxel * sizeof(uint16_t));

    file = fopen("outRef128.vol", "w");
    numberOfVoxel = 128 * 128 *128 ;
    if (!fread((void*) referenceVolume,sizeof(uint16_t),
                numberOfVoxel, ctFileReader_file))
    {
        printf("Failed to read projection matrix! Exit.\n");
        exit(EXIT_FAILURE);
    }
    fwrite(referenceVolume, sizeof(uint16_t),numberOfVoxel, file);
    fclose(file);

    file = fopen("outRef256.vol", "w");
    numberOfVoxel = 256 * 256 *256 ;
    if (!fread((void*) referenceVolume,sizeof(uint16_t),
                numberOfVoxel, ctFileReader_file))
    {
        printf("Failed to read projection matrix! Exit.\n");
        exit(EXIT_FAILURE);
    }
    fwrite(referenceVolume, sizeof(uint16_t),numberOfVoxel, file);
    fclose(file);

    file = fopen("outRef512.vol", "w");
    numberOfVoxel = 512 * 512 *512 ;
    if (!fread((void*) referenceVolume,sizeof(uint16_t),
                numberOfVoxel, ctFileReader_file))
    {
        printf("Failed to read projection matrix! Exit.\n");
        exit(EXIT_FAILURE);
    }
    fwrite(referenceVolume, sizeof(uint16_t),numberOfVoxel, file);
    fclose(file);

    file = fopen("outRef1024.vol", "w");
    numberOfVoxel = 1024 * 1024 *1024 ;
    if (!fread((void*) referenceVolume,sizeof(uint16_t),
                numberOfVoxel, ctFileReader_file))
    {
        printf("Failed to read projection matrix! Exit.\n");
        exit(EXIT_FAILURE);
    }
    fwrite(referenceVolume, sizeof(uint16_t),numberOfVoxel, file);
    fclose(file);
}

#endif

#if 0
/* imageNumber is one based */
void 
ctFileReader_reposition(int imageNumber)
{
    int error;
    fpos_t newPosition;
    uint64_t junk = 12* sizeof(double) + imageSize * sizeof(float);

#if 0
    if (imageNumber > numberOfImages)
#endif

    newPosition += (fpos_t) (imageNumber * junk);
    if (fsetpos(ctFileReader_file, &newPosition) < 0)
    {
        fprintf(stderr,"ERROR - [%s:%d] %s \n",__FILE__,__LINE__,strerror(errno));
    }
}
#endif

void ctFileReaderClose(RabbitCtFileType *ctFile)
{
  fclose(ctFile->file);
}
