/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef CTFILEREADER_H
#define CTFILEREADER_H

#include "ctFileReader_types.h"

extern void ctFileReaderOpenFile(char *filename, RabbitCtFileType *data);
extern int ctFileReaderReadImage(RabbitCtFileType *ctFile, double *matrix, float *image);
extern void ctFileReaderReadGeometry(RabbitCtFileType *ctFile, double *projectionData);
extern void ctFileReaderClose(RabbitCtFileType *ctFile);

#endif /* CTFILEREADER_H */
