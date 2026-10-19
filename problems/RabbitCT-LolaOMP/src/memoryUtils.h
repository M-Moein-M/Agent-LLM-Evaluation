/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef MEMORYUTILS_H
#define MEMORYUTILS_H
#include <stdint.h>

#include "memoryUtils_types.h"
#include "rabbitCt.h"
extern void memoryUtilsInit();
extern void memoryUtilsAllocate(float **ptr, uint64_t size);
extern void memoryUtilsZeroPadInit(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding);
extern void memoryUtilsZeroPadAllocate(
    RabbitCtGlobalData *rcgd, ZeroPaddingType *padding);
extern void memoryUtilsZeroPadEnterExp(
    RabbitCtGlobalData *rcgd, ZeroPaddingType *padding, Projection *projectionBuffer);
extern void memoryUtilsZeroPadEnter(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding);
extern void memoryUtilsZeroPadLeave(RabbitCtGlobalData *rcgd, ZeroPaddingType *padding);

#endif /*MEMORYUTILS_H*/
