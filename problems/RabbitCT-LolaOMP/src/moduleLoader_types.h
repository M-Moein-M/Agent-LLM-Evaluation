/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef MODULELOADER_TYPES_H
#define MODULELOADER_TYPES_H

#include "rabbitCt.h"

typedef int (*FncPrepareAlgorithmType)(RabbitCtGlobalData *);
typedef int (*FncAlgorithmIterationType)(RabbitCtGlobalData *);
typedef int (*FncFinishAlgorithmType)(RabbitCtGlobalData *);

#endif
