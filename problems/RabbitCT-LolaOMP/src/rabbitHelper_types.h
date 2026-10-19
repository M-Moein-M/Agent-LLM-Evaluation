/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef RABBITHELPER_TYPES_H
#define RABBITHELPER_TYPES_H

/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdint.h>

#include "memoryUtils_types.h"

/* #####   EXPORTED MACROS   ############################################## */

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#define TRUE 1
#define FALSE 0

#define HLINE "-------------------------------------------------------------\n"
#define SLINE "*************************************************************\n"

#define LLU_CAST (unsigned long long)

#endif /*RABBITHELPER_TYPES_H*/
