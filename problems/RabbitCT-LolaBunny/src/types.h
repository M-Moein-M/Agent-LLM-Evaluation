/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef TYPES_H
#define TYPES_H

/* #####   HEADER FILE INCLUDES   ######################################### */
#include "rabbitHelper_types.h"
#include <stdint.h>

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
#define LL_CAST (long long)

#endif /*TYPES_H*/
