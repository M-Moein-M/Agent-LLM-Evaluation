/* Copyright (C) the cluster, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.    */
#ifndef __UTIL_H_
#define __UTIL_H_
#define HLINE                                                                            \
    "----------------------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#endif // __UTIL_H_
