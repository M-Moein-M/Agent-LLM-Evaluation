/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#ifndef ERROR_H
#define ERROR_H

#include <errno.h>
#include <string.h>

#define str(x) #x

#ifdef WARN
#define WARNING(msg)                                                                     \
  fprintf(stderr, "WARNING - [%s:%d]" str(msg) " \n", __FILE__, __LINE__)
#else
#define WARNING(msg)
#endif

#define ERROR                                                                            \
  fprintf(stderr, "ERROR - [%s:%d] %s\n", __FILE__, __LINE__, strerror(errno));          \
  exit(EXIT_FAILURE)

#define ERROR_MSG(msg)                                                                   \
  fprintf(stderr, "ERROR - [%s:%d]" str(msg) " \n", __FILE__, __LINE__);                 \
  exit(EXIT_FAILURE)

#define ERROR_PMSG(msg, var)                                                             \
  fprintf(stderr, "ERROR - [%s:%d]" str(msg) " \n", __FILE__, __LINE__, var);            \
  exit(EXIT_FAILURE)

#define CHECK_ERROR(func, msg)                                                           \
  if ((func) < 0) {                                                                      \
    fprintf(stderr,                                                                      \
        "ERROR - [%s:%d]" str(msg) " - %s \n",                                           \
        __FILE__,                                                                        \
        __LINE__,                                                                        \
        strerror(errno));                                                                \
  }

#endif /*ERROR_H*/
