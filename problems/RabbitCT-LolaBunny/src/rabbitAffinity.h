/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#ifndef RABBITAFFINITY_H
#define RABBITAFFINITY_H
#include "rabbitAffinity_types.h"

extern void rabbitAffinity_init();
extern void rabbitAffinity_finalize();
extern int rabbitAffinity_processGetProcessorId();
extern int rabbitAffinity_threadGetProcessorId();
extern void rabbitAffinity_pinProcess(int processorId);
extern void rabbitAffinity_pinThread(int processorId);
extern const AffinityDomainType *rabbitAffinity_getDomain(const char *domain);
extern void rabbitAffinity_printDomains();

#endif /*RABBITAFFINITY_H*/
