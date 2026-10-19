/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif

#include "error.h"
#include "rabbitAffinity.h"

/* #####   EXPORTED VARIABLES   ########################################### */

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#ifdef __linux__
#define gettid() syscall(SYS_gettid)
#endif

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

#ifdef __linux__
static int getProcessorID(cpu_set_t *cpu_set)
{
  int processorId;

  for (processorId = 0; processorId < MAX_NUM_THREADS; processorId++) {
    if (CPU_ISSET(processorId, cpu_set)) {
      break;
    }
  }
  return processorId;
}
#endif

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int rabbitAffinity_processGetProcessorId()
{
#ifdef __linux__
  int ret;
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  ret = sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpu_set);

  if (ret < 0) {
    ERROR;
  }

  return getProcessorID(&cpu_set);
#else
  return 0;
#endif
}

int rabbitAffinity_threadGetProcessorId()
{
#ifdef __linux__
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  sched_getaffinity(gettid(), sizeof(cpu_set_t), &cpu_set);

  return getProcessorID(&cpu_set);
#else
  return 0;
#endif
}

void rabbitAffinity_pinThread(int processorId)
{
#ifdef __linux__
  int ret;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();
  CPU_ZERO(&cpuset);
  CPU_SET(processorId, &cpuset);
  ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

  if (ret != 0) {
    ERROR;
  }
#elif defined(__APPLE__)
  thread_affinity_policy_data_t policy = { processorId };
  thread_policy_set(pthread_mach_thread_np(pthread_self()),
      THREAD_AFFINITY_POLICY,
      (thread_policy_t)&policy,
      THREAD_AFFINITY_POLICY_COUNT);
#endif
}

void rabbitAffinity_pinProcess(int processorId)
{
#ifdef __linux__
  int ret;
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);
  CPU_SET(processorId, &cpuset);
  ret = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

  if (ret < 0) {
    ERROR;
  }
#elif defined(__APPLE__)
  thread_affinity_policy_data_t policy = { processorId };
  thread_policy_set(pthread_mach_thread_np(pthread_self()),
      THREAD_AFFINITY_POLICY,
      (thread_policy_t)&policy,
      THREAD_AFFINITY_POLICY_COUNT);
#endif
}
