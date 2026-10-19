/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
/* #####   HEADER FILE INCLUDES   ######################################### */
#include <time.h>

#include "rabbitHelper_types.h"
#include "rabbitTimer.h"

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static uint64_t CpuClock       = 0;
static uint64_t CyclesForCpuid = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static uint64_t getCpuSpeed(void)
{
#ifdef __x86_64
  int i;
  TscCounterType start;
  TscCounterType stop;
  uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
  struct timeval tv1;
  struct timeval tv2;
  struct timezone tzp;
  struct timespec delay = { 0, 800000000 }; /* calibration time: 800 ms */

  for (i = 0; i < 2; i++) {
    RDTSC(start);
    gettimeofday(&tv1, &tzp);
    nanosleep(&delay, NULL);
    RDTSC(stop);
    gettimeofday(&tv2, &tzp);

    result = MIN(result, stop.int64 - start.int64 - CyclesForCpuid);
  }

  return (result) * 1000000 /
         (((uint64_t)tv2.tv_sec * 1000000 + tv2.tv_usec) -
             ((uint64_t)tv1.tv_sec * 1000000 + tv1.tv_usec));

#elif defined(_ARCH_PPC)
  FILE *fpipe;
  char *command = "grep timebase /proc/cpuinfo | awk '{ print $3 }'";
  char buff[256];

  if (!(fpipe = (FILE *)popen(command, "r"))) {
    perror("Problems with pipe");
    exit(1);
  }

  fgets(buff, 256, fpipe);

  printf("Output: %d \n", atoi(buff));
  return (uint64_t)atoi(buff);

#else
  /* Fallback: ticks are nanoseconds, so the "clock frequency" is 1 GHz */
  return 1000000000ULL;
#endif
}

static uint64_t getCpuidCycles(void)
{
#ifdef __x86_64
  int i;
  TscCounterType start;
  TscCounterType stop;
  uint64_t result = 1000000000ULL;

  for (i = 0; i < 5; i++) {
    RDTSC(start);
    RDTSC(stop);

    result = MIN(result, stop.int64 - start.int64);
  }

  return result;
#else
  /* No serialising instruction overhead on this platform */
  return 0ULL;
#endif
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void rabbitTimer_init(void)
{
  getCpuidCycles();
  getCpuSpeed();

  CyclesForCpuid = getCpuidCycles();
  CpuClock       = getCpuSpeed();
}

void rabbitTimer_startCycles(CyclesDataType *time)
{
#ifdef __x86_64
  TscCounterType start;
  TscCounterType stop;
  getCpuidCycles();

  RDTSC(start);
  RDTSC(stop);

  time->base = CyclesForCpuid + (stop.int64 - start.int64 - CyclesForCpuid);
  RDTSC(time->start);
#else
  time->base = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  time->start.int64 = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

void rabbitTimer_stopCycles(CyclesDataType *time)
{
#ifdef __x86_64
  RDTSC(time->stop);
#elif defined(_ARCH_PPC)
  uint32_t tbl, tbu0, tbu1;
  do {
    __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__("mftb %0" : "=r"(tbl));
    __asm__ __volatile__("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  time->stop.int64 = (((uint64_t)tbu0) << 32) | tbl;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  time->stop.int64 = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

void rabbitTimer_start(TimerDataType *time)
{
  gettimeofday(&(time->before), NULL);

#ifdef DEBUG
  printf("Timer Start - Seconds: %ld \t uSeconds: %ld \n", before.tv_sec, before.tv_usec);
#endif
}

void rabbitTimer_stop(TimerDataType *time)
{
  gettimeofday(&(time->after), NULL);

#ifdef DEBUG
  printf("Timer Start - Seconds: %ld \t uSeconds: %ld \n", after.tv_sec, after.tv_usec);
#endif
}

float rabbitTimer_print(TimerDataType *time)
{
  long int sec;
  float timeDuration;

  sec          = time->after.tv_sec - time->before.tv_sec;

  timeDuration = (((sec * 1000000) + time->after.tv_usec) - time->before.tv_usec);

#ifdef VERBOSE
  printf("*******************************************\n");
  printf("TIME [ms]\t:\t %f \n", timeDuration);
  printf("*******************************************\n\n");
#endif

  return timeDuration * 0.000001F;
}

uint64_t rabbitTimer_printCycles(CyclesDataType *time)
{
  return time->stop.int64 - time->start.int64 - time->base;
}

uint64_t rabbitTimer_printCyclesTime(CyclesDataType *time)
{
  uint64_t cycles = time->stop.int64 - time->start.int64 - time->base;
  return cycles * 1000000ULL / CpuClock;
}

uint64_t rabbitTimer_getCpuClock(void)
{
  return CpuClock;
}

uint64_t rabbitTimer_getCpuidCycles(void)
{
  return CyclesForCpuid;
}
