/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __linux__
#include <dirent.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#endif

#include "error.h"
#include "rabbitNuma.h"

/* #####   EXPORTED VARIABLES   ########################################### */

RabbitNumaTopology numa_info;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#ifdef __linux__
#define get_mempolicy(policy, nmask, maxnode, addr, flags)                               \
  syscall(SYS_get_mempolicy, policy, nmask, maxnode, addr, flags)
#define set_mempolicy(mode, nmask, maxnode)                                              \
  syscall(SYS_set_mempolicy, mode, nmask, maxnode)
#endif

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

#ifdef __linux__
static int maxIdConfiguredNode = 0;
#endif

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int str2int(const char *str)
{
  char *endptr;
  errno = 0;
  unsigned long val;
  val = strtoul(str, &endptr, 10);

  if ((errno == ERANGE && val == LONG_MAX) || (errno != 0 && val == 0)) {
    ERROR;
  }

  if (endptr == str) {
    ERROR_MSG(No digits were found);
  }

  return (int)val;
}

#ifdef __linux__
static void setConfiguredNodes(void)
{
  DIR *dir;
  struct dirent *de;

  dir = opendir("/sys/devices/system/node");

  if (!dir) {
    maxIdConfiguredNode = 0;
  } else {
    while ((de = readdir(dir)) != NULL) {
      int nd;
      if (strncmp(de->d_name, "node", 4)) {
        continue;
      }

      nd = str2int(de->d_name + 4);

      if (maxIdConfiguredNode < nd) {
        maxIdConfiguredNode = nd;
      }
    }
    closedir(dir);
  }
}

static void nodeMeminfo(int node, uint32_t *totalMemory, uint32_t *freeMemory)
{
  FILE *fp;
  char filename[256];
  char line[256];

  snprintf(filename, sizeof(filename), "/sys/devices/system/node/node%d/meminfo", node);

  if (NULL != (fp = fopen(filename, "r"))) {
    while (fgets(line, sizeof(line), fp)) {
      unsigned long val = 0;
      if (strstr(line, "MemTotal:") && sscanf(line + 18, "%lu", &val) == 1) {
        *totalMemory = (uint32_t)val;
      } else if (strstr(line, "MemFree:") && sscanf(line + 18, "%lu", &val) == 1) {
        *freeMemory = (uint32_t)val;
      }
    }
    fclose(fp);
  } else {
    ERROR;
  }
}

static int nodeProcessorList(int node, uint32_t **list)
{
  FILE *fp;
  char filename[256];
  char buf[4096];
  int count    = 0;
  int cursor   = 0;
  int unitSize = 32; /* 8 nibbles */

  *list        = (uint32_t *)malloc(MAX_NUM_THREADS * sizeof(uint32_t));

  /* the cpumap interface should be always there */
  snprintf(filename, sizeof(filename), "/sys/devices/system/node/node%d/cpumap", node);

  if (NULL != (fp = fopen(filename, "r"))) {
    if (fgets(buf, sizeof(buf), fp) == NULL) {
      fclose(fp);
      return -1;
    }
    fclose(fp);

    /* parse comma-separated hex words from right to left */
    int len = (int)strlen(buf);
    /* strip trailing whitespace/newline */
    while (len > 0 &&
           (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' ')) {
      buf[--len] = '\0';
    }

    /* split on commas, collect token pointers in reverse order */
    char *tokens[256];
    int ntokens   = 0;
    char *saveptr = NULL;
    char *tok     = strtok_r(buf, ",", &saveptr);
    while (tok && ntokens < 256) {
      tokens[ntokens++] = tok;
      tok               = strtok_r(NULL, ",", &saveptr);
    }

    for (int i = ntokens - 1; i >= 0; i--) {
      char *endptr      = NULL;
      unsigned long val = strtoul(tokens[i], &endptr, 16);

      if ((errno != 0 && val == LONG_MAX) || (errno != 0 && val == 0)) {
        ERROR;
      }

      if (endptr == tokens[i]) {
        ERROR_MSG(No digits were found);
      }

      if (val != 0UL) {
        for (int j = 0; j < unitSize; j++) {
          if (val & (1UL << j)) {
            if (count < MAX_NUM_THREADS) {
              (*list)[count] = (j + cursor);
            } else {
              ERROR_MSG(Number Of threads too large);
            }
            count++;
          }
        }
      }
      cursor += unitSize;
    }

    return count;
  }

  /* something went wrong */
  return -1;
}
#endif /* __linux__ */

static int findProcessor(uint32_t nodeId, uint32_t coreId)
{
  int i;

  for (i = 0; i < numa_info.nodes[nodeId].numberOfProcessors; i++) {
    if (numa_info.nodes[nodeId].processors[i] == coreId) {
      return 1;
    }
  }
  return 0;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int rabbitNuma_init()
{
#ifdef __linux__
  int errno;
  uint32_t i;

  if (get_mempolicy(NULL, NULL, 0, 0, 0) < 0 && errno == ENOSYS) {
    return -1;
  }

  /* First determine maximum number of nodes */
  setConfiguredNodes();
  numa_info.numberOfNodes = maxIdConfiguredNode + 1;
  numa_info.nodes =
      (RabbitNumaNode *)malloc(numa_info.numberOfNodes * sizeof(RabbitNumaNode));

  for (i = 0; i < numa_info.numberOfNodes; i++) {
    nodeMeminfo(i, &numa_info.nodes[i].totalMemory, &numa_info.nodes[i].freeMemory);
    numa_info.nodes[i].numberOfProcessors =
        nodeProcessorList(i, &numa_info.nodes[i].processors);
  }

  if (numa_info.nodes[0].numberOfProcessors < 0) {
    return -1;
  } else {
    return 0;
  }
#else
  /* Single NUMA node with all online processors */
  int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
  int i;

  numa_info.numberOfNodes        = 1;
  numa_info.nodes                = (RabbitNumaNode *)malloc(sizeof(RabbitNumaNode));
  numa_info.nodes[0].totalMemory = 0;
  numa_info.nodes[0].freeMemory  = 0;
  numa_info.nodes[0].numberOfProcessors = nproc;
  numa_info.nodes[0].processors         = (uint32_t *)malloc(nproc * sizeof(uint32_t));

  for (i = 0; i < nproc; i++) {
    numa_info.nodes[0].processors[i] = (uint32_t)i;
  }

  return 0;
#endif
}

void rabbitNuma_setInterleaved(int *processorList, int numberOfProcessors)
{
#ifdef __linux__
  uint32_t i;
  int j;
  int ret            = 0;
  int numberOfNodes  = 0;
  unsigned long mask = 0UL;

  for (i = 0; i < numa_info.numberOfNodes; i++) {
    for (j = 0; j < numberOfProcessors; j++) {
      if (findProcessor(i, processorList[j])) {
        mask |= (1UL << i);
        numberOfNodes++;
        continue;
      }
    }
  }

  ret = set_mempolicy(MPOL_INTERLEAVE, &mask, numberOfNodes);

  if (ret < 0) {
    ERROR;
  }
#else
  (void)processorList;
  (void)numberOfProcessors;
  /* NUMA memory policy not supported on this platform */
#endif
}
