/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.    */
#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXLINE 4096

void initParameter(Parameter* param)
{
    param->xlength = 1.0;
    param->ylength = 1.0;
    param->imax    = 100;
    param->jmax    = 100;
    param->itermax = 1000;
    param->eps     = 0.0001;
    param->omg     = 1.8;
    param->rho     = 0.99;
}

void readParameter(Parameter* param, const char* filename)
{
    FILE* fp = fopen(filename, "r");
    char line[MAXLINE];
    int i;

    if (!fp) {
        fprintf(stderr, "Could not open parameter file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    while (!feof(fp)) {
        line[0] = '\0';
        if (fgets(line, MAXLINE, fp) == NULL && ferror(fp) != 0) {
            fprintf(stderr, "Error in fgets function\n");
        }

        for (i = 0; line[i] != '\0' && line[i] != '#'; i++)
            ;
        line[i] = '\0';

        char* tok = strtok(line, " ");
        char* val = strtok(NULL, " ");

#define PARSE_PARAM(p, f)                                                                \
    if (strncmp(tok, #p, sizeof(#p) / sizeof(#p[0]) - 1) == 0) {                         \
        param->p = f(val);                                                               \
    }
#define PARSE_STRING(p) PARSE_PARAM(p, strdup)
#define PARSE_INT(p)    PARSE_PARAM(p, atoi)
#define PARSE_REAL(p)   PARSE_PARAM(p, atof)

        if (tok != NULL && val != NULL) {
            PARSE_REAL(xlength);
            PARSE_REAL(ylength);
            PARSE_INT(imax);
            PARSE_INT(jmax);
            PARSE_INT(itermax);
            PARSE_REAL(eps);
            PARSE_REAL(omg);
            PARSE_REAL(rho);
        }
    }

    fclose(fp);
}

void printParameter(Parameter* param)
{
    printf("Parameters:\n");
    printf("Geometry data:\n");
    printf("\tDomain box size (x, y): %e, %e\n", param->xlength, param->ylength);
    printf("\tCells (x, y): %d, %d\n", param->imax, param->jmax);
    printf("Iterative solver parameters:\n");
    printf("\tMax iterations: %d\n", param->itermax);
    printf("\tepsilon (stopping tolerance) : %e\n", param->eps);
    printf("\tomega (SOR relaxation): %e\n", param->omg);
}
