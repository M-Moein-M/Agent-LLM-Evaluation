/* Copyright (C) the cluster, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.    */
#ifndef __PARAMETER_H_
#define __PARAMETER_H_

typedef struct {
    double xlength, ylength;
    int imax, jmax;
    int itermax;
    double eps, omg, rho, gamma;
} Parameter;

void initParameter(Parameter*);
void readParameter(Parameter*, const char*);
void printParameter(Parameter*);
#endif
