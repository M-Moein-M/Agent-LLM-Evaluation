/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef RABBITPROGRESS_H
#define RABBITPROGRESS_H

extern void rabbitProgress_init(int maxValue, int rabbitSize);
extern void rabbitProgress_progress(int pos);

#endif
