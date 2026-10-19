#!/bin/bash
# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved. This file is part of RabbitCT.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.

TOOLCHAIN="${TOOLCHAIN:-CLANG}"
RUNNER="./rabbitRunner-${TOOLCHAIN}"
INPUT="./RabbitInput/RabbitInput.rct"
VARIANT="LolaBunny"

usage() {
    echo "Usage: $0 -s <size> [-v <variant>] [-n <numProc>] [-o <outfile>]"
    echo ""
    echo "  -s <size>      Problem size: 128, 256, 512, 1024"
    echo "  -v <variant>   Algorithm variant (default: ${VARIANT})"
    echo "                 Available: LolaBunny"
    echo "  -n <numProc>   Number of processors for likwid-pin (optional)"
    echo "  -o <outfile>   Output volume file (optional)"
    echo ""
    echo "Environment: TOOLCHAIN (default: CLANG)"
    exit 1
}

while getopts "s:v:n:o:h" opt; do
    case $opt in
        s) SIZE="$OPTARG" ;;
        v) VARIANT="$OPTARG" ;;
        n) NUM_PROC="$OPTARG" ;;
        o) VOLOUT="-o $OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$SIZE" ]; then
    echo "Error: -s <size> is required"
    usage
fi

if [ ! -x "$RUNNER" ]; then
    echo "Error: Runner not found: $RUNNER"
    echo "Build with: make TOOLCHAIN=$TOOLCHAIN"
    exit 1
fi

REFVOL="./RabbitInput/Reference${SIZE}.vol"
CHECK=""
if [ -f "$REFVOL" ]; then
    CHECK="-c $REFVOL"
fi

CMD="$RUNNER -b 1 -m $VARIANT -i $INPUT $CHECK $VOLOUT -s $SIZE"

if [ -n "$NUM_PROC" ]; then
    PIN="${PIN:-likwid-pin}"
    CMD="$PIN -c N:0-${NUM_PROC} $CMD"
fi

echo "$CMD"
exec $CMD
