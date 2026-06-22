#pragma once

// Order-preserving stream compaction.
//
// Copy every element of `d_in` (length `n`, device memory) whose value is
// strictly greater than `threshold` into `d_out` (device memory, capacity `n`),
// PRESERVING the elements' original relative order. Return the number of
// elements written.
//
// `d_out[0 .. return_value)` must hold the kept elements in the same order they
// appear in `d_in`. The return value is the count on the host.
int compact(const int* d_in, int n, int threshold, int* d_out);
