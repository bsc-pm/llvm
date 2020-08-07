/*--------------------------------------------------------------------
  (C) Copyright 2006-2013 Barcelona Supercomputing Center
                          Centro Nacional de Supercomputacion

  This file is part of Mercurium C/C++ source-to-source compiler.

  See AUTHORS file in the top level directory for information
  regarding developers and contributors.

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.

  Mercurium C/C++ source-to-source compiler is distributed in the hope
  that it will be useful, but WITHOUT ANY WARRANTY; without even the
  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public
  License along with Mercurium C/C++ source-to-source compiler; if
  not, write to the Free Software Foundation, Inc., 675 Mass Ave,
  Cambridge, MA 02139, USA.
--------------------------------------------------------------------*/

// RUN: %oss-compile-and-run
// RUN: %oss-O2-compile-and-run

#include <assert.h>

#define N 10

#pragma oss task out( { p[i][j], i = init1:ub1:step1, j =init2:ub2:step2 } )
void gen(int (*p)[N], int init1, int ub1, int step1, int init2, int ub2, int step2, int val) {
    for (int i = init1; i <= ub1; i += step1)
        for (int j = init1; j <= ub1; j += step2)
            p[i][j] = val;
}
#pragma oss task in( { p[i][j], i = init1:ub1:step1, j =init2:ub2:step2 } )
void consume(int (*p)[N], int init1, int ub1, int step1, int init2, int ub2, int step2, int val) {
    for (int i = init1; i <= ub1; i += step1)
        for (int j = init1; j <= ub1; j += step2)
            assert(p[i][j] == val);
}

int main() {
    int M[N][N];
    gen(M, 0, N-1, 2, 0, N-1, 2, 7);
    gen(M, 1, N-1, 2, 1, N-1, 2, 8);
    consume(M, 0, N-1, 2, 0, N-1, 2, 7);
    consume(M, 1, N-1, 2, 1, N-1, 2, 8);
    #pragma oss taskwait
}
