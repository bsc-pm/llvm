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

// This test checks that discrete and range dependencies are equivalent

#include <assert.h>

#define N 10
int array[N][N];

int main() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            array[i][j] = 0;

    int start = 0;
    // Put a dependency over the diagonal
    #pragma oss task out( { array[i][j], i={start + 0, \
                                        start + 1, \
                                        start + 2, \
                                        start + 3, \
                                        start + 4, \
                                        start + 5, \
                                        start + 6, \
                                        start + 7, \
                                        start + 8, \
                                        start + 9}, j=i+1:N-1 } )
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (j > i)
                    array[i][j] = 1;

    }
    // Put a dependency under the diagonal
    #pragma oss task out( { array[i][j], i={start + 0, \
                                        start + 1, \
                                        start + 2, \
                                        start + 3, \
                                        start + 4, \
                                        start + 5, \
                                        start + 6, \
                                        start + 7, \
                                        start + 8, \
                                        start + 9}, j=0:i } )
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (j <= i)
                    array[i][j] = 2;

    }

    // Put a dependency over the diagonal
    #pragma oss task in( { array[i][j], i=0;N, j=i+1:N-1 } )
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (j > i)
                    assert(array[i][j] == 1);
    }

    // Put a dependency under the diagonal
    #pragma oss task in( { array[i][j], i=0;N, j=0:i } )
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (j <= i)
                    assert(array[i][j] == 2);
    }

    #pragma oss taskwait
}

