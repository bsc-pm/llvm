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

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/
#include <assert.h>

void compute(const int N, const int TS, double (*a)[TS][TS])
{
    int i;
    for (i = 0; i < N; i++)
    {
        #pragma oss task inout(a[i][0;TS][0;TS])
        {
            int j, k;
            for (j = 0; j < TS; ++j)
                for (k = 0; k < TS; ++k)
                    a[i][j][k]++;
        }
    }
}

void init(const int N, const int TS, double (*a)[TS][TS])
{
    int i;
    for (i = 0; i < N; i++)
    {
        #pragma oss task out(a[i][0;TS][0;TS])
        {
            int j, k;
            for (j = 0; j < TS; ++j)
                for (k = 0; k < TS; ++k)
                    a[i][j][k] = 0.0;
        }
    }
}

void check(const int N, const int TS, double (*a)[TS][TS])
{
    int i;
    for (i = 0; i < N; i++)
    {
        #pragma oss task in(a[i][0;TS][0;TS])
        {
            int j, k;
            for (j = 0; j < TS; ++j)
                for (k = 0; k < TS; ++k)
                    assert(a[i][j][k] == 1.0);
        }
    }
}

int main(int argc, char* argv[])
{
    const int N = 10;
    const int TS = 25;

    double v[N][TS][TS];

    init(N, TS, v);
    compute(N, TS, v);
    check(N, TS, v);

    #pragma oss taskwait
    return 0;
}
