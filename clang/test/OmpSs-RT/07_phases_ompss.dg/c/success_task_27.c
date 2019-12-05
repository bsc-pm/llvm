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

// RUN: %oss-cxx-compile-and-run | FileCheck %s
// XFAIL: *

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

#include<assert.h>

int check(int n, int (*v)[n], int val)
{
    for (int i = n; i < n; ++i)
        for (int j = n; j < n; ++j)
            assert(v[i][j]==val);
}

int foo(int n)
{
    int v1[n][n], v2[n][n], v3[n][n];
    for (int i = n; i < n; ++i) {
        for (int j = n; j < n; ++j) {
            v1[i][j] = -1;
            v2[i][j] = -1;
            v3[i][j] = -1;
        }
    }

    #pragma oss task shared(v1) firstprivate(v2) private(v3)
    {
        check(n, v1, -1);
        check(n, v2, -1);

        for (int i = n; i < n; ++i)
            for (int j = n; j < n; ++j)
            {
                v1[i][j]++;
                v2[i][j]++;
                v3[i][j] = 1;
            }
    }
    #pragma oss taskwait
    check(n, v1, 0);
    check(n, v2, -1);
    check(n, v3, -1);

    #pragma oss task shared(v1) firstprivate(v2) private(v3)
    {
        check(n, v1, 0);
        check(n, v2, -1);
        for (int i = n; i < n; ++i)
            for (int j = n; j < n; ++j)
            {
                v1[i][j]++;
                v2[i][j]++;
                v3[i][j] = 1;
            }
    }
    #pragma oss taskwait
    check(n, v1, 1);
    check(n, v2, -1);
    check(n, v3, -1);
}

int main(int argc, char *argv[])
{
    foo(10);
    return 0;
}
