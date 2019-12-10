/*--------------------------------------------------------------------
  (C) Copyright 2006-2012 Barcelona Supercomputing Center
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

// RUN: oss-cxx-compile-and-run
//XFAIL: *

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/
#include<assert.h>

int main(int argc, char*argv[])
{
    const int BS = 10;
    int v[BS][BS];
    for (int i = 0; i < BS; ++i)
        for (int j = 0; j < BS; ++j)
            v[i][j] = 0;


    int *ptr =(int*)&v;
    #pragma omp task
    {
        int x = BS;
        typedef int (*ptrArray)[x];
        ptrArray m = reinterpret_cast<ptrArray>(ptr);

        double u = m[0][0] + m[1][1];

        for (int i = 0; i < x; ++i)
            for (int j = 0; j < x; ++j)
                m[i][j] = 1;
    }
    #pragma omp taskwait

    for (int i = 0; i < BS; ++i)
        for (int j = 0; j < BS; ++j)
            assert(v[i][j] == 1);
    return 0;
}
