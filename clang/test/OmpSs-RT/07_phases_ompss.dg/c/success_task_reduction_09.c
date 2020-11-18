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

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility --variable=disable_final_clause_transformation:1")
</testinfo>
*/

#include <assert.h>
#include <stdlib.h>

void foo(int n, int *v)
{
    #pragma oss task weakreduction(+: [n]v) final(1)
    {
        #pragma oss task reduction(+: [n]v) final(1)
        {
            for(int i = 0; i < n; ++i)
                v[i]++;
        }
    }

    #pragma oss task reduction(+: [n]v) final(1)
    {
        for(int i = 0; i < n; ++i)
            v[i]++;
    }

}

int main()
{
    int i, *v;

    #pragma oss task out(v) final(1)
    {
        v = (int*)malloc(10*sizeof(int));
        for (int i = 0; i < 10; ++i)
        {
            v[i] = i;
        }
    }

    #pragma oss taskwait in(v)

    #pragma oss task weakreduction(+: [10]v) final(1)
    foo(10, v);

    foo(10, v);

    #pragma oss task in([10]v) final(1)
    {
        int sum = 0;
        for(i = 0; i < 10; ++i)
            sum += v[i];

        assert(sum == 85);
    }

    return 0;
}
