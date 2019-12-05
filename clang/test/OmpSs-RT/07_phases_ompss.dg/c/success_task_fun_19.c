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
// XFAIL: *

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>

*/
#include <stdlib.h>
#include <stdio.h>

typedef int DATA_TYPE;

#pragma oss task inout( [global_size]g_data )
static void add(
        DATA_TYPE * restrict g_data,
        const unsigned global_size,
        const unsigned inc_factor
        )
{
    int i;
    for (i = 0; i < global_size; i++)
    {
        g_data[i] += inc_factor;
    }
}

#pragma oss task in( [global_size]g_data )
static void add_check(
        DATA_TYPE * restrict g_data,
        const unsigned global_size,
        const unsigned base_position,
        const unsigned inc_factor
        )
{
    int i;
    for (i = 0; i < global_size; i++)
    {
        if (g_data[i] != (i + base_position + inc_factor + 1))
        {
            fprintf(stderr, "After dependency fulfilled: position a[%d] should be %d but it is %d\n", i, 
                    i + base_position + inc_factor + 1,
                    g_data[i]);
            abort();
        }
    }
}

enum { SIZE = 64 };

int main(int argc, char *argv[])
{
    DATA_TYPE a[SIZE];

    int i;
    for (i = 0; i < SIZE; i++)
    {
        a[i] = i + 1;
    }

    add(a, SIZE/2, 2);
    add(&(a[SIZE/2]), SIZE/2, 4);

#pragma oss taskwait

    for (i = 0; i < SIZE/2; i++)
    {
        if (a[i] != (i+ 1 + 2))
        {
            fprintf(stderr, "After taskwait: position a[%d] should be %d but it is %d\n", i, i + 3, a[i]);
            abort();
        }
    }

    for (i = SIZE/2; i < SIZE; i++)
    {
        if (a[i] != (i+ 1 + 4))
        {
            fprintf(stderr, "After taskwait: position a[%d] should be %d but it is %d\n", i, i + 3, a[i]);
            abort();
        }
    }

    // Now with tasks

    for (i = 0; i < SIZE; i++)
    {
        a[i] = i + 1;
    }

    add(a, SIZE/2, 2);
    add(&(a[SIZE/2]), SIZE/2, 4);
    add_check(a, SIZE/2, 0, 2);
    add_check(&(a[SIZE/2]), SIZE/2, SIZE/2, 4);

#pragma oss taskwait

    return 0;
}
