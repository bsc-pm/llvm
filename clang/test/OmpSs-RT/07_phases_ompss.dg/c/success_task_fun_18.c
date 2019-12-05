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
#include <string.h>

#pragma oss task copy_inout([n]f)
void foo_failure(int *f, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        f[i] = i + 1;
    }
}

int main(int argc, char *argv[])
{
    int n = 20;
    int v[n];

    memset(v, 0, sizeof(v));

    foo_failure(v, n);
#pragma oss taskwait

    int i;
    for (i = 0; i < n; i++)
    {
        if (v[i] != (i + 1))
        {
            fprintf(stderr, "v[%d] == %d but should be %d\n", i, v[i], i + 1);
            abort();
        }
    }

    return 0;
}

