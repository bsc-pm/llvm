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
test_CFLAGS=-std=gnu99
</testinfo>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#pragma oss target device(smp) copy_deps
#pragma oss task inout(a[0;10])
void foo1(int* a)
{
    fprintf(stderr, "foo1\n");
    for (int i = 0; i < 10; i++) 
        a[i] = i;
}

#pragma oss target device(smp) copy_deps
#pragma oss task in(a[0;10])
void foo2(int* a)
{
    fprintf(stderr, "foo2\n");
    for (int i = 0; i < 10; i++) 
    {
        if (a[i] != i) abort();
    }
}

int main (int argc, char*argv)
{
    int v[10];
    memset(v, 0, sizeof(v));

    for (int j = 0; j < 10; j++)
    {
        foo1(&v[0]);
        foo2(v);
    }

#pragma oss taskwait
    return 0;
}
