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

// RUN: %oss-cxx-compile && NANOS6_CONFIG=%S/../../nanos6.toml %oss-run 2>&1 | FileCheck %s
// RUN: %oss-cxx-O2-compile && NANOS6_CONFIG=%S/../../nanos6.toml %oss-run 2>&1 | FileCheck %s


/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

#include<stdio.h>

struct A
{
    int x[10];
#pragma oss task out(x[i])
    void f(int i)
    {
        x[i] = 0;
    }
};

int main()
{
    A a;
    A* ptr_a = &a;

    fprintf(stderr, "%p\n", &a.x);

    a.f(0);
    ptr_a->f(1);
#pragma oss taskwait
}

// CHECK: [[ADDR:.*]]
// CHECK: start:[[ADDR]]
