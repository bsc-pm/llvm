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

// RUN: %oss-cxx-compile && NANOS6_CONFIG=%S/../../nanos6.toml %oss-run 2>&1 | FileCheck %s
// RUN: %oss-cxx-O2-compile && NANOS6_CONFIG=%S/../../nanos6.toml %oss-run 2>&1 | FileCheck %s

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
test_exec_faulty=config/mercurium-ompss
</testinfo>
*/


#include<assert.h>
#include<stdio.h>

struct C
{
    int n;
};

struct B
{
    C * c;
    int n;
};

struct A
{
    int n;
    B* b;

#pragma oss task inout(n,b->n, b->c->n)
    void f()
    {
        n++;
        b->n++;
        b->c->n++;
    }
};

int main()
{
    A a;
    B b;
    C c;
    c.n = 1;
    b.n = 2;
    b.c = &c;

    a.n = 3;
    a.b = &b;

    fprintf(stderr, "%p\n", &a.n);
    fprintf(stderr, "%p\n", &a.b->n);
    fprintf(stderr, "%p\n", &a.b->c->n);

    a.f();
    #pragma oss taskwait
    assert(a.n == 4);
    assert(a.b->n == 3);
    assert(a.b->c->n == 2);
}

// CHECK: [[ADDR:.*]]
// CHECK: [[ADDR1:.*]]
// CHECK: [[ADDR2:.*]]

// CHECK: start:[[ADDR]]
// CHECK: start:[[ADDR1]]
// CHECK: start:[[ADDR2]]
