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
// XFAIL: *

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/
#include <stdlib.h>

template <typename T>
struct A
{
    T t;
    A() : t(0) { }
};

template <typename T>
void f(A<T> *a, int first, int last, int val)
{
#pragma omp task out(a[first:last])
    {
        int i;
        for (i = first; i <= last; i++)
        {
            a[i].t = val;
        }
    }
}

template <typename T>
void g(A<T> *a, int length, int val)
{
#pragma omp task in(a[0;length])
    {
        int i;
        for (i = 0; i < length; i++)
        {
            if (a[i].t != val)
            {
                abort();
            }
        }
    }
}

int main(int argc, char *argv[])
{
    A<int> a[10];

    for (int j = 0; j < 1000; j++)
    {
        f(a, 0, 9, j);
        g(a, 10, j);
    }

#pragma omp taskwait
}
