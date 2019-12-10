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

// RUN: %oss-cxx-compile-and-run

/*
<testinfo>
test_generator=("config/mercurium-ompss c++11" "config/mercurium-ompss-2 openmp-compatibility c++11")
test_nolink=yes
</testinfo>
*/

template <int t_arg>
struct A;

template <int t_arg>
void bar(A<t_arg> &a)
{
    #pragma oss task firstprivate(a)
    {
    }
}

template <int t_arg>
struct A
{
    static constexpr int size = t_arg;

    int member[size];

    void foo()
    {
        bar(*this);
    }

};

template <>
void bar<1>(A<1> &a)
{
}

int main()
{
    A<5> a;
    a.foo();
}
