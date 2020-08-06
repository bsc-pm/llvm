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

// RUN: %oss-cxx-compile-and-run
// RUN: %oss-cxx-O2-compile-and-run

#include <vector>
#include <cassert>

void check1(std::vector<int>& w)
{
    int *pw = w.data();
    #pragma oss task out({ pw[i], i = 0 : w.size()-1 : 2 }) shared(w)
    {
        for (int i = 0; i < w.size(); i += 2)
            w[i] = 1;
    }
    #pragma oss task out({ pw[i], i = 1 : w.size()-1 : 2 }) shared(w)
    {
        for (int i = 1; i < w.size(); i += 2)
            w[i] = 2;
    }
    #pragma oss task in({ pw[i], i = 0 : w.size()-1 : 1 }) shared(w)
    {
        for (int i = 0; i < w.size(); i += 1)
            assert(i%2 ? w[i] == 2 : w[i] == 1);
    }
    #pragma oss taskwait
}

int main() {
    std::vector<int> v(1000, 0);
    check1(v);
}
