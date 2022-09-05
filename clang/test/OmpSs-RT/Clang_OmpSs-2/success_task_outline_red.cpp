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

#include <assert.h>
struct S {
  #pragma oss declare reduction(asdf : int : omp_out += omp_in) initializer(omp_priv = 0)
};

#pragma oss task reduction(S::asdf : [1]p)
void foo(int *p) {
  (*p)++;
}

int main() {
  int x = 0;
  for (int i = 0; i < 1e5; ++i)
    foo(&x);
  #pragma oss taskwait
  assert(x == 1e5);
}
