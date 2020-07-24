
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

// Test that lowering builds the instructions taking into
// account the signedness and signextension

#include <assert.h>
int main() {
   int lb = 1000;
   int ub = 0;
   int step = 1;

   int real_lb = 1;
   int real_ub = 233;
   int array[real_ub + real_lb];

   for (int i = real_lb; i < real_ub; ++i) array[i] = 0;

   // test outline code
   #pragma oss task for shared(array)
   for (unsigned char i = 1000; i > 0; i -= step) {
       array[i]++;
   }
   #pragma oss taskwait

   // test final code
   #pragma oss task final(1) shared(array)
   #pragma oss task for
   for (unsigned char i = 1000; i > 0; i -= step) {
       array[i]++;
   }
   #pragma oss taskwait

   for (int i = real_lb; i < real_ub; ++i)
       assert(array[i] == 2);
}
