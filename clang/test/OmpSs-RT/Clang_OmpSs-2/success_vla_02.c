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

// RUN: %oss-compile && NANOS6_CONFIG_OVERRIDE="version.dependencies=regions,$NANOS6_CONFIG_OVERRIDE" %oss-run
// RUN: %oss-O2-compile && NANOS6_CONFIG_OVERRIDE="version.dependencies=regions,$NANOS6_CONFIG_OVERRIDE" %oss-run

// This test checks that we're generating code properly for the compute_dep
// function. the shape expr type of task A is int [c], but we need b to compute
// the dependency base

#include <assert.h>
int main() {
    int a, b, c;
    a = b = 5;
    c = 8;
    int vla[a][b];
    vla[1][2] = -1;
    #pragma oss task out([c](vla[0])) // A, flatten vla
    { vla[1][2] = 43; }
    #pragma oss task in(vla[1][2])
    { assert(vla[1][2] == 43); }
    #pragma oss taskwait
}
