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

// RUN: %oss-compile
// RUN: %oss-O2-compile

struct P {
    constexpr P() {};
    static constexpr int M = 9;
};
struct S {
    static constexpr int N = 4;
    static constexpr const P& p = P();
};

struct Q {
    enum { X = 4 };
};

static constexpr int Z = 33;
int R = 77;
int &rR = R;

int main(int argc, char *argv[])
{
    int n = 3;
    int v[3];
    int *p = v;
    S s;
    Q q;

    #pragma oss task inout([s.N]v)
    {}
    #pragma oss task inout([s.p.M]v)
    {}
    // constant expressions bug #64
    // #pragma oss task inout([S::N]v, [Z]v)
    // {}
    #pragma oss task inout([rR]v)
    {}
    #pragma oss task inout([q.X]v)
    {}
    #pragma oss task inout([sizeof(int)]p)
    {}
    #pragma oss task inout([sizeof(n)]p)
    {}

    return 0;
}
