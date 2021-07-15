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

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

#include <stdbool.h>
#include <assert.h>

#define N 10

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int main()
{
    int x;

    // Addition

    x = 0;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(+: x)
        {
            x += 2;
        }
    }
    #pragma oss task reduction(+: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x == N*2);

    // Multiplication

    x = 10;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(*: x)
        {
            x *= 2;
        }
    }
    #pragma oss task reduction(*: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x == 10*(1 << N));

    // Substraction

    x = 100;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(-: x)
        {
            x -= 2;
        }
    }
    #pragma oss task reduction(-: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x == 100 - N*2);

    // Bitwise AND

    x = ~0;
    for (int i = 0; i < sizeof(int)*8; ++i)
    {
        #pragma oss task reduction(&: x) firstprivate(i)
        {
            if (i%2 == 0)
                x &= ~(1 << i);
        }
    }
    #pragma oss task reduction(&: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    for (int j = 0; j < sizeof(int); ++j)
    {
        assert(((unsigned char*)&x)[j] == 0xAA);
    }

    // Bitwise OR

    x = 0;
    for (int i = 0; i < sizeof(int)*8; ++i)
    {
        #pragma oss task reduction(|: x) firstprivate(i)
        {
            if (i%2 == 0)
                x |= (1 << i);
        }
    }
    #pragma oss task reduction(|: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    for (int j = 0; j < sizeof(int); ++j)
    {
        assert(((unsigned char*)&x)[j] == 0x55);
    }

    // Bitwise XOR

    x = ~0;
    for (int i = 0; i < sizeof(int)*8; ++i)
    {
        #pragma oss task reduction(^: x) firstprivate(i)
        {
            if (i%2 == 0)
                x ^= (1 << i);
        }
    }
    #pragma oss task reduction(^: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    for (int j = 0; j < sizeof(int); ++j)
    {
        assert(((unsigned char*)&x)[j] == 0xAA);
    }

    // Logical AND

    x = true;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(&&: x) firstprivate(i)
        {
            x = x && true;
        }
    }
    #pragma oss task reduction(&&: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x);

    // Logical OR

    x = false;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(||: x) firstprivate(i)
        {
            if (i%2 == 0)
                x = x || true;
            else
                x = x || false;
        }
    }
    #pragma oss task reduction(||: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x);

    // MAX

    x = 0;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(max: x) firstprivate(i)
        {
            x = MAX(x, i);
        }
    }
    #pragma oss task reduction(max: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x == N - 1);

    // MIN

    x = N;
    for (int i = 0; i < N; ++i)
    {
        #pragma oss task reduction(min: x) firstprivate(i)
        {
            x = MIN(x, i);
        }
    }
    #pragma oss task reduction(min: x)
    {
        // Empty task to test initialization
    }

    #pragma oss taskwait
    assert(x == 0);
}
