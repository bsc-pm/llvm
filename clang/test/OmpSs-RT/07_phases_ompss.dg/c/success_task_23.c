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

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
test_CFLAGS=-std=gnu99
test_nolink=yes
</testinfo>
*/
void dealloc_tiled_matrix(int MBS, int NBS, int M, int N, double (*a)[N/NBS][MBS][NBS]) {
        // Build a fictitious dependency structure to free the whole tiled matrix at once
        for (int i=0; i<M/MBS; i ++) {
                for (int j=0; j<N/NBS; j++) {
                        if (i != 0 || j != 0) {
                                #pragma oss task inout(a[i][j]) concurrent(a[0][0])
                                {
                                }
                        }
                }
        }

        //#pragma oss task inout(a[0][0])
        //free(a);
}


int main(int argc, char *argv[])
{
    return 0;
}
