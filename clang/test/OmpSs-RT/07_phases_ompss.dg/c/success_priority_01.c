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

// RUN: %oss-cxx-compile-and-run | FileCheck %s
// XFAIL: *

/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
test_compile_fail_nanos6_mercurium=yes
test_compile_fail_nanos6_imcc=yes
</testinfo>
*/
#include <omp.h>
#include <stdio.h>

// This is to ensure the high priority task is not executed twice or more
int done = 0;

#pragma oss task /*output(*depth)*/ priority(0)
void normal_task (int * var)
{
   int i;
   ++( *var );
}

#pragma oss task /*input(*depth)*/ priority(10000)
void high_task(int * var)
{
   int i;
   #pragma oss critical( my_lock )
   {
      if( done == 0 )
       *var = 0;
      done = 1;
   }
}

#define NUM_ITERS 100


int main ()
{
   int A = 0;
   int i, j;
   int check = 1;

   nanos_stop_scheduler();
   nanos_wait_until_threads_paused();

   for (j=0; j < NUM_ITERS; j++) {
      normal_task(&A);
   }

   for (i = 0; i < omp_get_num_threads(); i++) {
      high_task(&A);
   }

   nanos_start_scheduler();
   nanos_wait_until_threads_unpaused();

#pragma oss taskwait
   check = ( A != 0 );

   if ( !check ) {
       fprintf(stderr, "FAIL %d\n", A);
      return 1;
   }
   return 0;
}


