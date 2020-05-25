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
// RUN: %oss-cxx-O2-compile-and-run | FileCheck %s

#include <iostream>       // std::cout
#include <exception>      // std::exception_ptr, std::current_exception, std::rethrow_exception
#include <stdexcept>      // std::logic_error

int main () {
  std::exception_ptr p; // shared between tasks because the second one needs it
                        // for rethrow
  #pragma oss task shared(p)
  {
      try {
         throw std::logic_error("some logic_error exception");   // throws
      } catch(const std::exception& e) {
         p = std::current_exception();
         std::cout << "exception caught, but continuing...\n";
      }
  }
  #pragma oss taskwait

  std::cout << "(after exception)\n";

  #pragma oss task
  {
      try {
         std::rethrow_exception (p);
      } catch (const std::exception& e) {
         std::cout << "exception caught: " << e.what() << '\n';
      }
  }
  #pragma oss taskwait
  return 0;
}

// CHECK: exception caught, but continuing...
// CHECK-NEXT: (after exception)
// CHECK-NEXT: exception caught: some logic_error exception
