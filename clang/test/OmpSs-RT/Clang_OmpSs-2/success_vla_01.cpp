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

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

struct S {
    uint32_t x;
    S() {x = -1;};
    S(S& s, int x = 666) { this->x = 999;};
    S(const S& s, int x = 666) { this->x = 999;};
    ~S() {printf("ADIOS! %d\n", x);};
};

void align_test() {
    uint32_t n = 4;
    uint32_t vla[n];
    uint32_t vla1[n];
    uint32_t x;
    #pragma oss task firstprivate(vla, vla1, x)
    {
        if ((intptr_t)vla % 4) printf("KO...\n");
        else printf("OK!\n");
        if ((intptr_t)vla1 % 4) printf("KO...\n");
        else printf("OK!\n");
    }
    #pragma oss taskwait
}

void pod_test() {
    uint32_t n = 4;
    uint32_t array[10];
    uint32_t vla[n];
    uint32_t vla1[n];
    array[0] = array[1] = array[2] = 11;
    vla[0] = vla[1] = vla[2] = 7;
    vla1[0] = vla1[1] = vla1[2] = 4;
    // array    10*sizeof(uint32_t) |
    // vlap     sizeof(uint32_t *)  |
    // vla1p    sizeof(uint32_t *)  |
    // n        sizeof(uint32_t)    | 76 rounded to 80
    // vladim   sizeof(uint64_t)    |
    // vla1dim  sizeof(uint64_t)  <-|
    // vladata  4*sizeof(uint32_t)
    // vla1data 4*sizeof(uint32_t)
    // TOTAL:   112

    #pragma oss task firstprivate(array, vla, vla1)
    {
        printf("%ld\n", (uintptr_t)&vla1[n] - (uintptr_t)array);
        printf("T1: vla: %d %d %d\n", vla[0], vla[1], vla[2]);
        printf("T1: vla1: %d %d %d\n", vla1[0], vla1[1], vla1[2]);
        printf("T1: array: %d %d %d\n", array[0], array[1], array[2]);
    }
    #pragma oss taskwait
    #pragma oss task private(array, vla, vla1)
    {
        printf("%ld\n", (uintptr_t)&vla1[n] - (uintptr_t)array);
        printf("T2: vla: %d %d %d \n", vla[0], vla[1], vla[2]);
        printf("T2: vla1: %d %d %d \n", vla1[0], vla1[1], vla1[2]);
        printf("T2: array: %d %d %d \n", array[0], array[1], array[2]);
    }
    #pragma oss taskwait
}

void nonpod_private_test() {
    int n = 4;
    S vla[n];
    S array[4];
    for (int i = 0; i < 4; ++i) {
        vla[i].x = array[i].x = 78;
    }
    // array    4*sizeof(S)          |
    // vlap     sizeof(S *)          |
    // n        sizeof(uint32_t)     | 36 rounded to 48
    // vladim   sizeof(uint64_t)   <-|
    // vladata  4*sizeof(S)
    // TOTAL:   64
    #pragma oss task private(array, vla)
    {
        // printf("%d\n", (uintptr_t)array%sizeof(uint32_t));
        printf("%ld\n", (uintptr_t)&vla[n] - (uintptr_t)array);
        printf("%d %d\n", vla[0].x, array[0].x);
    }
    #pragma oss taskwait
    for (int i = 0; i < 4; ++i) {
        assert(vla[i].x == 78 && array[i].x == 78);
    }
}

void nonpod_firstprivate_test() {
    int n = 4;
    S vla[n];
    S array[4];
    for (int i = 0; i < 4; ++i) {
        vla[i].x = array[i].x = 43;
    }
    // array    4*sizeof(S)          |
    // vlap     sizeof(S *)          |
    // n        sizeof(uint32_t)     | 36 rounded to 48
    // vladim   sizeof(uint64_t)   <-|
    // vladata  4*sizeof(S)
    // TOTAL:   64
    #pragma oss task firstprivate(array, vla, n)
    {
        // printf("%d\n", (uintptr_t)array%sizeof(uint32_t));
        printf("%ld\n", (uintptr_t)&vla[n] - (uintptr_t)array);
        printf("%d %d\n", vla[0].x, array[0].x);
    }
    #pragma oss taskwait
    for (int i = 0; i < 4; ++i) {
        assert(vla[i].x == 43 && array[i].x == 43);
    }
}

int main() {
    align_test();
    pod_test();
    nonpod_private_test();
    nonpod_firstprivate_test();

    return 0;
}

// CHECK: OK!
// CHECK-NEXT: OK!

// CHECK: 112
// CHECK-NEXT: T1: vla: 7 7 7
// CHECK-NEXT: T1: vla1: 4 4 4
// CHECK-NEXT: T1: array: 11 11 11

// CHECK: 112
// CHECK-NEXT: T2: vla: 0 0 0
// CHECK-NEXT: T2: vla1: 0 0 0
// CHECK-NEXT: T2: array: 0 0 0

// CHECK: 64
// CHECK-NEXT: -1 -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! -1
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78
// CHECK-NEXT: ADIOS! 78

// CHECK: 64
// CHECK-NEXT: 999 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999
// CHECK-NEXT: ADIOS! 999

// CHECK: ADIOS! 43
// CHECK-NEXT: ADIOS! 43
// CHECK-NEXT: ADIOS! 43
// CHECK-NEXT: ADIOS! 43
// CHECK-NEXT: ADIOS! 43
// CHECK-NEXT: ADIOS! 43
// CHECK-NEXT: ADIOS! 43
// CHECK-NEXT: ADIOS! 43

