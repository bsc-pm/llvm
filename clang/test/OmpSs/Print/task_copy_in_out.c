// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(fpga) copy_in([100]i)
void foo0(int *i) {}
#pragma oss task device(fpga) copy_out([100]i)
void foo1(int *i) {}
#pragma oss task device(fpga) copy_inout([100]i)
void foo2(int *i) {}

// CHECK: #pragma oss task device(fpga) copy_in([100]i)
// CHECK-NEXT: void foo0(int *i)
// CHECK: #pragma oss task device(fpga) copy_out([100]i)
// CHECK-NEXT: void foo1(int *i)
// CHECK: #pragma oss task device(fpga) copy_inout([100]i)
// CHECK-NEXT: void foo2(int *i)
