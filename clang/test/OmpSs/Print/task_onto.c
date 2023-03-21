// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(fpga) onto(0x300000000)
void foo() {}

// CHECK: #pragma oss task device(fpga) onto(12884901888L)
// CHECK-NEXT: void foo()
