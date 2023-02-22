// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(fpga) affinity(1)
void foo() {}

// CHECK: #pragma oss task device(fpga) affinity(1)
// CHECK-NEXT: void foo()
