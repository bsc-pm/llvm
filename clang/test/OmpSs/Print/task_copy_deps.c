// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(fpga) copy_deps
void foo() {}

// CHECK: #pragma oss task device(fpga) copy_deps
// CHECK-NEXT: void foo()
