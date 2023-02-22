// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(smp)
void foo();
#pragma oss task device(cuda)
void foo1();
#pragma oss task device(opencl)
void foo2();
#pragma oss task device(fpga)
void foo3() {}

void bar() {
    #pragma oss task device(smp)
    {}
    #pragma oss task device(cuda)
    {}
    #pragma oss task device(opencl)
    {}
}

// CHECK: #pragma oss task device(smp)
// CHECK-NEXT: void foo();
// CHECK: #pragma oss task device(cuda)
// CHECK-NEXT: void foo1();
// CHECK: #pragma oss task device(opencl)
// CHECK-NEXT: void foo2();
// CHECK: #pragma oss task device(fpga)
// CHECK-NEXT: void foo3()
// CHECK: void bar() {
// CHECK:     #pragma oss task device(smp)
// CHECK:     #pragma oss task device(cuda)
// CHECK:     #pragma oss task device(opencl)
