// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task wait
void foo();

void bar() {
    #pragma oss task wait
    {}
}

// CHECK: #pragma oss task wait
// CHECK-NEXT: void foo();

// CHECK: void bar() {
// CHECK-NEXT:     #pragma oss task wait
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: }
