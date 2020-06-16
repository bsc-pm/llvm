// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task label("hola")
void foo();

void bar() {
    #pragma oss task label("caracola")
    {}
}

// CHECK: #pragma oss task label("hola")
// CHECK-NEXT: void foo();

// CHECK: void bar() {
// CHECK-NEXT:     #pragma oss task label("caracola")
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: }
