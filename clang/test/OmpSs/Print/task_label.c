// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task label("hola")
void foo();

const char text[] = "asdf";
#pragma oss task label(text)
void foo1();

void bar() {
    #pragma oss task label("caracola")
    {}
}

// CHECK: #pragma oss task label("hola")
// CHECK-NEXT: void foo();

// CHECK: #pragma oss task label(text)
// CHECK-NEXT: void foo1();

// CHECK: void bar() {
// CHECK-NEXT:     #pragma oss task label("caracola")
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: }
