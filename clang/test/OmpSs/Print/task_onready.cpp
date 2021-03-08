// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

#pragma oss task onready(x)
void foo(int x);

void bar(int n) {
    int vla[n];
    #pragma oss task onready(foo<int>())
    {}
    #pragma oss task onready(n)
    {}
    #pragma oss task onready(vla[1])
    {}
}

// CHECK: #pragma oss task onready(x)
// CHECK-NEXT: void foo(int x);

// CHECK: #pragma oss task onready(foo<int>())
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss task onready(n)
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss task onready(vla[1])
// CHECK-NEXT:     {
// CHECK-NEXT:     }

