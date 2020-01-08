// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss task priority(foo<int>())
    {}
    #pragma oss task priority(n)
    {}
    #pragma oss task priority(vla[1])
    {}
}

// CHECK: #pragma oss task priority(foo<int>())
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss task priority(n)
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss task priority(vla[1])
// CHECK-NEXT:     {
// CHECK-NEXT:     }

