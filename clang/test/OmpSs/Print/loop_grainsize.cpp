// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss taskloop grainsize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(vla[1])
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(vla[1])
    for (int i = 0; i < 10; ++i) {}
}

// CHECK: #pragma oss taskloop grainsize(foo<int>())
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop for grainsize(foo<int>())
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop grainsize(n)
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop for grainsize(n)
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop grainsize(vla[1])
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop for grainsize(vla[1])
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }

