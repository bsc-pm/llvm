// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    // #pragma oss task for collapse(1)
    // for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop collapse(1)
    for (int i = 0; i < 10; ++i) {}
    // #pragma oss taskloop for collapse(1)
    // for (int i = 0; i < 10; ++i) {}
}

// CHECK:    #pragma oss taskloop collapse(1)
// CHECK-NEXT:        for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:        }

