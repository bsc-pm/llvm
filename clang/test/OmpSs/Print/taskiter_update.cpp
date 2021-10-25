// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

void bar() {
    int i;
    #pragma oss taskiter update
    for (i = 0; i < 10; ++i) {}
}

// CHECK:    #pragma oss taskiter update
// CHECK-NEXT:        for (i = 0; i < 10; ++i) {
// CHECK-NEXT:        }
