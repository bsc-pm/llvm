// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

void bar() {
    int i;
    #pragma oss taskiter unroll(2)
    for (i = 0; i < 10; ++i) {}
    #pragma oss taskiter unroll(2)
    while (i < 10) {}
}

// CHECK:    #pragma oss taskiter unroll(2)
// CHECK-NEXT:        for (i = 0; i < 10; ++i) {
// CHECK-NEXT:        }
// CHECK-NEXT:    #pragma oss taskiter unroll(2)
// CHECK-NEXT:        while (i < 10)
// CHECK-NEXT:            {
// CHECK-NEXT:            }
