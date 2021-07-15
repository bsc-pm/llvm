// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss task for chunksize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(vla[1])
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(vla[1])
    for (int i = 0; i < 10; ++i) {}
}

// CHECK: #pragma oss task for chunksize(foo<int>())
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop for chunksize(foo<int>())
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss task for chunksize(n)
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop for chunksize(n)
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss task for chunksize(vla[1])
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }
// CHECK-NEXT: #pragma oss taskloop for chunksize(vla[1])
// CHECK-NEXT:     for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:     }

