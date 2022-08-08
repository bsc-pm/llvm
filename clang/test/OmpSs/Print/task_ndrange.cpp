// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) ndrange(1, 1, 1)
void foo();

#pragma oss task device(cuda) ndrange(N, *x, *y)
template<int N, typename T>
void foo1(T *x, T *y);

// CHECK: #pragma oss task ndrange(1, 1, 1) device(cuda)
// CHECK-NEXT: void foo();
// CHECK: #pragma oss task ndrange(N, *x, *y) device(cuda)
// CHECK-NEXT: template <int N, typename T> void foo1(T *x, T *y);
