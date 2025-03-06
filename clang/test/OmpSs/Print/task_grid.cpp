// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) grid(1, 1, 1, 1, 1, 1)
void foo();

#pragma oss task device(cuda) grid(N, *x, *y, N / 2, *x, *y)
template<int N, typename T>
void foo1(T *x, T *y);

// CHECK: #pragma oss task grid(1, 1, 1, 1, 1, 1) device(cuda)
// CHECK-NEXT: void foo();
// CHECK: #pragma oss task grid(N, *x, *y, N / 2, *x, *y) device(cuda)
// CHECK-NEXT: template <int N, typename T> void foo1(T *x, T *y);
