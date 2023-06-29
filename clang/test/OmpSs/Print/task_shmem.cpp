// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(1)
void foo();

#pragma oss task device(cuda) ndrange(N, *x, *y) shmem(shm)
template<int N, typename T, int shm>
void foo1(T *x, T *y);

#pragma oss task device(cuda) ndrange(N, *x, *y) shmem(shm)
template<int N, typename T>
void foo1(T *x, T *y, T shm);

#pragma oss task device(cuda) ndrange(N, *x, *y) shmem(shm)
template<int N, typename T>
void foo1(T *x, T *y, int shm);

// CHECK: #pragma oss task ndrange(1, 1, 1) shmem(1) device(cuda)
// CHECK: void foo();
// CHECK: #pragma oss task ndrange(N, *x, *y) shmem(shm) device(cuda)
// CHECK: template <int N, typename T, int shm> void foo1(T *x, T *y);
// CHECK: #pragma oss task ndrange(N, *x, *y) shmem(shm) device(cuda)
// CHECK: template <int N, typename T> void foo1(T *x, T *y, T shm);
// CHECK: #pragma oss task ndrange(N, *x, *y) shmem(shm) device(cuda)
// CHECK: template <int N, typename T> void foo1(T *x, T *y, int shm);

