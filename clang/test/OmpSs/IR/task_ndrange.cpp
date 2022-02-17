// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) ndrange(1, 1, 1)
void foo();
#pragma oss task device(cuda) ndrange(N, *x, *y)
template <int N, typename T>
void foo1(T *x, T *y);
#pragma oss task device(cuda) ndrange(1, *x, *y)
void foo2(int *x, int *y);

void bar() {
    foo();
    int x, y;
    foo1<1>(&x, &y);
    foo2(&x, &y);
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"),
// CHECK-SAME: "QUAL.OSS.DEVICE.NDRANGE"(i32 1, i32 1, i32 1)
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"),
// CHECK-SAME: "QUAL.OSS.DEVICE.NDRANGE"(i32 1, i32 %2, i32 %4)
// CHECK: %10 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"),
// CHECK-SAME: "QUAL.OSS.DEVICE.NDRANGE"(i32 1, i32 %7, i32 %9)

