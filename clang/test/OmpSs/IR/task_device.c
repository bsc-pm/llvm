// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(smp)
void foo();
#pragma oss task device(cuda)
void foo1();
#pragma oss task device(opencl)
void foo2();
#pragma oss task device(fpga)
void foo3();

void bar() {
    foo();
    foo1();
    foo2();
    foo3();
    #pragma oss task device(cuda)
    {}
    #pragma oss task device(opencl)
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DEVICE"(i32 0)
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DEVICE"(i32 1)
// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DEVICE"(i32 4)
// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DEVICE"(i32 5)
// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DEVICE"(i32 1)
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DEVICE"(i32 4)
