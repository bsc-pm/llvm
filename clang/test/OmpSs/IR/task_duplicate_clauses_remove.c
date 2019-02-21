// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo() {
    int a;
    #pragma oss task shared(a) shared(a)
    {}
    #pragma oss task private(a) private(a)
    {}
    #pragma oss task firstprivate(a) firstprivate(a)
    {}
}
// CHECK: call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a) ]
// CHECK: call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32* %a) ]
// CHECK: call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %a) ]

