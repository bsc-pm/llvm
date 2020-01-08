// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss task priority(foo<int>())
    {}
    #pragma oss task priority(n)
    {}
    #pragma oss task priority(vla[1])
    {}
}

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIORITY"(i32 %call), "QUAL.OSS.CAPTURED"(i32 %call) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIORITY"(i32 %4), "QUAL.OSS.CAPTURED"(i32 %4) ]
// CHECK: %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIORITY"(i32 %6), "QUAL.OSS.CAPTURED"(i32 %6) ]


