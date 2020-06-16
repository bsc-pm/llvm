// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo() {
    #pragma oss task label("T1")
    {}
}

// CHECK: @.str = private unnamed_addr constant [3 x i8] c"T1\00", align 1
// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.LABEL"([3 x i8]* @.str) ]
