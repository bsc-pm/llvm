// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo() {
    int a;
    #pragma oss task if(a)
    {}
}

// CHECK: %a = alloca i32, align 4
// CHECK-NEXT: %0 = load i32, i32* %a, align 4
// CHECK-NEXT: %tobool = icmp ne i32 %0, 0
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.IF"(i1 %tobool) ]
