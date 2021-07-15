// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task final(a)
void bar(int a) {}

void foo() {
    int a;
    #pragma oss task final(a)
    {}
    bar(3);
}

// CHECK: %0 = load i32, i32* %a, align 4
// CHECK-NEXT: %tobool = icmp ne i32 %0, 0
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FINAL"(i1 %tobool) ]

// CHECK: store i32 3, i32* %call_arg, align 4
// CHECK-NEXT: %2 = load i32, i32* %call_arg, align 4
// CHECK-NEXT: %tobool1 = icmp ne i32 %2, 0
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.FINAL"(i1 %tobool1), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"bar:4:9\00") ]
