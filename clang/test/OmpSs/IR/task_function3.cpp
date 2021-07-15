// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
    #pragma oss task
    void foo1(int x) {}
};

void bar() {
    S s;
    s.foo1(1 + 2);
}

// CHECK: store i32 3, i32* %call_arg, align 4
// CHECK-NEXT: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %s), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.DECL.SOURCE"([10 x i8] c"foo1:5:13\00") ]
