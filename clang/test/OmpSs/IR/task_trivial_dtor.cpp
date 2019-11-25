// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
    int x;
    S();
    // ~S();
};

void foo() {
    S s;
    #pragma oss task private(s)
    {}
    #pragma oss task firstprivate(s)
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(%struct.S* %s), "QUAL.OSS.INIT"(%struct.S* %s, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %s), "QUAL.OSS.COPY"(%struct.S* %s, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_) ]

