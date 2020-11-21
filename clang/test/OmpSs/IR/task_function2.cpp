// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task
template<typename T>
void foo(T *a) {}

int main() {
    foo<int>(nullptr);
}

// CHECK: store i32* null, i32** %call_arg, align 8
// CHECK-NEXT: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"foo:4:9\00") ]
