// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// References are passed as shared so task outline can modify
// the original variable

#pragma oss task
void foo(int &x) {}

int main() {
    int x;
    foo(x);
}

// CHECK: %1 = load i32*, i32** %call_arg, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %1), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"foo:7:9\00") ]
