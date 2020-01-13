// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

__attribute__((noreturn)) void foo();
int main() {
    #pragma oss task
    {
        foo();
    }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT: call void (...) @foo()
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT: ret i32 0

