// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) ndrange(1, 1, 1)
void foo1();

#pragma oss task device(cuda)
void foo2() {

}

void bar() {
    foo1();
    foo2();
}

// Only impure task emit a call
// CHECK: %1 = call token @llvm.directive.region.entry()
// CHECK: call void @foo2()
