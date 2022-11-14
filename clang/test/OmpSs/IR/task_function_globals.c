// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int x;
#pragma oss task in(x)
void foo();

int main() {
  foo();
}


// CHECK: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00")
// CHECK-SAME: "QUAL.OSS.SHARED"(ptr @x, i32 undef)
