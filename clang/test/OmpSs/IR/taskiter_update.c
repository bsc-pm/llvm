// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
void foo(int n) {
  #pragma oss taskiter update
  for (int i = 0; i < 10; ++i) { }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKITER.FOR\00")
// CHECK-SAME: "QUAL.OSS.LOOP.UPDATE"(i1 true)

