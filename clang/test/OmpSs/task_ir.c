// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task shared(i, pi, ai)
  {}
}

// CHECK: call token @llvm.ompss.region.entry() [ "kind"([5 x i8] c"task\00"), "shared"(i32* %i), "shared"(i32** %pi), "shared"([5 x i32]* %ai) ]
