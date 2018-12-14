// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main(void) {
  #pragma oss taskwait
}

// CHECK: call i1 @llvm.ompss.marker() [ "kind"([9 x i8] c"taskwait\00") ]
