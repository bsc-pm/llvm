// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss assert("a == b")
// CHECK-LABEL: @main(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 0, !dbg [[DBG12:![0-9]+]]
//
int main() {}

// CHECK: !{i32 1, !"OmpSs-2 Metadata", ![[MD:.*]]}
// CHECK: ![[MD]] = !{![[MD1:.*]]}
// CHECK: ![[MD1]] = !{!"assert", !"a == b"}
