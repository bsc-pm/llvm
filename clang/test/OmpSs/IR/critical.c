// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main() {
  #pragma oss critical(asdf)
  {}
  #pragma oss critical
  {}
}

// CHECK: %0 = call i1 @llvm.directive.marker() [ "DIR.OSS"([15 x i8] c"CRITICAL.START\00", [21 x i8] c"nanos6_critical_asdf\00") ]
// CHECK-NEXT: %1 = call i1 @llvm.directive.marker() [ "DIR.OSS"([13 x i8] c"CRITICAL.END\00", [21 x i8] c"nanos6_critical_asdf\00") ]
// CHECK: %2 = call i1 @llvm.directive.marker() [ "DIR.OSS"([15 x i8] c"CRITICAL.START\00", [24 x i8] c"nanos6_critical_default\00") ]
// CHECK-NEXT: %3 = call i1 @llvm.directive.marker() [ "DIR.OSS"([13 x i8] c"CRITICAL.END\00", [24 x i8] c"nanos6_critical_default\00") ]

