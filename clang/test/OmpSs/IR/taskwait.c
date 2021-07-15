// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main(void) {
  int a;
  #pragma oss taskwait
  #pragma oss taskwait in(a)
}

// CHECK: call i1 @llvm.directive.marker() [ "DIR.OSS"([9 x i8] c"TASKWAIT\00") ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.IF"(i1 false), "QUAL.OSS.SHARED"(i32* %a), "QUAL.OSS.DEP.IN"(i32* %a, [2 x i8] c"a\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %a) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)
