// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// The purpose of this test is to check that, altough we
// need to capture 'this' as firstprivate in the task outline
// we do not need it when building compute_dep function/call

struct S {
  S() {}
  void foo(int a);
};

#pragma oss task in(*a)
void bar(int *a, S *s) {};

void S::foo(int a) {
  bar(&a, this);
}

int main() {
  S s;
}


// CHECK:  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg),
// CHECK-SAME: "QUAL.OSS.FIRSTPRIVATE"(%struct.S** %call_arg2),
// CHECK-SAME: "QUAL.OSS.DEP.IN"(i32** %call_arg, [3 x i8] c"*a\00", %struct._depend_unpack_t (i32**)* @compute_dep, i32** %call_arg),
// CHECK-SAME: "QUAL.OSS.DECL.SOURCE"([9 x i8] c"bar:13:9\00") ]
