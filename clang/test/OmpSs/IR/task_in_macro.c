// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#define N(a) _Pragma("oss task in(a)") \
{}

int main() {
    int a;
    N(a);
}

// CHECK: call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a), "QUAL.OSS.DEP.IN"(i32* %a, [5 x i8] c"N(a)\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %a) ]

