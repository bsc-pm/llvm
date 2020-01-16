// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task depend(mutexinoutset: p[1 : 5]) concurrent(p[1 ; 5])
void foo1(int *p) {}

void bar() {
    foo1(0);
}

// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg), "QUAL.OSS.DEP.CONCURRENT"(i32* %0, i64 20, i64 4, i64 24), "QUAL.OSS.DEP.CONCURRENT"(i32* %1, i64 20, i64 4, i64 24) ], !dbg !11
