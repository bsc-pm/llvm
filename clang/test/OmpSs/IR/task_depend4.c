// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo1(int a) {
    // concurrent
    #pragma oss task depend(mutexinoutset: a)
    {}
    #pragma oss task concurrent(a)
    {}
    // commutative
    #pragma oss task depend(inoutset: a)
    {}
    #pragma oss task commutative(a)
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a.addr), "QUAL.OSS.DEP.CONCURRENT"(i32* %a.addr, i64 4, i64 0, i64 4) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a.addr), "QUAL.OSS.DEP.CONCURRENT"(i32* %a.addr, i64 4, i64 0, i64 4) ]
// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a.addr), "QUAL.OSS.DEP.COMMUTATIVE"(i32* %a.addr, i64 4, i64 0, i64 4) ]
// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %a.addr), "QUAL.OSS.DEP.COMMUTATIVE"(i32* %a.addr, i64 4, i64 0, i64 4) ]
