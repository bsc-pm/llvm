// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
void matmul(const long TS, void *C) {
    #pragma oss task weakinout(C)
    {
        __attribute__((unused))
        double (* restrict C_block)[TS] = C;
    }
}

// Check we firstprivatize TS

// CHECK:  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i8** %C.addr), "QUAL.OSS.FIRSTPRIVATE"(i64* %TS.addr), "QUAL.OSS.DEP.WEAKINOUT"(i8** %C.addr, i64 8, i64 0, i64 8) ]
