// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
void vla_senction_dep(int n, int k, int j) {
    int array[n + 1][k + 2][j + 3];
    int array2[n + 1][k + 2][j + 3];
    // DSA duplicated clauses are removed
    #pragma oss task out(array[0 : 5]) in(array, array2) shared(array, array) shared(array)
    {}
}

// CHECK: %23 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.SHARED"(i32* %vla6), "QUAL.OSS.VLA.DIMS"(i32* %vla6, i64 %10, i64 %12, i64 %14), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5, i64 %10, i64 %12, i64 %14), "QUAL.OSS.DEP.IN"(i32* %vla, i64 %17, i64 0, i64 %18, i64 %3, i64 0, i64 %3, i64 %1, i64 0, i64 %1), "QUAL.OSS.DEP.IN"(i32* %vla6, i64 %19, i64 0, i64 %20, i64 %12, i64 0, i64 %12, i64 %10, i64 0, i64 %10), "QUAL.OSS.DEP.OUT"(i32* %vla, i64 %21, i64 0, i64 %22, i64 %3, i64 0, i64 %3, i64 %1, i64 0, i64 6) ]
