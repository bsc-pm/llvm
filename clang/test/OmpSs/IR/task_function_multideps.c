// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#define N 4

#pragma oss task out( { p[i][j], i = init1:ub1:step1, j =init2:ub2:step2 } )
void gen(int (*p)[N], int init1, int ub1, int step1, int init2, int ub2, int step2) {}

int main() {
    int M[N][N];
    gen(M, 0, N-1, 1, 0, N-1, 1);
    gen(M, 0, N-1, 1, 0, N-1, 1);
}

// CHECK: %i.call_arg = alloca i32, align 4
// CHECK-NEXT: %j.call_arg = alloca i32, align 4

// CHECK: %i.call_arg15 = alloca i32, align 4
// CHECK-NEXT: %j.call_arg16 = alloca i32, align 4

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32* %i.call_arg), "QUAL.OSS.PRIVATE"(i32* %j.call_arg), "QUAL.OSS.FIRSTPRIVATE"([4 x i32]** %call_arg), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg1), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg2), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg3), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg4), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg5), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg6), "QUAL.OSS.MULTIDEP.RANGE.OUT"(i32* %i.call_arg, i32* %j.call_arg, %struct._depend_unpack_t (i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i64)* @compute_dep, i32* %i.call_arg, i32* %call_arg1, i32* %call_arg2, i32* %call_arg3, i32* %j.call_arg, i32* %call_arg4, i32* %call_arg5, i32* %call_arg6, [4 x i32]** %call_arg, [53 x i8] c"{ p[i][j], i = init1:ub1:step1, j =init2:ub2:step2 }\00", %struct._depend_unpack_t.0 ([4 x i32]**, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*)* @compute_dep.1, [4 x i32]** %call_arg, i32* %i.call_arg, i32* %j.call_arg, i32* %call_arg1, i32* %call_arg4, i32* %call_arg2, i32* %call_arg5, i32* %call_arg3, i32* %call_arg6), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"gen:6:9\00") ]
// CHECK: %8 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32* %i.call_arg15), "QUAL.OSS.PRIVATE"(i32* %j.call_arg16), "QUAL.OSS.FIRSTPRIVATE"([4 x i32]** %call_arg7), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg9), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg10), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg11), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg12), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg13), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg14), "QUAL.OSS.MULTIDEP.RANGE.OUT"(i32* %i.call_arg15, i32* %j.call_arg16, %struct._depend_unpack_t.1 (i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i64)* @compute_dep.2, i32* %i.call_arg15, i32* %call_arg9, i32* %call_arg10, i32* %call_arg11, i32* %j.call_arg16, i32* %call_arg12, i32* %call_arg13, i32* %call_arg14, [4 x i32]** %call_arg7, [53 x i8] c"{ p[i][j], i = init1:ub1:step1, j =init2:ub2:step2 }\00", %struct._depend_unpack_t.2 ([4 x i32]**, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*)* @compute_dep.3, [4 x i32]** %call_arg7, i32* %i.call_arg15, i32* %j.call_arg16, i32* %call_arg9, i32* %call_arg12, i32* %call_arg10, i32* %call_arg13, i32* %call_arg11, i32* %call_arg14), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"gen:6:9\00") ]

