// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
void foo(int sizex,
         int sizey,
         int (*p1)[sizex][sizey],
         int (**p2)[sizex][sizey],
         int *p3[sizex][sizey],
         int p4[sizex][sizey]) {
    int a;
    #pragma oss task in(p1[3], p4[2], p3[5], p2[3])
    #pragma oss task shared(p1, p4, p3, p2)
    #pragma oss task private(p1, p4, p3, p2)
    #pragma oss task firstprivate(p1, p4, p3, p2)
    {
        (*p1)[0][1] = 3;
        (*p2)[0][1][4] = 3;
        p4[2][3] = 4;
        p3[5][3] = 0;
    }
}

// Function params decay to pointer

// CHECK: %26 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %p1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32** %p4.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p3.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7), "QUAL.OSS.DEP.IN"(i32* %16, i64 %17, i64 0, i64 %18, i64 %1, i64 0, i64 %1, i64 1, i64 3, i64 4), "QUAL.OSS.DEP.IN"(i32* %19, i64 %20, i64 0, i64 %21, i64 1, i64 2, i64 3), "QUAL.OSS.DEP.IN"(i32** %22, i64 %23, i64 0, i64 %24, i64 1, i64 5, i64 6), "QUAL.OSS.DEP.IN"(i32** %25, i64 8, i64 24, i64 32) ]
// CHECK: %27 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32** %p1.addr), "QUAL.OSS.SHARED"(i32** %p4.addr), "QUAL.OSS.SHARED"(i32*** %p3.addr), "QUAL.OSS.SHARED"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7) ]
// CHECK: %28 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32** %p1.addr), "QUAL.OSS.PRIVATE"(i32** %p4.addr), "QUAL.OSS.PRIVATE"(i32*** %p3.addr), "QUAL.OSS.PRIVATE"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7) ]
// CHECK: %29 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %p1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32** %p4.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p3.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7) ]

