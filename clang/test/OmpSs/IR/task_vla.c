// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int n) {
    int i;
    int array[n][7][n+1][7];
    #pragma oss task depend(in: array[i][i+1], *array)
    {}
    int (*p_array)[n+1][7][n+1][7];
    #pragma oss task depend(in: p_array[0])
    {}
}

// CHECK: %7 = load i32, i32* %n.addr, align 4
// CHECK-NEXT: %8 = sext i32 %7 to i64
// CHECK-NEXT: %9 = load i32, i32* %n.addr, align 4
// CHECK-NEXT: %add1 = add nsw i32 %9, 1
// CHECK-NEXT: %10 = sext i32 %add1 to i64
// CHECK-NEXT: %11 = load i32, i32* %i, align 4
// CHECK-NEXT: %add2 = add nsw i32 %11, 1
// CHECK-NEXT: %12 = sext i32 %add2 to i64
// CHECK-NEXT: %13 = add i64 %12, 1
// CHECK-NEXT: %14 = load i32, i32* %i, align 4
// CHECK-NEXT: %15 = sext i32 %14 to i64
// CHECK-NEXT: %16 = add i64 %15, 1
// CHECK-NEXT: %17 = load i32, i32* %n.addr, align 4
// CHECK-NEXT: %add3 = add nsw i32 %17, 1
// CHECK-NEXT: %18 = sext i32 %add3 to i64
// CHECK-NEXT: %19 = load i32, i32* %n.addr, align 4
// CHECK-NEXT: %add4 = add nsw i32 %19, 1
// CHECK-NEXT: %20 = sext i32 %add4 to i64
// CHECK-NEXT: %21 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED.VLA"([7 x i32]* %vla, i64 %8, i64 7, i64 %10, i64 7), "QUAL.OSS.FIRSTPRIVATE"(i32* %i), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, i64 28, i64 0, i64 28, i64 %18, i64 0, i64 %18, i64 7, i64 %12, i64 %13, i64 1, i64 %15, i64 %16), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, i64 28, i64 0, i64 28, i64 %20, i64 0, i64 %20, i64 7, i64 0, i64 7) ]

// CHECK: %26 = load [7 x i32]*, [7 x i32]** %p_array, align 8
// CHECK-NEXT: %27 = load i32, i32* %n.addr, align 4
// CHECK-NEXT: %add7 = add nsw i32 %27, 1
// CHECK-NEXT: %28 = sext i32 %add7 to i64
// CHECK-NEXT: %29 = load i32, i32* %n.addr, align 4
// CHECK-NEXT: %add8 = add nsw i32 %29, 1
// CHECK-NEXT: %30 = sext i32 %add8 to i64
// CHECK-NEXT: %31 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([7 x i32]** %p_array), "QUAL.OSS.DEP.IN"([7 x i32]* %26, i64 28, i64 0, i64 28, i64 %30, i64 0, i64 %30, i64 7, i64 0, i64 7, i64 %28, i64 0, i64 %28, i64 1, i64 0, i64 1) ]
