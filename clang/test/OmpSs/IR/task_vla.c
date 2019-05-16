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

// CHECK  store i64 %1, i64* %__vla_expr0, align 8
// CHECK-NEXT  store i64 %3, i64* %__vla_expr1, align 8
// CHECK-NEXT  %7 = load i32, i32* %i, align 4
// CHECK-NEXT  %add1 = add nsw i32 %7, 1
// CHECK-NEXT  %8 = sext i32 %add1 to i64
// CHECK-NEXT  %9 = add i64 %8, 1
// CHECK-NEXT  %10 = load i32, i32* %i, align 4
// CHECK-NEXT  %11 = sext i32 %10 to i64
// CHECK-NEXT  %12 = add i64 %11, 1
// CHECK-NEXT  %13 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED.VLA"([7 x i32]* %vla, i64 %1, i64 7, i64 %3, i64 7), "QUAL.OSS.FIRSTPRIVATE"(i32* %i), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, i64 28, i64 0, i64 28, i64 %3, i64 0, i64 %3, i64 7, i64 %8, i64 %9, i64 1, i64 %11, i64 %12), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, i64 28, i64 0, i64 28, i64 %3, i64 0, i64 %3, i64 7, i64 0, i64 7) ]
// CHECK-NEXT  call void @llvm.directive.region.exit(token %13)
// CHECK-NEXT  %14 = load i32, i32* %n.addr, align 4
// CHECK-NEXT  %add2 = add nsw i32 %14, 1
// CHECK-NEXT  %15 = zext i32 %add2 to i64
// CHECK-NEXT  %16 = load i32, i32* %n.addr, align 4
// CHECK-NEXT  %add3 = add nsw i32 %16, 1
// CHECK-NEXT  %17 = zext i32 %add3 to i64
// CHECK-NEXT  %18 = load [7 x i32]*, [7 x i32]** %p_array, align 8
// CHECK-NEXT  %19 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([7 x i32]** %p_array), "QUAL.OSS.DEP.IN"([7 x i32]* %18, i64 28, i64 0, i64 28, i64 %17, i64 0, i64 %17, i64 7, i64 0, i64 7, i64 %15, i64 0, i64 %15, i64 1, i64 0, i64 1) ]
// CHECK-NEXT  call void @llvm.directive.region.exit(token %19)

int p;
void foo1(int x) {
    int y;
    #pragma oss task
    {
        int vla[x][7][y][p];
    }
    #pragma oss task
    {
        int z = x + y + p;
    }
}

// CHECK:  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @p), "QUAL.OSS.FIRSTPRIVATE"(i32* %x.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %y) ]
// CHECK:  %12 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @p), "QUAL.OSS.FIRSTPRIVATE"(i32* %x.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %y) ]

void foo2(int x) {
    int y;
    int array[x + 1][y + 1];
    #pragma oss task
    { array[0][0] = 1; }
}

// CHECK: store i64 %1, i64* %__vla_expr0, align 8
// CHECK-NEXT: store i64 %3, i64* %__vla_expr1, align 8
// CHECK-NEXT: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE.VLA"(i32* %vla, i64 %1, i64 %3) ]
// CHECK-NEXT: %7 = mul nsw i64 0, %3

