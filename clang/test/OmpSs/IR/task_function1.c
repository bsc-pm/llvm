// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task in(*a)
void foo(int size, int (*a)[size]) {
}

int main() {
    int n;
    int mat[n][n];
    foo(n - 1, mat);
}

// CHECK: %call_arg = alloca i32, align 4
// CHECK-NEXT: %call_arg1 = alloca i32*, align 8

// CHECK:  %6 = load i32, i32* %n, align 4
// CHECK-NEXT: %sub = sub nsw i32 %6, 1
// CHECK-NEXT: store i32 %sub, i32* %call_arg, align 4
// CHECK: store i32* %vla, i32** %call_arg1, align 8
// CHECK: %12 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg1), "QUAL.OSS.DEP.IN"(i32* %9, i64 %10, i64 0, i64 %11), "QUAL.OSS.CAPTURED"(i64 %8) ]
// CHECK-NEXT: %13 = load i32, i32* %call_arg, align 4
// CHECK-NEXT: %14 = load i32*, i32** %call_arg1, align 8
// CHECK-NEXT: call void @foo(i32 %13, i32* %14)

#pragma oss task out([size/77]p)
void foo1(int size, int *p) {
}

void bar() {
    int n = 10;
    int *v;
    foo1(n/55, v);
    foo1(n/99, v);
}

// checks we do not reuse VLASize between task outline calls

// CHECK: %3 = load i32, i32* %call_arg, align 4
// CHECK-NEXT: %div2 = sdiv i32 %3, 77
// CHECK-NEXT: %4 = zext i32 %div2 to i64
// CHECK-NEXT: %5 = mul i64 %4, 4
// CHECK-NEXT: %6 = mul i64 %4, 4
// CHECK-NEXT: %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg1), "QUAL.OSS.DEP.OUT"(i32* %2, i64 %5, i64 0, i64 %6) ]

// CHECK: %13 = load i32, i32* %call_arg3, align 4
// CHECK-NEXT: %div6 = sdiv i32 %13, 77
// CHECK-NEXT: %14 = zext i32 %div6 to i64
// CHECK-NEXT: %15 = mul i64 %14, 4
// CHECK-NEXT: %16 = mul i64 %14, 4
// CHECK-NEXT: %17 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg3), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg5), "QUAL.OSS.DEP.OUT"(i32* %12, i64 %15, i64 0, i64 %16) ]

