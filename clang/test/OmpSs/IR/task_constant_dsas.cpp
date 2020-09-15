// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics


// This test checks that we emit DSA for constants. The use case of them is
// when the code gets the constant address.

constexpr int a = 0;
constexpr int *p = 0;
constexpr int const &ra = a;
constexpr int *const &rp = p;
constexpr int N = 4;
int asdf();
const int b = asdf();
enum { M = 5 };

void foo() {
    #pragma oss task firstprivate(N, ra, a, rp, p, b)
    {
        const int *x = &N;
        const int *y = &ra;
        const int *z = &a;
        const int *const *i = &rp;
        const int *const *j = &p;
        const int *k = &b;
        const int l = M;
    }
}

void bar() {
    constexpr int a = 0;
    constexpr int *p = 0;
    constexpr int N = 4;
    const int b = asdf();
    #pragma oss task firstprivate(N, ra, a, rp, p, b)
    {
        const int *x = &N;
        const int *z = &a;
        const int *const*j = &p;
        const int *k = &b;
        const int l = M;
    }
}

// CHECK: %0 = load i32*, i32** @ra, align 8
// CHECK-NEXT: %1 = load i32**, i32*** @rp, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* @_ZL1N), "QUAL.OSS.FIRSTPRIVATE"(i32* %0), "QUAL.OSS.FIRSTPRIVATE"(i32* @_ZL1a), "QUAL.OSS.FIRSTPRIVATE"(i32** %1), "QUAL.OSS.FIRSTPRIVATE"(i32** @_ZL1p), "QUAL.OSS.FIRSTPRIVATE"(i32* @_ZL1b) ]
// CHECK-NEXT: %x = alloca i32*, align 8
// CHECK-NEXT: %y = alloca i32*, align 8
// CHECK-NEXT: %z = alloca i32*, align 8
// CHECK-NEXT: %i = alloca i32**, align 8
// CHECK-NEXT: %j = alloca i32**, align 8
// CHECK-NEXT: %k = alloca i32*, align 8
// CHECK-NEXT: %l = alloca i32, align 4
// CHECK-NEXT: store i32* @_ZL1N, i32** %x, align 8
// CHECK-NEXT: store i32* %0, i32** %y, align 8
// CHECK-NEXT: store i32* @_ZL1a, i32** %z, align 8
// CHECK-NEXT: store i32** %1, i32*** %i, align 8
// CHECK-NEXT: store i32** @_ZL1p, i32*** %j, align 8
// CHECK-NEXT: store i32* @_ZL1b, i32** %k, align 8
// CHECK-NEXT: store i32 5, i32* %l, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)

// CHECK: %0 = load i32*, i32** @ra, align 8
// CHECK-NEXT: %1 = load i32**, i32*** @rp, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %N), "QUAL.OSS.FIRSTPRIVATE"(i32* %0), "QUAL.OSS.FIRSTPRIVATE"(i32* %a), "QUAL.OSS.FIRSTPRIVATE"(i32** %1), "QUAL.OSS.FIRSTPRIVATE"(i32** %p), "QUAL.OSS.FIRSTPRIVATE"(i32* %b) ]
// CHECK-NEXT: %x = alloca i32*, align 8
// CHECK-NEXT: %z = alloca i32*, align 8
// CHECK-NEXT: %j = alloca i32**, align 8
// CHECK-NEXT: %k = alloca i32*, align 8
// CHECK-NEXT: %l = alloca i32, align 4
// CHECK-NEXT: store i32* %N, i32** %x, align 8
// CHECK-NEXT: store i32* %a, i32** %z, align 8
// CHECK-NEXT: store i32** %p, i32*** %j, align 8
// CHECK-NEXT: store i32* %b, i32** %k, align 8
// CHECK-NEXT: store i32 5, i32* %l, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)
