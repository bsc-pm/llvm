// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int i, int *q, int *p) {
    #pragma oss taskiter
    while (i < 10) {}
    #pragma oss taskiter
    while (p < q) {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([15 x i8] c"TASKITER.WHILE\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %i.addr), "QUAL.OSS.WHILE.COND"(i1 (i32*)* @compute_while_cond, i32* %i.addr) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([15 x i8] c"TASKITER.WHILE\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %p.addr), "QUAL.OSS.FIRSTPRIVATE"(i32** %q.addr), "QUAL.OSS.WHILE.COND"(i1 (i32**, i32**)* @compute_while_cond.1, i32** %p.addr, i32** %q.addr) ]

// CHECK: define internal i1 @compute_while_cond(i32* %i)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %i, align 4
// CHECK-NEXT:   %cmp = icmp slt i32 %0, 10
// CHECK-NEXT:   ret i1 %cmp
// CHECK-NEXT: }

// CHECK: define internal i1 @compute_while_cond.1(i32** %p, i32** %q)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %p.addr = alloca i32**, align 8
// CHECK-NEXT:   %q.addr = alloca i32**, align 8
// CHECK-NEXT:   store i32** %p, i32*** %p.addr, align 8
// CHECK-NEXT:   store i32** %q, i32*** %q.addr, align 8
// CHECK-NEXT:   %0 = load i32*, i32** %p, align 8
// CHECK-NEXT:   %1 = load i32*, i32** %q, align 8
// CHECK-NEXT:   %cmp = icmp ult i32* %0, %1
// CHECK-NEXT:   ret i1 %cmp
// CHECK-NEXT: }

