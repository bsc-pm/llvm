// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo1() {
    int **a;
    int b[5][6];
    int (*c)[5];
    int d[5][6][7];
    int e[5];
    #pragma oss task depend(in: a, *a, a[1], a[1][2])
    {}
    #pragma oss task depend(in: b, *b, b[1], b[1][2])
    {}
    #pragma oss task depend(in: c, *c, c[1], c[1][2])
    {}
    #pragma oss task depend(in: d, d[1][2][3])
    {}
    #pragma oss task depend(in: e, e[1])
    {}
}

// CHECK: %0 = load i32**, i32*** %a, align 8
// CHECK-NEXT: %1 = load i32**, i32*** %a, align 8
// CHECK-NEXT: %2 = load i32**, i32*** %a, align 8
// CHECK-NEXT: %arrayidx = getelementptr inbounds i32*, i32** %2, i64 1
// CHECK-NEXT: %3 = load i32*, i32** %arrayidx, align 8
// CHECK-NEXT: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32*** %a), "QUAL.OSS.DEP.IN"(i32*** %a, i64 8, i64 0, i64 8), "QUAL.OSS.DEP.IN"(i32** %0, i64 8, i64 0, i64 8), "QUAL.OSS.DEP.IN"(i32** %1, i64 8, i64 8, i64 16), "QUAL.OSS.DEP.IN"(i32* %3, i64 4, i64 8, i64 12) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %4)

// CHECK: %arraydecay = getelementptr inbounds [5 x [6 x i32]], [5 x [6 x i32]]* %b, i64 0, i64 0
// CHECK-NEXT: %arraydecay1 = getelementptr inbounds [5 x [6 x i32]], [5 x [6 x i32]]* %b, i64 0, i64 0
// CHECK-NEXT: %arraydecay2 = getelementptr inbounds [5 x [6 x i32]], [5 x [6 x i32]]* %b, i64 0, i64 0
// CHECK-NEXT: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([5 x [6 x i32]]* %b), "QUAL.OSS.DEP.IN"([5 x [6 x i32]]* %b, i64 24, i64 0, i64 24, i64 5, i64 0, i64 5), "QUAL.OSS.DEP.IN"([6 x i32]* %arraydecay, i64 24, i64 0, i64 24), "QUAL.OSS.DEP.IN"([6 x i32]* %arraydecay1, i64 24, i64 0, i64 24, i64 5, i64 1, i64 2), "QUAL.OSS.DEP.IN"([6 x i32]* %arraydecay2, i64 24, i64 8, i64 12, i64 5, i64 1, i64 2) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %5)

// CHECK: %arraydecay3 = getelementptr inbounds [5 x [6 x [7 x i32]]], [5 x [6 x [7 x i32]]]* %d, i64 0, i64 0
// CHECK-NEXT: %10 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([5 x [6 x [7 x i32]]]* %d), "QUAL.OSS.DEP.IN"([5 x [6 x [7 x i32]]]* %d, i64 28, i64 0, i64 28, i64 6, i64 0, i64 6, i64 5, i64 0, i64 5), "QUAL.OSS.DEP.IN"([6 x [7 x i32]]* %arraydecay3, i64 28, i64 12, i64 16, i64 6, i64 2, i64 3, i64 5, i64 1, i64 2) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %10)

// CHECK: %arraydecay4 = getelementptr inbounds [5 x i32], [5 x i32]* %e, i64 0, i64 0
// CHECK-NEXT: %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([5 x i32]* %e), "QUAL.OSS.DEP.IN"([5 x i32]* %e, i64 20, i64 0, i64 20), "QUAL.OSS.DEP.IN"(i32* %arraydecay4, i64 20, i64 4, i64 8) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %11)

void foo2() {
    struct A {
        int x;
    } a;
    struct B {
        int x[10];
    } b;
    struct C {
        int (*x)[10];
    } c;
    struct D {
        int *x;
    } d;
    #pragma oss task depend(in: a.x)
    {}
    #pragma oss task depend(in: b.x, b.x[0])
    {}
    #pragma oss task depend(in: c.x, c.x[0], c.x[0][1])
    {}
    #pragma oss task depend(in: *(d.x))
    {}
}

// CHECK: %x = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
// CHECK-NEXT: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.A* %a), "QUAL.OSS.DEP.IN"(i32* %x, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

// CHECK: %x1 = getelementptr inbounds %struct.B, %struct.B* %b, i32 0, i32 0
// CHECK-NEXT: %x2 = getelementptr inbounds %struct.B, %struct.B* %b, i32 0, i32 0
// CHECK-NEXT: %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %x2, i64 0, i64 0
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.B* %b), "QUAL.OSS.DEP.IN"([10 x i32]* %x1, i64 40, i64 0, i64 40), "QUAL.OSS.DEP.IN"(i32* %arraydecay, i64 40, i64 0, i64 4) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)

// CHECK: %x3 = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 0
// CHECK-NEXT: %x4 = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 0
// CHECK-NEXT: %2 = load [10 x i32]*, [10 x i32]** %x4, align 8
// CHECK-NEXT: %x5 = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 0
// CHECK-NEXT: %3 = load [10 x i32]*, [10 x i32]** %x5, align 8
// CHECK-NEXT: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.C* %c), "QUAL.OSS.DEP.IN"([10 x i32]** %x3, i64 8, i64 0, i64 8), "QUAL.OSS.DEP.IN"([10 x i32]* %2, i64 40, i64 0, i64 40, i64 1, i64 0, i64 1), "QUAL.OSS.DEP.IN"([10 x i32]* %3, i64 40, i64 4, i64 8, i64 1, i64 0, i64 1) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %4)

// CHECK: %x6 = getelementptr inbounds %struct.D, %struct.D* %d, i32 0, i32 0
// CHECK-NEXT: %5 = load i32*, i32** %x6, align 8
// CHECK-NEXT: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.D* %d), "QUAL.OSS.DEP.IN"(i32* %5, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %6)

