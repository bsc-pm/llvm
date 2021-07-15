// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int array[10][20];
void foo1(int **p, int n) {
    #pragma oss task depend(in : [n + 1]p, [n + 2]array)
    {}
    #pragma oss task depend(in : ([3]p)[2], ([4]array)[3])
    {}
    #pragma oss task depend(in : ([3]p)[2 : n], ([4]array)[3 : n])
    {}
    #pragma oss task depend(in : [3]p[2], [4]array[3])
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.DEP.IN"(i32*** %p.addr, [9 x i8] c"[n + 1]p\00", %struct._depend_unpack_t (i32***, i32*)* @compute_dep, i32*** %p.addr, i32* %n.addr), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, [13 x i8] c"[n + 2]array\00", %struct._depend_unpack_t.0 (i32*, [10 x [20 x i32]]*)* @compute_dep.1, i32* %n.addr, [10 x [20 x i32]]* @array) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.DEP.IN"(i32*** %p.addr, [10 x i8] c"([3]p)[2]\00", %struct._depend_unpack_t.1 (i32***)* @compute_dep.2, i32*** %p.addr), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, [14 x i8] c"([4]array)[3]\00", %struct._depend_unpack_t.2 ([10 x [20 x i32]]*)* @compute_dep.3, [10 x [20 x i32]]* @array) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.DEP.IN"(i32*** %p.addr, [14 x i8] c"([3]p)[2 : n]\00", %struct._depend_unpack_t.3 (i32***, i32*)* @compute_dep.4, i32*** %p.addr, i32* %n.addr), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, [18 x i8] c"([4]array)[3 : n]\00", %struct._depend_unpack_t.4 (i32*, [10 x [20 x i32]]*)* @compute_dep.5, i32* %n.addr, [10 x [20 x i32]]* @array) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.DEP.IN"(i32*** %p.addr, [8 x i8] c"[3]p[2]\00", %struct._depend_unpack_t.5 (i32***)* @compute_dep.6, i32*** %p.addr), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, [12 x i8] c"[4]array[3]\00", %struct._depend_unpack_t.6 ([10 x [20 x i32]]*)* @compute_dep.7, [10 x [20 x i32]]* @array) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %3)


// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32*** %p, i32* %n)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %0 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %1 = load i32, i32* %n, align 4
// CHECK-NEXT:   %add = add nsw i32 %1, 1
// CHECK-NEXT:   %2 = zext i32 %add to i64
// CHECK-NEXT:   %3 = mul i64 %2, 8
// CHECK-NEXT:   %4 = mul i64 %2, 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %0, i32*** %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %4, i64* %8, align 8
// CHECK-NEXT:   %9 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %9
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %n, [10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 0
// CHECK-NEXT:   %0 = load i32, i32* %n, align 4
// CHECK-NEXT:   %add = add nsw i32 %0, 2
// CHECK-NEXT:   %1 = zext i32 %add to i64
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store [20 x i32]* %arraydecay, [20 x i32]** %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %1, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %1, i64* %8, align 8
// CHECK-NEXT:   %9 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %9
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32*** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %0 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %1 = bitcast i32** %0 to [3 x i32*]*
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [3 x i32*], [3 x i32*]* %1, i64 0, i64 0
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %arraydecay, i32*** %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 24, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 16, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 24, i64* %5, align 8
// CHECK-NEXT:   %6 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %6
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.2 @compute_dep.3([10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.2, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 0
// CHECK-NEXT:   %0 = bitcast [20 x i32]* %arraydecay to [4 x [20 x i32]]*
// CHECK-NEXT:   %arraydecay1 = getelementptr inbounds [4 x [20 x i32]], [4 x [20 x i32]]* %0, i64 0, i64 0
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 0
// CHECK-NEXT:   store [20 x i32]* %arraydecay1, [20 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 4, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 4, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.2 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.4(i32*** %p, i32* %n)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.3, align 8
// CHECK:   %0 = load i32, i32* %n, align 4
// CHECK-NEXT:   %1 = sext i32 %0 to i64
// CHECK-NEXT:   %2 = add i64 2, %1
// CHECK-NEXT:   %3 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %4 = bitcast i32** %3 to [3 x i32*]*
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [3 x i32*], [3 x i32*]* %4, i64 0, i64 0
// CHECK-NEXT:   %5 = mul i64 %2, 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %arraydecay, i32*** %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 24, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 16, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %5, i64* %9, align 8
// CHECK-NEXT:   %10 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.3 %10
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.4 @compute_dep.5(i32* %n, [10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.4, align 8
// CHECK:   %0 = load i32, i32* %n, align 4
// CHECK-NEXT:   %1 = sext i32 %0 to i64
// CHECK-NEXT:   %2 = add i64 3, %1
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 0
// CHECK-NEXT:   %3 = bitcast [20 x i32]* %arraydecay to [4 x [20 x i32]]*
// CHECK-NEXT:   %arraydecay1 = getelementptr inbounds [4 x [20 x i32]], [4 x [20 x i32]]* %3, i64 0, i64 0
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 0
// CHECK-NEXT:   store [20 x i32]* %arraydecay1, [20 x i32]** %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 4, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 3, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %2, i64* %10, align 8
// CHECK-NEXT:   %11 = load %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.4 %11
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.5 @compute_dep.6(i32*** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.5, align 8
// CHECK:   %0 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %arrayidx = getelementptr inbounds i32*, i32** %0, i64 2
// CHECK-NEXT:   %1 = load i32*, i32** %arrayidx, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %1, i32** %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 12, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 12, i64* %5, align 8
// CHECK-NEXT:   %6 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.5 %6
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.6 @compute_dep.7([10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.6, align 8
// CHECK:   %arrayidx = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 3
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [20 x i32], [20 x i32]* %arrayidx, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 16, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 16, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.6 %4
// CHECK-NEXT: }


void foo2() {
    int n, m;
    int vla[10];
    #pragma oss task inout(([n]vla)[m])
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x i32]* %vla), "QUAL.OSS.FIRSTPRIVATE"(i32* %n), "QUAL.OSS.FIRSTPRIVATE"(i32* %m), "QUAL.OSS.DEP.INOUT"([10 x i32]* %vla, [12 x i8] c"([n]vla)[m]\00", %struct._depend_unpack_t.7 ([10 x i32]*, i32*, i32*)* @compute_dep.8, [10 x i32]* %vla, i32* %n, i32* %m) ]

// CHECK: define internal %struct._depend_unpack_t.7 @compute_dep.8([10 x i32]* %vla, i32* %n, i32* %m)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.7, align 8
// CHECK:   %0 = load i32, i32* %m, align 4
// CHECK-NEXT:   %1 = sext i32 %0 to i64
// CHECK-NEXT:   %2 = add i64 %1, 1
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %vla, i64 0, i64 0
// CHECK-NEXT:   %3 = load i32, i32* %n, align 4
// CHECK-NEXT:   %4 = zext i32 %3 to i64
// CHECK-NEXT:   %5 = mul i64 %4, 4
// CHECK-NEXT:   %6 = mul i64 %1, 4
// CHECK-NEXT:   %7 = mul i64 %2, 4
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %5, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 %6, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %7, i64* %11, align 8
// CHECK-NEXT:   %12 = load %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.7 %12
// CHECK-NEXT: }


#pragma oss task in(pvla[sizeof(*pvla)])
void foo3(int n, int (*pvla)[n - 1]) {
}

void bar() {
    int mat[10][10];
    foo3(10, mat);
}

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg1), "QUAL.OSS.DEP.IN"(i32** %call_arg1, [20 x i8] c"pvla[sizeof(*pvla)]\00", %struct._depend_unpack_t.8 (i32**, i64)* @compute_dep.9, i32** %call_arg1, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1), "QUAL.OSS.DECL.SOURCE"([11 x i8] c"foo3:233:9\00") ]

// CHECK: define internal %struct._depend_unpack_t.8 @compute_dep.9(i32** %pvla, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.8, align 8
// CHECK:   %1 = load i32*, i32** %pvla, align 8
// CHECK-NEXT:   %2 = mul nuw i64 4, %0
// CHECK-NEXT:   %3 = add i64 %2, 1
// CHECK-NEXT:   %4 = load i32*, i32** %pvla, align 8
// CHECK-NEXT:   %5 = mul i64 %0, 4
// CHECK-NEXT:   %6 = mul i64 %0, 4
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %4, i32** %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %5, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %6, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 1, i64* %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 %2, i64* %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %3, i64* %13, align 8
// CHECK-NEXT:   %14 = load %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.8 %14
// CHECK-NEXT: }

void foo4(int x, int y, int z) {
    int a, b, c;
    int vla[x + 1][y + 2][z + 3];
    #pragma oss task inout(([a](vla[2]))[b])
    {}
    #pragma oss task inout([c]([a][b]vla))
    {}
    #pragma oss task inout([c]([a][b]vla[sizeof(vla)]))
    {}
}

// CHECK: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.FIRSTPRIVATE"(i32* %a), "QUAL.OSS.FIRSTPRIVATE"(i32* %b), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5), "QUAL.OSS.DEP.INOUT"(i32* %vla, [17 x i8] c"([a](vla[2]))[b]\00", %struct._depend_unpack_t.9 (i32*, i32*, i32*, i64, i64, i64)* @compute_dep.10, i32* %vla, i32* %a, i32* %b, i64 %1, i64 %3, i64 %5) ]
// CHECK: %10 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.FIRSTPRIVATE"(i32* %a), "QUAL.OSS.FIRSTPRIVATE"(i32* %b), "QUAL.OSS.FIRSTPRIVATE"(i32* %c), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5), "QUAL.OSS.DEP.INOUT"(i32* %vla, [15 x i8] c"[c]([a][b]vla)\00", %struct._depend_unpack_t.10 (i32*, i32*, i32*, i32*, i64, i64, i64)* @compute_dep.11, i32* %vla, i32* %a, i32* %b, i32* %c, i64 %1, i64 %3, i64 %5) ]
// CHECK: %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.FIRSTPRIVATE"(i32* %a), "QUAL.OSS.FIRSTPRIVATE"(i32* %b), "QUAL.OSS.FIRSTPRIVATE"(i32* %c), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5), "QUAL.OSS.DEP.INOUT"(i32* %vla, [28 x i8] c"[c]([a][b]vla[sizeof(vla)])\00", %struct._depend_unpack_t.11 (i32*, i32*, i32*, i32*, i64, i64, i64)* @compute_dep.12, i32* %vla, i32* %a, i32* %b, i32* %c, i64 %1, i64 %3, i64 %5) ]

// CHECK: define internal %struct._depend_unpack_t.9 @compute_dep.10(i32* %vla, i32* %a, i32* %b, i64 %0, i64 %1, i64 %2)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.9, align 8
// CHECK:   %3 = load i32, i32* %b, align 4
// CHECK-NEXT:   %4 = sext i32 %3 to i64
// CHECK-NEXT:   %5 = add i64 %4, 1
// CHECK-NEXT:   %6 = mul nuw i64 %1, %2
// CHECK-NEXT:   %7 = mul nsw i64 2, %6
// CHECK-NEXT:   %arrayidx = getelementptr inbounds i32, i32* %vla, i64 %7
// CHECK-NEXT:   %8 = load i32, i32* %a, align 4
// CHECK-NEXT:   %9 = zext i32 %8 to i64
// CHECK-NEXT:   %10 = mul i64 %2, 4
// CHECK-NEXT:   %11 = mul i64 %2, 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arrayidx, i32** %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %10, i64* %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %14, align 8
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %11, i64* %15, align 8
// CHECK-NEXT:   %16 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %9, i64* %16, align 8
// CHECK-NEXT:   %17 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 %4, i64* %17, align 8
// CHECK-NEXT:   %18 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %5, i64* %18, align 8
// CHECK-NEXT:   %19 = load %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.9 %19
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.10 @compute_dep.11(i32* %vla, i32* %a, i32* %b, i32* %c, i64 %0, i64 %1, i64 %2)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.10, align 8
// CHECK:   %3 = load i32, i32* %c, align 4
// CHECK-NEXT:   %4 = zext i32 %3 to i64
// CHECK-NEXT:   %5 = load i32, i32* %b, align 4
// CHECK-NEXT:   %6 = zext i32 %5 to i64
// CHECK-NEXT:   %7 = mul i64 %2, 4
// CHECK-NEXT:   %8 = mul i64 %2, 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %vla, i32** %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %7, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %8, i64* %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %1, i64* %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %14, align 8
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %1, i64* %15, align 8
// CHECK-NEXT:   %16 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 7
// CHECK-NEXT:   store i64 %6, i64* %16, align 8
// CHECK-NEXT:   %17 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 8
// CHECK-NEXT:   store i64 0, i64* %17, align 8
// CHECK-NEXT:   %18 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 9
// CHECK-NEXT:   store i64 %6, i64* %18, align 8
// CHECK-NEXT:   %19 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 10
// CHECK-NEXT:   store i64 %4, i64* %19, align 8
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 11
// CHECK-NEXT:   store i64 0, i64* %20, align 8
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 12
// CHECK-NEXT:   store i64 %4, i64* %21, align 8
// CHECK-NEXT:   %22 = load %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.10 %22
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.11 @compute_dep.12(i32* %vla, i32* %a, i32* %b, i32* %c, i64 %0, i64 %1, i64 %2)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.11, align 8
// CHECK:   %3 = mul nuw i64 %0, %1
// CHECK-NEXT:   %4 = mul nuw i64 %3, %2
// CHECK-NEXT:   %5 = mul nuw i64 4, %4
// CHECK-NEXT:   %6 = mul nuw i64 %1, %2
// CHECK-NEXT:   %7 = mul nsw i64 %5, %6
// CHECK-NEXT:   %arrayidx = getelementptr inbounds i32, i32* %vla, i64 %7
// CHECK-NEXT:   %8 = load i32, i32* %c, align 4
// CHECK-NEXT:   %9 = zext i32 %8 to i64
// CHECK-NEXT:   %10 = load i32, i32* %b, align 4
// CHECK-NEXT:   %11 = zext i32 %10 to i64
// CHECK-NEXT:   %12 = mul i64 %2, 4
// CHECK-NEXT:   %13 = mul i64 %2, 4
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arrayidx, i32** %14, align 8
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %12, i64* %15, align 8
// CHECK-NEXT:   %16 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %16, align 8
// CHECK-NEXT:   %17 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %13, i64* %17, align 8
// CHECK-NEXT:   %18 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %11, i64* %18, align 8
// CHECK-NEXT:   %19 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %19, align 8
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %11, i64* %20, align 8
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 7
// CHECK-NEXT:   store i64 %9, i64* %21, align 8
// CHECK-NEXT:   %22 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 8
// CHECK-NEXT:   store i64 0, i64* %22, align 8
// CHECK-NEXT:   %23 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 9
// CHECK-NEXT:   store i64 %9, i64* %23, align 8
// CHECK-NEXT:   %24 = load %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.11 %24
// CHECK-NEXT: }
