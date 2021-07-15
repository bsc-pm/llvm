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

// CHECK: %16 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %p1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32** %p4.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p3.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7), "QUAL.OSS.DEP.IN"(i32** %p1.addr, [6 x i8] c"p1[3]\00", %struct._depend_unpack_t (i32**, i64, i64)* @compute_dep, i32** %p1.addr, i64 %1, i64 %3), "QUAL.OSS.DEP.IN"(i32** %p4.addr, [6 x i8] c"p4[2]\00", %struct._depend_unpack_t.0 (i32**, i64)* @compute_dep.1, i32** %p4.addr, i64 %15), "QUAL.OSS.DEP.IN"(i32*** %p3.addr, [6 x i8] c"p3[5]\00", %struct._depend_unpack_t.1 (i32***, i64)* @compute_dep.2, i32*** %p3.addr, i64 %11), "QUAL.OSS.DEP.IN"(i32*** %p2.addr, [6 x i8] c"p2[3]\00", %struct._depend_unpack_t.2 (i32***, i64, i64)* @compute_dep.3, i32*** %p2.addr, i64 %5, i64 %7) ]
// CHECK-NEXT: %17 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32** %p1.addr), "QUAL.OSS.SHARED"(i32** %p4.addr), "QUAL.OSS.SHARED"(i32*** %p3.addr), "QUAL.OSS.SHARED"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7) ]
// CHECK-NEXT: %18 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32** %p1.addr), "QUAL.OSS.PRIVATE"(i32** %p4.addr), "QUAL.OSS.PRIVATE"(i32*** %p3.addr), "QUAL.OSS.PRIVATE"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7) ]
// CHECK-NEXT: %19 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %p1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32** %p4.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p3.addr), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p2.addr), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %15, i64 %11, i64 %5, i64 %7) ]


// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32** %p1, i64 %0, i64 %1)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %2 = load i32*, i32** %p1, align 8
// CHECK-NEXT:   %3 = mul i64 %1, 4
// CHECK-NEXT:   %4 = mul i64 %1, 4
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %2, i32** %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %4, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %0, i64* %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 7
// CHECK-NEXT:   store i64 1, i64* %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 8
// CHECK-NEXT:   store i64 3, i64* %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 9
// CHECK-NEXT:   store i64 4, i64* %14, align 8
// CHECK-NEXT:   %15 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %15
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32** %p4, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %1 = load i32*, i32** %p4, align 8
// CHECK-NEXT:   %2 = mul i64 %0, 4
// CHECK-NEXT:   %3 = mul i64 %0, 4
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %1, i32** %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %2, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %3, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 1, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 2, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 3, i64* %10, align 8
// CHECK-NEXT:   %11 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %11
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32*** %p3, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %1 = load i32**, i32*** %p3, align 8
// CHECK-NEXT:   %2 = mul i64 %0, 8
// CHECK-NEXT:   %3 = mul i64 %0, 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %1, i32*** %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %2, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %3, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 1, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %10, align 8
// CHECK-NEXT:   %11 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %11
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.2 @compute_dep.3(i32*** %p2, i64 %0, i64 %1)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.2, align 8
// CHECK:   %2 = load i32**, i32*** %p2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %2, i32*** %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 8, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 24, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 32, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.2 %7
// CHECK-NEXT: }

