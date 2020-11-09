// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

typedef int* Pointer;

void foo(int x) {
    Pointer aFix[10][10];
    int aVLA[x][7];
    #pragma oss task depend(in : aFix[:], aFix[1:], aFix[: 2], aFix[3 : 4])
    {}
    #pragma oss task depend(in : aFix[5][:], aFix[5][1:], aFix[5][: 2], aFix[5][3 : 4])
    {}
    #pragma oss task depend(in : aVLA[:], aVLA[1:], aVLA[: 2], aVLA[3 : 4])
    {}
    #pragma oss task depend(in : aVLA[5][:], aVLA[5][1:], aVLA[5][: 2], aVLA[5][3 : 4])
    {}
}

// CHECK: %aFix = alloca [10 x [10 x i32*]], align
// CHECK-NEXT: %saved_stack = alloca i8*, align 8
// CHECK-NEXT: %__vla_expr0 = alloca i64, align 8
// CHECK-NEXT: store i32 %x, i32* %x.addr, align 4
// CHECK-NEXT: %0 = load i32, i32* %x.addr, align 4
// CHECK-NEXT: %1 = zext i32 %0 to i64
// CHECK-NEXT: %2 = call i8* @llvm.stacksave()
// CHECK-NEXT: store i8* %2, i8** %saved_stack, align 8
// CHECK-NEXT: %vla = alloca [7 x i32], i64 %1, align {{.*}}
// CHECK-NEXT: store i64 %1, i64* %__vla_expr0, align 8
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [8 x i8] c"aFix[:]\00", %struct._depend_unpack_t ([10 x [10 x i32*]]*)* @compute_dep, [10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [9 x i8] c"aFix[1:]\00", %struct._depend_unpack_t.0 ([10 x [10 x i32*]]*)* @compute_dep.1, [10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [10 x i8] c"aFix[: 2]\00", %struct._depend_unpack_t.1 ([10 x [10 x i32*]]*)* @compute_dep.2, [10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [12 x i8] c"aFix[3 : 4]\00", %struct._depend_unpack_t.2 ([10 x [10 x i32*]]*)* @compute_dep.3, [10 x [10 x i32*]]* %aFix) ]
// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [11 x i8] c"aFix[5][:]\00", %struct._depend_unpack_t.3 ([10 x [10 x i32*]]*)* @compute_dep.4, [10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [12 x i8] c"aFix[5][1:]\00", %struct._depend_unpack_t.4 ([10 x [10 x i32*]]*)* @compute_dep.5, [10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [13 x i8] c"aFix[5][: 2]\00", %struct._depend_unpack_t.5 ([10 x [10 x i32*]]*)* @compute_dep.6, [10 x [10 x i32*]]* %aFix), "QUAL.OSS.DEP.IN"([10 x [10 x i32*]]* %aFix, [15 x i8] c"aFix[5][3 : 4]\00", %struct._depend_unpack_t.6 ([10 x [10 x i32*]]*)* @compute_dep.7, [10 x [10 x i32*]]* %aFix) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([7 x i32]* %vla), "QUAL.OSS.VLA.DIMS"([7 x i32]* %vla, i64 %1, i64 7), "QUAL.OSS.CAPTURED"(i64 %1, i64 7), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [8 x i8] c"aVLA[:]\00", %struct._depend_unpack_t.7 ([7 x i32]*, i64)* @compute_dep.8, [7 x i32]* %vla, i64 %1), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [9 x i8] c"aVLA[1:]\00", %struct._depend_unpack_t.8 ([7 x i32]*, i64)* @compute_dep.9, [7 x i32]* %vla, i64 %1), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [10 x i8] c"aVLA[: 2]\00", %struct._depend_unpack_t.9 ([7 x i32]*, i64)* @compute_dep.10, [7 x i32]* %vla, i64 %1), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [12 x i8] c"aVLA[3 : 4]\00", %struct._depend_unpack_t.10 ([7 x i32]*, i64)* @compute_dep.11, [7 x i32]* %vla, i64 %1) ]
// CHECK: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([7 x i32]* %vla), "QUAL.OSS.VLA.DIMS"([7 x i32]* %vla, i64 %1, i64 7), "QUAL.OSS.CAPTURED"(i64 %1, i64 7), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [11 x i8] c"aVLA[5][:]\00", %struct._depend_unpack_t.11 ([7 x i32]*, i64)* @compute_dep.12, [7 x i32]* %vla, i64 %1), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [12 x i8] c"aVLA[5][1:]\00", %struct._depend_unpack_t.12 ([7 x i32]*, i64)* @compute_dep.13, [7 x i32]* %vla, i64 %1), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [13 x i8] c"aVLA[5][: 2]\00", %struct._depend_unpack_t.13 ([7 x i32]*, i64)* @compute_dep.14, [7 x i32]* %vla, i64 %1), "QUAL.OSS.DEP.IN"([7 x i32]* %vla, [15 x i8] c"aVLA[5][3 : 4]\00", %struct._depend_unpack_t.14 ([7 x i32]*, i64)* @compute_dep.15, [7 x i32]* %vla, i64 %1) ]


// CHECK: define internal %struct._depend_unpack_t @compute_dep([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 10, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 1, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 10, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 2, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.2 @compute_dep.3([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.2, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 3, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 7, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.2 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.4([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.3, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.3 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.4 @compute_dep.5([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.4, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 8, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 80, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.4 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.5 @compute_dep.6([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.5, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 16, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.5 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.6 @compute_dep.7([10 x [10 x i32*]]* %aFix)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.6, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [10 x i32*]], [10 x [10 x i32*]]* %aFix, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32*]* %arraydecay, [10 x i32*]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 24, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 56, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.6 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.7 @compute_dep.8([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.7, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %0, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.7 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.8 @compute_dep.9([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.8, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 1, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %0, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.8 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.9 @compute_dep.10([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.9, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 2, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.9 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.10 @compute_dep.11([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.10, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 7, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.10 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.11 @compute_dep.12([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.11, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.11 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.12 @compute_dep.13([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.12, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.12 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.13 @compute_dep.14([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.13, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 8, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.13 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.14 @compute_dep.15([7 x i32]* %aVLA, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.14, align 8
// CHECK:   %1 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 0
// CHECK-NEXT:   store [7 x i32]* %aVLA, [7 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 12, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 28, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 5, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 6, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.14 %8
// CHECK-NEXT: }


void bar() {
    int **p;
    int (*kk)[10];
    int array[10][20];
    #pragma oss task depend(in: kk[0 : 11])
    {}
    #pragma oss task depend(in: p[0 : 11])
    {}
    #pragma oss task depend(in: array[0: 11][7 : 11])
    {}
    struct C {
        int (*x)[10];
    } c;

    #pragma oss task depend(in: c.x[0 : 11])
    {}

    #pragma oss task in(kk[0 ; 11])
    {}
    #pragma oss task in(p[0 ; 11])
    {}
    #pragma oss task in(array[0; 11][7 ; 11])
    {}

    #pragma oss task in(kk[0 : 11 - 1])
    {}
    #pragma oss task in(p[0 : 11 - 1])
    {}
    #pragma oss task in(array[0: 11 - 1][7 : 7 + 11 - 1])
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]** %kk), "QUAL.OSS.DEP.IN"([10 x i32]** %kk, [11 x i8] c"kk[0 : 11]\00", %struct._depend_unpack_t.15 ([10 x i32]**)* @compute_dep.16, [10 x i32]** %kk) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p), "QUAL.OSS.DEP.IN"(i32*** %p, [10 x i8] c"p[0 : 11]\00", %struct._depend_unpack_t.16 (i32***)* @compute_dep.17, i32*** %p) ]
// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* %array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* %array, [21 x i8] c"array[0: 11][7 : 11]\00", %struct._depend_unpack_t.17 ([10 x [20 x i32]]*)* @compute_dep.18, [10 x [20 x i32]]* %array) ]
// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.C* %c), "QUAL.OSS.DEP.IN"(%struct.C* %c, [12 x i8] c"c.x[0 : 11]\00", %struct._depend_unpack_t.18 (%struct.C*)* @compute_dep.19, %struct.C* %c) ]
// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]** %kk), "QUAL.OSS.DEP.IN"([10 x i32]** %kk, [11 x i8] c"kk[0 ; 11]\00", %struct._depend_unpack_t.19 ([10 x i32]**)* @compute_dep.20, [10 x i32]** %kk) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p), "QUAL.OSS.DEP.IN"(i32*** %p, [10 x i8] c"p[0 ; 11]\00", %struct._depend_unpack_t.20 (i32***)* @compute_dep.21, i32*** %p) ]
// CHECK: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* %array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* %array, [21 x i8] c"array[0; 11][7 ; 11]\00", %struct._depend_unpack_t.21 ([10 x [20 x i32]]*)* @compute_dep.22, [10 x [20 x i32]]* %array) ]
// CHECK: %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]** %kk), "QUAL.OSS.DEP.IN"([10 x i32]** %kk, [15 x i8] c"kk[0 : 11 - 1]\00", %struct._depend_unpack_t.22 ([10 x i32]**)* @compute_dep.23, [10 x i32]** %kk) ]
// CHECK: %8 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p), "QUAL.OSS.DEP.IN"(i32*** %p, [14 x i8] c"p[0 : 11 - 1]\00", %struct._depend_unpack_t.23 (i32***)* @compute_dep.24, i32*** %p) ]
// CHECK: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* %array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* %array, [33 x i8] c"array[0: 11 - 1][7 : 7 + 11 - 1]\00", %struct._depend_unpack_t.24 ([10 x [20 x i32]]*)* @compute_dep.25, [10 x [20 x i32]]* %array) ]


// CHECK: define internal %struct._depend_unpack_t.15 @compute_dep.16([10 x i32]** %kk)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.15, align 8
// CHECK:   %0 = load [10 x i32]*, [10 x i32]** %kk, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32]* %0, [10 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 40, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 11, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.15, %struct._depend_unpack_t.15* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.15 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.16 @compute_dep.17(i32*** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.16, align 8
// CHECK:   %0 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.16, %struct._depend_unpack_t.16* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %0, i32*** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.16, %struct._depend_unpack_t.16* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 88, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.16, %struct._depend_unpack_t.16* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.16, %struct._depend_unpack_t.16* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 88, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.16, %struct._depend_unpack_t.16* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.16 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.17 @compute_dep.18([10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.17, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 0
// CHECK-NEXT:   store [20 x i32]* %arraydecay, [20 x i32]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 72, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.17, %struct._depend_unpack_t.17* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.17 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.18 @compute_dep.19(%struct.C* %c)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.18, align 8
// CHECK:   %x = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 0
// CHECK-NEXT:   %0 = load [10 x i32]*, [10 x i32]** %x, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32]* %0, [10 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 40, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 11, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.18, %struct._depend_unpack_t.18* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.18 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.19 @compute_dep.20([10 x i32]** %kk)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.19, align 8
// CHECK:   %0 = load [10 x i32]*, [10 x i32]** %kk, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32]* %0, [10 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 40, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 11, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.19, %struct._depend_unpack_t.19* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.19 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.20 @compute_dep.21(i32*** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.20, align 8
// CHECK:   %0 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.20, %struct._depend_unpack_t.20* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %0, i32*** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.20, %struct._depend_unpack_t.20* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 88, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.20, %struct._depend_unpack_t.20* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.20, %struct._depend_unpack_t.20* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 88, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.20, %struct._depend_unpack_t.20* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.20 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.21 @compute_dep.22([10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.21, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 0
// CHECK-NEXT:   store [20 x i32]* %arraydecay, [20 x i32]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 72, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.21, %struct._depend_unpack_t.21* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.21 %7
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.22 @compute_dep.23([10 x i32]** %kk)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.22, align 8
// CHECK:   %0 = load [10 x i32]*, [10 x i32]** %kk, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 0
// CHECK-NEXT:   store [10 x i32]* %0, [10 x i32]** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 40, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.22, %struct._depend_unpack_t.22* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.22 %8
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.23 @compute_dep.24(i32*** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.23, align 8
// CHECK:   %0 = load i32**, i32*** %p, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.23, %struct._depend_unpack_t.23* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %0, i32*** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.23, %struct._depend_unpack_t.23* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.23, %struct._depend_unpack_t.23* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.23, %struct._depend_unpack_t.23* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 88, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.23, %struct._depend_unpack_t.23* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.23 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.24 @compute_dep.25([10 x [20 x i32]]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.24, align 8
// CHECK:   %arraydecay = getelementptr inbounds [10 x [20 x i32]], [10 x [20 x i32]]* %array, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 0
// CHECK-NEXT:   store [20 x i32]* %arraydecay, [20 x i32]** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 80, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 28, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 72, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 10, i64* %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 11, i64* %6, align 8
// CHECK-NEXT:   %7 = load %struct._depend_unpack_t.24, %struct._depend_unpack_t.24* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.24 %7
// CHECK-NEXT: }
