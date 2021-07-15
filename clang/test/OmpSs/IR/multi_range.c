// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int v[10][20];
void constants() {
  #pragma oss task in( { v[i][j], i = 0:10-1, j=0;20 } )
  {}
  #pragma oss task in( { v[i][j], i = 0:10-1:1, j=0;20:1 } )
  {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t (i32*, i32*, i64)* @compute_dep, i32* %i, i32* %j, [10 x [20 x i32]]* @v, [32 x i8] c"{ v[i][j], i = 0:10-1, j=0;20 }\00", %struct._depend_unpack_t.0 (i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.1, i32* %i, i32* %j, [10 x [20 x i32]]* @v) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.PRIVATE"(i32* %j2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i1, i32* %j2, %struct._depend_unpack_t.1 (i32*, i32*, i64)* @compute_dep.2, i32* %i1, i32* %j2, [10 x [20 x i32]]* @v, [36 x i8] c"{ v[i][j], i = 0:10-1:1, j=0;20:1 }\00", %struct._depend_unpack_t.2 (i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.3, i32* %i1, i32* %j2, [10 x [20 x i32]]* @v) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %10
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %10, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = load i32, i32* %i, align 4
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %6, align 4
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %5, i32* %7, align 4
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 9, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %9, align 4
// CHECK-NEXT:   br label %1
// CHECK: 10:                                               ; preds = %entry
// CHECK-NEXT:   %11 = load i32, i32* %j, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %11, i32* %13, align 4
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 19, i32* %14, align 4
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %15, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32* %i, i32* %j, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %10
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %10, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = load i32, i32* %i, align 4
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %6, align 4
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %5, i32* %7, align 4
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 9, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %9, align 4
// CHECK-NEXT:   br label %1
// CHECK: 10:                                               ; preds = %entry
// CHECK-NEXT:   %11 = load i32, i32* %j, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %11, i32* %13, align 4
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 19, i32* %14, align 4
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %15, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

void nonconstants_int(
    int lb1, int ub1, int step1,
    int lb2, int ub2, int step2) {

  #pragma oss task in( { v[i][j], i = lb1:ub1, j=lb2;ub2 } )
  {}
  #pragma oss task in( { v[i][j], i = lb1:ub1:step1, j=lb2;ub2:step2 } )
  {}
}

// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb2.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub2.addr), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t.3 (i32*, i32*, i32*, i32*, i32*, i32*, i64)* @compute_dep.4, i32* %i, i32* %lb1.addr, i32* %ub1.addr, i32* %j, i32* %lb2.addr, i32* %ub2.addr, [10 x [20 x i32]]* @v, [36 x i8] c"{ v[i][j], i = lb1:ub1, j=lb2;ub2 }\00", %struct._depend_unpack_t.4 (i32*, i32*, i32*, i32*, i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.5, i32* %i, i32* %j, i32* %lb1.addr, i32* %lb2.addr, i32* %ub1.addr, i32* %ub2.addr, [10 x [20 x i32]]* @v) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.PRIVATE"(i32* %j2), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step1.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb2.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub2.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step2.addr), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i1, i32* %j2, %struct._depend_unpack_t.5 (i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i64)* @compute_dep.6, i32* %i1, i32* %lb1.addr, i32* %ub1.addr, i32* %step1.addr, i32* %j2, i32* %lb2.addr, i32* %ub2.addr, i32* %step2.addr, [10 x [20 x i32]]* @v, [48 x i8] c"{ v[i][j], i = lb1:ub1:step1, j=lb2;ub2:step2 }\00", %struct._depend_unpack_t.6 (i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.7, i32* %i1, i32* %j2, i32* %lb1.addr, i32* %lb2.addr, i32* %ub1.addr, i32* %ub2.addr, i32* %step1.addr, i32* %step2.addr, [10 x [20 x i32]]* @v) ]

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.4(i32* %i, i32* %lb1, i32* %ub1, i32* %j, i32* %lb2, i32* %ub2, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.3, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb1.addr = alloca i32*, align 8
// CHECK-NEXT:   %ub1.addr = alloca i32*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb2.addr = alloca i32*, align 8
// CHECK-NEXT:   %ub2.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i32* %lb1, i32** %lb1.addr, align 8
// CHECK-NEXT:   store i32* %ub1, i32** %ub1.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i32* %lb2, i32** %lb2.addr, align 8
// CHECK-NEXT:   store i32* %ub2, i32** %ub2.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %12
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %12, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.3 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = load i32, i32* %lb1, align 4
// CHECK-NEXT:   %6 = load i32, i32* %i, align 4
// CHECK-NEXT:   %7 = load i32, i32* %ub1, align 4
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 %5, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %6, i32* %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 %7, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %11, align 4
// CHECK-NEXT:   br label %1
// CHECK: 12:                                               ; preds = %entry
// CHECK-NEXT:   %13 = load i32, i32* %lb2, align 4
// CHECK-NEXT:   %14 = load i32, i32* %j, align 4
// CHECK-NEXT:   %15 = load i32, i32* %ub2, align 4
// CHECK-NEXT:   %16 = add i32 %13, %15
// CHECK-NEXT:   %17 = add i32 %16, -1
// CHECK-NEXT:   %18 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 %13, i32* %18, align 4
// CHECK-NEXT:   %19 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %14, i32* %19, align 4
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 %17, i32* %20, align 4
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %21, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.5 @compute_dep.6(i32* %i, i32* %lb1, i32* %ub1, i32* %step1, i32* %j, i32* %lb2, i32* %ub2, i32* %step2, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.5, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb1.addr = alloca i32*, align 8
// CHECK-NEXT:   %ub1.addr = alloca i32*, align 8
// CHECK-NEXT:   %step1.addr = alloca i32*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb2.addr = alloca i32*, align 8
// CHECK-NEXT:   %ub2.addr = alloca i32*, align 8
// CHECK-NEXT:   %step2.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i32* %lb1, i32** %lb1.addr, align 8
// CHECK-NEXT:   store i32* %ub1, i32** %ub1.addr, align 8
// CHECK-NEXT:   store i32* %step1, i32** %step1.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i32* %lb2, i32** %lb2.addr, align 8
// CHECK-NEXT:   store i32* %ub2, i32** %ub2.addr, align 8
// CHECK-NEXT:   store i32* %step2, i32** %step2.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %13
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %13, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.5 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = load i32, i32* %lb1, align 4
// CHECK-NEXT:   %6 = load i32, i32* %i, align 4
// CHECK-NEXT:   %7 = load i32, i32* %ub1, align 4
// CHECK-NEXT:   %8 = load i32, i32* %step1, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 %5, i32* %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %6, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 %7, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 %8, i32* %12, align 4
// CHECK-NEXT:   br label %1
// CHECK: 13:                                               ; preds = %entry
// CHECK-NEXT:   %14 = load i32, i32* %lb2, align 4
// CHECK-NEXT:   %15 = load i32, i32* %j, align 4
// CHECK-NEXT:   %16 = load i32, i32* %ub2, align 4
// CHECK-NEXT:   %17 = add i32 %14, %16
// CHECK-NEXT:   %18 = add i32 %17, -1
// CHECK-NEXT:   %19 = load i32, i32* %step2, align 4
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 %14, i32* %20, align 4
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %15, i32* %21, align 4
// CHECK-NEXT:   %22 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 %18, i32* %22, align 4
// CHECK-NEXT:   %23 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 %19, i32* %23, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

void nonconstants_short(
    short lb1, short ub1, short step1,
    short lb2, short ub2, short step2) {

  #pragma oss task in( { v[i][j], i = lb1:ub1, j=lb2;ub2 } )
  {}
  #pragma oss task in( { v[i][j], i = lb1:ub1:step1, j=lb2;ub2:step2 } )
  {}
}

// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb1.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub1.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb2.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub2.addr), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t.7 (i32*, i16*, i16*, i32*, i16*, i16*, i64)* @compute_dep.8, i32* %i, i16* %lb1.addr, i16* %ub1.addr, i32* %j, i16* %lb2.addr, i16* %ub2.addr, [10 x [20 x i32]]* @v, [36 x i8] c"{ v[i][j], i = lb1:ub1, j=lb2;ub2 }\00", %struct._depend_unpack_t.8 (i32*, i32*, i16*, i16*, i16*, i16*, [10 x [20 x i32]]*)* @compute_dep.9, i32* %i, i32* %j, i16* %lb1.addr, i16* %lb2.addr, i16* %ub1.addr, i16* %ub2.addr, [10 x [20 x i32]]* @v) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.PRIVATE"(i32* %j4), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb1.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub1.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %step1.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb2.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub2.addr), "QUAL.OSS.FIRSTPRIVATE"(i16* %step2.addr), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i2, i32* %j4, %struct._depend_unpack_t.9 (i32*, i16*, i16*, i16*, i32*, i16*, i16*, i16*, i64)* @compute_dep.10, i32* %i2, i16* %lb1.addr, i16* %ub1.addr, i16* %step1.addr, i32* %j4, i16* %lb2.addr, i16* %ub2.addr, i16* %step2.addr, [10 x [20 x i32]]* @v, [48 x i8] c"{ v[i][j], i = lb1:ub1:step1, j=lb2;ub2:step2 }\00", %struct._depend_unpack_t.10 (i32*, i32*, i16*, i16*, i16*, i16*, i16*, i16*, [10 x [20 x i32]]*)* @compute_dep.11, i32* %i2, i32* %j4, i16* %lb1.addr, i16* %lb2.addr, i16* %ub1.addr, i16* %ub2.addr, i16* %step1.addr, i16* %step2.addr, [10 x [20 x i32]]* @v) ]

// CHECK: define internal %struct._depend_unpack_t.7 @compute_dep.8(i32* %i, i16* %lb1, i16* %ub1, i32* %j, i16* %lb2, i16* %ub2, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.7, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb1.addr = alloca i16*, align 8
// CHECK-NEXT:   %ub1.addr = alloca i16*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb2.addr = alloca i16*, align 8
// CHECK-NEXT:   %ub2.addr = alloca i16*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i16* %lb1, i16** %lb1.addr, align 8
// CHECK-NEXT:   store i16* %ub1, i16** %ub1.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i16* %lb2, i16** %lb2.addr, align 8
// CHECK-NEXT:   store i16* %ub2, i16** %ub2.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %12
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %12, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.7 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv = sext i16 %5 to i32
// CHECK-NEXT:   %6 = load i32, i32* %i, align 4
// CHECK-NEXT:   %7 = load i16, i16* %ub1, align 2
// CHECK-NEXT:   %conv1 = sext i16 %7 to i32
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 %conv, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %6, i32* %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 %conv1, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %11, align 4
// CHECK-NEXT:   br label %1
// CHECK: 12:                                               ; preds = %entry
// CHECK-NEXT:   %13 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv2 = sext i16 %13 to i32
// CHECK-NEXT:   %14 = load i32, i32* %j, align 4
// CHECK-NEXT:   %15 = load i16, i16* %ub2, align 2
// CHECK-NEXT:   %conv3 = sext i16 %15 to i32
// CHECK-NEXT:   %16 = add i32 %conv2, %conv3
// CHECK-NEXT:   %17 = add i32 %16, -1
// CHECK-NEXT:   %18 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 %conv2, i32* %18, align 4
// CHECK-NEXT:   %19 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %14, i32* %19, align 4
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 %17, i32* %20, align 4
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %21, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.9 @compute_dep.10(i32* %i, i16* %lb1, i16* %ub1, i16* %step1, i32* %j, i16* %lb2, i16* %ub2, i16* %step2, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.9, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb1.addr = alloca i16*, align 8
// CHECK-NEXT:   %ub1.addr = alloca i16*, align 8
// CHECK-NEXT:   %step1.addr = alloca i16*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb2.addr = alloca i16*, align 8
// CHECK-NEXT:   %ub2.addr = alloca i16*, align 8
// CHECK-NEXT:   %step2.addr = alloca i16*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i16* %lb1, i16** %lb1.addr, align 8
// CHECK-NEXT:   store i16* %ub1, i16** %ub1.addr, align 8
// CHECK-NEXT:   store i16* %step1, i16** %step1.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i16* %lb2, i16** %lb2.addr, align 8
// CHECK-NEXT:   store i16* %ub2, i16** %ub2.addr, align 8
// CHECK-NEXT:   store i16* %step2, i16** %step2.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %13
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %13, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.9 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv = sext i16 %5 to i32
// CHECK-NEXT:   %6 = load i32, i32* %i, align 4
// CHECK-NEXT:   %7 = load i16, i16* %ub1, align 2
// CHECK-NEXT:   %conv1 = sext i16 %7 to i32
// CHECK-NEXT:   %8 = load i16, i16* %step1, align 2
// CHECK-NEXT:   %conv2 = sext i16 %8 to i32
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 %conv, i32* %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %6, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 %conv1, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 %conv2, i32* %12, align 4
// CHECK-NEXT:   br label %1
// CHECK: 13:                                               ; preds = %entry
// CHECK-NEXT:   %14 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv3 = sext i16 %14 to i32
// CHECK-NEXT:   %15 = load i32, i32* %j, align 4
// CHECK-NEXT:   %16 = load i16, i16* %ub2, align 2
// CHECK-NEXT:   %conv4 = sext i16 %16 to i32
// CHECK-NEXT:   %17 = add i32 %conv3, %conv4
// CHECK-NEXT:   %18 = add i32 %17, -1
// CHECK-NEXT:   %19 = load i16, i16* %step2, align 2
// CHECK-NEXT:   %conv5 = sext i16 %19 to i32
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 %conv3, i32* %20, align 4
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %15, i32* %21, align 4
// CHECK-NEXT:   %22 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 %18, i32* %22, align 4
// CHECK-NEXT:   %23 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 %conv5, i32* %23, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

