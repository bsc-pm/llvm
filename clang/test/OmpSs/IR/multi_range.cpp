// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int v[10][20];
template<typename T>
void constants() {
  #pragma oss task in( { v[i][j], i = 0:10-1, j=0;20 } )
  {}
  #pragma oss task in( { v[i][j], i = 0:10-1:1, j=0;20:1 } )
  {}
}

template<typename T>
void nonconstants() {
  T lb1, ub1, step1;
  T lb2, ub2, step2;
  #pragma oss task in( { v[i][j], i = lb1:ub1, j=lb2;ub2 } )
  {}
  #pragma oss task in( { v[i][j], i = lb1:ub1:step1, j=lb2;ub2:step2 } )
  {}
}

void f() {
  constants<int>();
  nonconstants<int>();
  nonconstants<short>();
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t (i32*, i32*)* @compute_dep, i32* %i, i32* %j, [10 x [20 x i32]]* @v, %struct._depend_unpack_t.0 (i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.1, i32* %i, i32* %j, [10 x [20 x i32]]* @v) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.PRIVATE"(i32* %j2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i1, i32* %j2, %struct._depend_unpack_t.1 (i32*, i32*)* @compute_dep.2, i32* %i1, i32* %j2, [10 x [20 x i32]]* @v, %struct._depend_unpack_t.2 (i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.3, i32* %i1, i32* %j2, [10 x [20 x i32]]* @v) ]

// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb1), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub1), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb2), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t.3 (i32*, i32*, i32*, i32*, i32*, i32*)* @compute_dep.4, i32* %i, i32* %lb1, i32* %ub1, i32* %j, i32* %lb2, i32* %ub2, [10 x [20 x i32]]* @v, %struct._depend_unpack_t.4 (i32*, i32*, i32*, i32*, i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.5, i32* %i, i32* %j, i32* %lb1, i32* %lb2, i32* %ub1, i32* %ub2, [10 x [20 x i32]]* @v) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.PRIVATE"(i32* %j2), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb1), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub1), "QUAL.OSS.FIRSTPRIVATE"(i32* %step1), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb2), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub2), "QUAL.OSS.FIRSTPRIVATE"(i32* %step2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i1, i32* %j2, %struct._depend_unpack_t.5 (i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*)* @compute_dep.6, i32* %i1, i32* %lb1, i32* %ub1, i32* %step1, i32* %j2, i32* %lb2, i32* %ub2, i32* %step2, [10 x [20 x i32]]* @v, %struct._depend_unpack_t.6 (i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.7, i32* %i1, i32* %j2, i32* %lb1, i32* %lb2, i32* %ub1, i32* %ub2, i32* %step1, i32* %step2, [10 x [20 x i32]]* @v) ]

// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb1), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub1), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb2), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t.7 (i32*, i16*, i16*, i32*, i16*, i16*)* @compute_dep.8, i32* %i, i16* %lb1, i16* %ub1, i32* %j, i16* %lb2, i16* %ub2, [10 x [20 x i32]]* @v, %struct._depend_unpack_t.8 (i32*, i32*, i16*, i16*, i16*, i16*, [10 x [20 x i32]]*)* @compute_dep.9, i32* %i, i32* %j, i16* %lb1, i16* %lb2, i16* %ub1, i16* %ub2, [10 x [20 x i32]]* @v) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.PRIVATE"(i32* %j4), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb1), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub1), "QUAL.OSS.FIRSTPRIVATE"(i16* %step1), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb2), "QUAL.OSS.FIRSTPRIVATE"(i16* %ub2), "QUAL.OSS.FIRSTPRIVATE"(i16* %step2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i2, i32* %j4, %struct._depend_unpack_t.9 (i32*, i16*, i16*, i16*, i32*, i16*, i16*, i16*)* @compute_dep.10, i32* %i2, i16* %lb1, i16* %ub1, i16* %step1, i32* %j4, i16* %lb2, i16* %ub2, i16* %step2, [10 x [20 x i32]]* @v, %struct._depend_unpack_t.10 (i32*, i32*, i16*, i16*, i16*, i16*, i16*, i16*, [10 x [20 x i32]]*)* @compute_dep.11, i32* %i2, i32* %j4, i16* %lb1, i16* %lb2, i16* %ub1, i16* %ub2, i16* %step1, i16* %step2, [10 x [20 x i32]]* @v) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j) {
// CHECK: entry:
// CHECK-NEXT:   %return.val = alloca %struct._depend_unpack_t, align 4
// CHECK-NEXT:   %0 = load i32, i32* %i, align 4
// CHECK-NEXT:   %1 = load i32, i32* %j, align 4
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
// CHECK-NEXT:   store i32 %0, i32* %3, align 4
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
// CHECK-NEXT:   store i32 9, i32* %4, align 4
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %5, align 4
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %6, align 4
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 5
// CHECK-NEXT:   store i32 %1, i32* %7, align 4
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 6
// CHECK-NEXT:   store i32 19, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %9, align 4
// CHECK-NEXT:   %10 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t %10
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32* %i1, i32* %j2) {
// CHECK: entry:
// CHECK-NEXT:   %return.val = alloca %struct._depend_unpack_t.1, align 4
// CHECK-NEXT:   %0 = load i32, i32* %i1, align 4
// CHECK-NEXT:   %1 = load i32, i32* %j2, align 4
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 1
// CHECK-NEXT:   store i32 %0, i32* %3, align 4
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 2
// CHECK-NEXT:   store i32 9, i32* %4, align 4
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %5, align 4
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %6, align 4
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 5
// CHECK-NEXT:   store i32 %1, i32* %7, align 4
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 6
// CHECK-NEXT:   store i32 19, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %9, align 4
// CHECK-NEXT:   %10 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %10
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.4(i32* %i, i32* %lb1, i32* %ub1, i32* %j, i32* %lb2, i32* %ub2) {
// CHECK: entry:
// CHECK:   %return.val = alloca %struct._depend_unpack_t.3, align 4
// CHECK:   %0 = load i32, i32* %lb1, align 4
// CHECK:   %1 = load i32, i32* %i, align 4
// CHECK:   %2 = load i32, i32* %ub1, align 4
// CHECK:   %3 = load i32, i32* %lb2, align 4
// CHECK:   %4 = load i32, i32* %j, align 4
// CHECK:   %5 = load i32, i32* %ub2, align 4
// CHECK:   %6 = add i32 %3, %5
// CHECK:   %7 = add i32 %6, -1
// CHECK:   %8 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 0
// CHECK:   store i32 %0, i32* %8, align 4
// CHECK:   %9 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 1
// CHECK:   store i32 %1, i32* %9, align 4
// CHECK:   %10 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 2
// CHECK:   store i32 %2, i32* %10, align 4
// CHECK:   %11 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 3
// CHECK:   store i32 1, i32* %11, align 4
// CHECK:   %12 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 4
// CHECK:   store i32 %3, i32* %12, align 4
// CHECK:   %13 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 5
// CHECK:   store i32 %4, i32* %13, align 4
// CHECK:   %14 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 6
// CHECK:   store i32 %7, i32* %14, align 4
// CHECK:   %15 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 7
// CHECK:   store i32 1, i32* %15, align 4
// CHECK:   %16 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, align 4
// CHECK:   ret %struct._depend_unpack_t.3 %16
// CHECK: }

// CHECK: define internal %struct._depend_unpack_t.5 @compute_dep.6(i32* %i1, i32* %lb1, i32* %ub1, i32* %step1, i32* %j2, i32* %lb2, i32* %ub2, i32* %step2) {
// CHECK: entry:
// CHECK-NEXT:   %return.val = alloca %struct._depend_unpack_t.5, align 4
// CHECK-NEXT:   %0 = load i32, i32* %lb1, align 4
// CHECK-NEXT:   %1 = load i32, i32* %i1, align 4
// CHECK-NEXT:   %2 = load i32, i32* %ub1, align 4
// CHECK-NEXT:   %3 = load i32, i32* %step1, align 4
// CHECK-NEXT:   %4 = load i32, i32* %lb2, align 4
// CHECK-NEXT:   %5 = load i32, i32* %j2, align 4
// CHECK-NEXT:   %6 = load i32, i32* %ub2, align 4
// CHECK-NEXT:   %7 = add i32 %4, %6
// CHECK-NEXT:   %8 = add i32 %7, -1
// CHECK-NEXT:   %9 = load i32, i32* %step2, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 0
// CHECK-NEXT:   store i32 %0, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 1
// CHECK-NEXT:   store i32 %1, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 2
// CHECK-NEXT:   store i32 %2, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 3
// CHECK-NEXT:   store i32 %3, i32* %13, align 4
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 4
// CHECK-NEXT:   store i32 %4, i32* %14, align 4
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 5
// CHECK-NEXT:   store i32 %5, i32* %15, align 4
// CHECK-NEXT:   %16 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 6
// CHECK-NEXT:   store i32 %8, i32* %16, align 4
// CHECK-NEXT:   %17 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 7
// CHECK-NEXT:   store i32 %9, i32* %17, align 4
// CHECK-NEXT:   %18 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.5 %18
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.7 @compute_dep.8(i32* %i, i16* %lb1, i16* %ub1, i32* %j, i16* %lb2, i16* %ub2) {
// CHECK: entry:
// CHECK-NEXT:   %return.val = alloca %struct._depend_unpack_t.7, align 4
// CHECK-NEXT:   %0 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv = sext i16 %0 to i32
// CHECK-NEXT:   %1 = load i32, i32* %i, align 4
// CHECK-NEXT:   %2 = load i16, i16* %ub1, align 2
// CHECK-NEXT:   %conv1 = sext i16 %2 to i32
// CHECK-NEXT:   %3 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv2 = sext i16 %3 to i32
// CHECK-NEXT:   %4 = load i32, i32* %j, align 4
// CHECK-NEXT:   %5 = load i16, i16* %ub2, align 2
// CHECK-NEXT:   %conv3 = sext i16 %5 to i32
// CHECK-NEXT:   %6 = add i32 %conv2, %conv3
// CHECK-NEXT:   %7 = add i32 %6, -1
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 0
// CHECK-NEXT:   store i32 %conv, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 1
// CHECK-NEXT:   store i32 %1, i32* %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 2
// CHECK-NEXT:   store i32 %conv1, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 4
// CHECK-NEXT:   store i32 %conv2, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 5
// CHECK-NEXT:   store i32 %4, i32* %13, align 4
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 6
// CHECK-NEXT:   store i32 %7, i32* %14, align 4
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %15, align 4
// CHECK-NEXT:   %16 = load %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.7 %16
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.9 @compute_dep.10(i32* %i2, i16* %lb1, i16* %ub1, i16* %step1, i32* %j4, i16* %lb2, i16* %ub2, i16* %step2) {
// CHECK: entry:
// CHECK-NEXT:   %return.val = alloca %struct._depend_unpack_t.9, align 4
// CHECK-NEXT:   %0 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv = sext i16 %0 to i32
// CHECK-NEXT:   %1 = load i32, i32* %i2, align 4
// CHECK-NEXT:   %2 = load i16, i16* %ub1, align 2
// CHECK-NEXT:   %conv1 = sext i16 %2 to i32
// CHECK-NEXT:   %3 = load i16, i16* %step1, align 2
// CHECK-NEXT:   %conv2 = sext i16 %3 to i32
// CHECK-NEXT:   %4 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv3 = sext i16 %4 to i32
// CHECK-NEXT:   %5 = load i32, i32* %j4, align 4
// CHECK-NEXT:   %6 = load i16, i16* %ub2, align 2
// CHECK-NEXT:   %conv4 = sext i16 %6 to i32
// CHECK-NEXT:   %7 = add i32 %conv3, %conv4
// CHECK-NEXT:   %8 = add i32 %7, -1
// CHECK-NEXT:   %9 = load i16, i16* %step2, align 2
// CHECK-NEXT:   %conv5 = sext i16 %9 to i32
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 0
// CHECK-NEXT:   store i32 %conv, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 1
// CHECK-NEXT:   store i32 %1, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 2
// CHECK-NEXT:   store i32 %conv1, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 3
// CHECK-NEXT:   store i32 %conv2, i32* %13, align 4
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 4
// CHECK-NEXT:   store i32 %conv3, i32* %14, align 4
// CHECK-NEXT:   %15 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 5
// CHECK-NEXT:   store i32 %5, i32* %15, align 4
// CHECK-NEXT:   %16 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 6
// CHECK-NEXT:   store i32 %8, i32* %16, align 4
// CHECK-NEXT:   %17 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, i32 0, i32 7
// CHECK-NEXT:   store i32 %conv5, i32* %17, align 4
// CHECK-NEXT:   %18 = load %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %return.val, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.9 %18
// CHECK-NEXT: }

