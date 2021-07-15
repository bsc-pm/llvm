// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int v[10][20];
template<typename T>
void constants() {
  #pragma oss task in( { v[i][j], i = {0, 1, 2}, j={3, 4, 5} } )
  {}
}

template<typename T>
void nonconstants() {
  T lb1, ub1, step1;
  T lb2, ub2, step2;
  #pragma oss task in( { v[i][j], i = {lb1, lb1 + 1, lb1 + 2}, j={lb2, lb2 + 1, lb2 + 2} } )
  {}
}

void f() {
  constants<int>();
  nonconstants<int>();
  nonconstants<short>();
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t (i32*, i32*, i64)* @compute_dep, i32* %i, i32* %j, [10 x [20 x i32]]* @v, [40 x i8] c"{ v[i][j], i = {0, 1, 2}, j={3, 4, 5} }\00", %struct._depend_unpack_t.0 (i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.2, i32* %i, i32* %j, [10 x [20 x i32]]* @v) ]
// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb1), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t.1 (i32*, i32*, i32*, i32*, i64)* @compute_dep.3, i32* %i, i32* %lb1, i32* %j, i32* %lb2, [10 x [20 x i32]]* @v, [68 x i8] c"{ v[i][j], i = {lb1, lb1 + 1, lb1 + 2}, j={lb2, lb2 + 1, lb2 + 2} }\00", %struct._depend_unpack_t.2 (i32*, i32*, i32*, i32*, [10 x [20 x i32]]*)* @compute_dep.4, i32* %i, i32* %j, i32* %lb1, i32* %lb2, [10 x [20 x i32]]* @v) ]
// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb1), "QUAL.OSS.FIRSTPRIVATE"(i16* %lb2), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %i, i32* %j, %struct._depend_unpack_t.3 (i32*, i16*, i32*, i16*, i64)* @compute_dep.5, i32* %i, i16* %lb1, i32* %j, i16* %lb2, [10 x [20 x i32]]* @v, [68 x i8] c"{ v[i][j], i = {lb1, lb1 + 1, lb1 + 2}, j={lb2, lb2 + 1, lb2 + 2} }\00", %struct._depend_unpack_t.4 (i32*, i32*, i16*, i16*, [10 x [20 x i32]]*)* @compute_dep.6, i32* %i, i32* %j, i16* %lb1, i16* %lb2, [10 x [20 x i32]]* @v) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   %discrete.array = alloca [3 x i32], align 4
// CHECK-NEXT:   %discrete.array1 = alloca [3 x i32], align 4
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %12
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %12, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %5 = bitcast [3 x i32]* %discrete.array to i8*
// CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %5, i8* align 4 bitcast ([3 x i32]* @__const._Z9constantsIiEvv.discrete.array to i8*), i64 12, i1 false)
// CHECK-NEXT:   %6 = load i32, i32* %i, align 4
// CHECK-NEXT:   %discreteidx = getelementptr [3 x i32], [3 x i32]* %discrete.array, i32 0, i32 %6
// CHECK-NEXT:   %7 = load i32, i32* %discreteidx, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %8, align 4
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %7, i32* %9, align 4
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 2, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %11, align 4
// CHECK-NEXT:   br label %1
// CHECK: 12:                                               ; preds = %entry
// CHECK-NEXT:   %13 = bitcast [3 x i32]* %discrete.array1 to i8*
// CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %13, i8* align 4 bitcast ([3 x i32]* @__const._Z9constantsIiEvv.discrete.array.1 to i8*), i64 12, i1 false)
// CHECK-NEXT:   %14 = load i32, i32* %j, align 4
// CHECK-NEXT:   %discreteidx2 = getelementptr [3 x i32], [3 x i32]* %discrete.array1, i32 0, i32 %14
// CHECK-NEXT:   %15 = load i32, i32* %discreteidx2, align 8
// CHECK-NEXT:   %16 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %16, align 4
// CHECK-NEXT:   %17 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %15, i32* %17, align 4
// CHECK-NEXT:   %18 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 2, i32* %18, align 4
// CHECK-NEXT:   %19 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %19, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.3(i32* %i, i32* %lb1, i32* %j, i32* %lb2, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb1.addr = alloca i32*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb2.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   %discrete.array = alloca [3 x i32], align 4
// CHECK-NEXT:   %discrete.array3 = alloca [3 x i32], align 4
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i32* %lb1, i32** %lb1.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i32* %lb2, i32** %lb2.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %14
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %14, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %arrayinit.begin = getelementptr inbounds [3 x i32], [3 x i32]* %discrete.array, i64 0, i64 0
// CHECK-NEXT:   %5 = load i32, i32* %lb1, align 4
// CHECK-NEXT:   store i32 %5, i32* %arrayinit.begin, align 4
// CHECK-NEXT:   %arrayinit.element = getelementptr inbounds i32, i32* %arrayinit.begin, i64 1
// CHECK-NEXT:   %6 = load i32, i32* %lb1, align 4
// CHECK-NEXT:   %add = add nsw i32 %6, 1
// CHECK-NEXT:   store i32 %add, i32* %arrayinit.element, align 4
// CHECK-NEXT:   %arrayinit.element1 = getelementptr inbounds i32, i32* %arrayinit.element, i64 1
// CHECK-NEXT:   %7 = load i32, i32* %lb1, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %7, 2
// CHECK-NEXT:   store i32 %add2, i32* %arrayinit.element1, align 4
// CHECK-NEXT:   %8 = load i32, i32* %i, align 4
// CHECK-NEXT:   %discreteidx = getelementptr [3 x i32], [3 x i32]* %discrete.array, i32 0, i32 %8
// CHECK-NEXT:   %9 = load i32, i32* %discreteidx, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %9, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 2, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %13, align 4
// CHECK-NEXT:   br label %1
// CHECK: 14:                                               ; preds = %entry
// CHECK-NEXT:   %arrayinit.begin4 = getelementptr inbounds [3 x i32], [3 x i32]* %discrete.array3, i64 0, i64 0
// CHECK-NEXT:   %15 = load i32, i32* %lb2, align 4
// CHECK-NEXT:   store i32 %15, i32* %arrayinit.begin4, align 4
// CHECK-NEXT:   %arrayinit.element5 = getelementptr inbounds i32, i32* %arrayinit.begin4, i64 1
// CHECK-NEXT:   %16 = load i32, i32* %lb2, align 4
// CHECK-NEXT:   %add6 = add nsw i32 %16, 1
// CHECK-NEXT:   store i32 %add6, i32* %arrayinit.element5, align 4
// CHECK-NEXT:   %arrayinit.element7 = getelementptr inbounds i32, i32* %arrayinit.element5, i64 1
// CHECK-NEXT:   %17 = load i32, i32* %lb2, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %17, 2
// CHECK-NEXT:   store i32 %add8, i32* %arrayinit.element7, align 4
// CHECK-NEXT:   %18 = load i32, i32* %j, align 4
// CHECK-NEXT:   %discreteidx9 = getelementptr [3 x i32], [3 x i32]* %discrete.array3, i32 0, i32 %18
// CHECK-NEXT:   %19 = load i32, i32* %discreteidx9, align 8
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %20, align 4
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %19, i32* %21, align 4
// CHECK-NEXT:   %22 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 2, i32* %22, align 4
// CHECK-NEXT:   %23 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %23, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.5(i32* %i, i16* %lb1, i32* %j, i16* %lb2, i64 %0)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.3, align 4
// CHECK-NEXT:   %i.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb1.addr = alloca i16*, align 8
// CHECK-NEXT:   %j.addr = alloca i32*, align 8
// CHECK-NEXT:   %lb2.addr = alloca i16*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   %discrete.array = alloca [3 x i32], align 4
// CHECK-NEXT:   %discrete.array5 = alloca [3 x i32], align 4
// CHECK-NEXT:   store i32* %i, i32** %i.addr, align 8
// CHECK-NEXT:   store i16* %lb1, i16** %lb1.addr, align 8
// CHECK-NEXT:   store i32* %j, i32** %j.addr, align 8
// CHECK-NEXT:   store i16* %lb2, i16** %lb2.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   switch i64 %0, label %3 [
// CHECK-NEXT:     i64 0, label %4
// CHECK-NEXT:     i64 1, label %14
// CHECK-NEXT:   ]
// CHECK: 1:                                                ; preds = %14, %4, %3
// CHECK-NEXT:   %2 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 4
// CHECK-NEXT:   ret %struct._depend_unpack_t.3 %2
// CHECK: 3:                                                ; preds = %entry
// CHECK-NEXT:   br label %1
// CHECK: 4:                                                ; preds = %entry
// CHECK-NEXT:   %arrayinit.begin = getelementptr inbounds [3 x i32], [3 x i32]* %discrete.array, i64 0, i64 0
// CHECK-NEXT:   %5 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv = sext i16 %5 to i32
// CHECK-NEXT:   store i32 %conv, i32* %arrayinit.begin, align 4
// CHECK-NEXT:   %arrayinit.element = getelementptr inbounds i32, i32* %arrayinit.begin, i64 1
// CHECK-NEXT:   %6 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv1 = sext i16 %6 to i32
// CHECK-NEXT:   %add = add nsw i32 %conv1, 1
// CHECK-NEXT:   store i32 %add, i32* %arrayinit.element, align 4
// CHECK-NEXT:   %arrayinit.element2 = getelementptr inbounds i32, i32* %arrayinit.element, i64 1
// CHECK-NEXT:   %7 = load i16, i16* %lb1, align 2
// CHECK-NEXT:   %conv3 = sext i16 %7 to i32
// CHECK-NEXT:   %add4 = add nsw i32 %conv3, 2
// CHECK-NEXT:   store i32 %add4, i32* %arrayinit.element2, align 4
// CHECK-NEXT:   %8 = load i32, i32* %i, align 4
// CHECK-NEXT:   %discreteidx = getelementptr [3 x i32], [3 x i32]* %discrete.array, i32 0, i32 %8
// CHECK-NEXT:   %9 = load i32, i32* %discreteidx, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32 0, i32* %10, align 4
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
// CHECK-NEXT:   store i32 %9, i32* %11, align 4
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
// CHECK-NEXT:   store i32 2, i32* %12, align 4
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
// CHECK-NEXT:   store i32 1, i32* %13, align 4
// CHECK-NEXT:   br label %1
// CHECK: 14:                                               ; preds = %entry
// CHECK-NEXT:   %arrayinit.begin6 = getelementptr inbounds [3 x i32], [3 x i32]* %discrete.array5, i64 0, i64 0
// CHECK-NEXT:   %15 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv7 = sext i16 %15 to i32
// CHECK-NEXT:   store i32 %conv7, i32* %arrayinit.begin6, align 4
// CHECK-NEXT:   %arrayinit.element8 = getelementptr inbounds i32, i32* %arrayinit.begin6, i64 1
// CHECK-NEXT:   %16 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv9 = sext i16 %16 to i32
// CHECK-NEXT:   %add10 = add nsw i32 %conv9, 1
// CHECK-NEXT:   store i32 %add10, i32* %arrayinit.element8, align 4
// CHECK-NEXT:   %arrayinit.element11 = getelementptr inbounds i32, i32* %arrayinit.element8, i64 1
// CHECK-NEXT:   %17 = load i16, i16* %lb2, align 2
// CHECK-NEXT:   %conv12 = sext i16 %17 to i32
// CHECK-NEXT:   %add13 = add nsw i32 %conv12, 2
// CHECK-NEXT:   store i32 %add13, i32* %arrayinit.element11, align 4
// CHECK-NEXT:   %18 = load i32, i32* %j, align 4
// CHECK-NEXT:   %discreteidx14 = getelementptr [3 x i32], [3 x i32]* %discrete.array5, i32 0, i32 %18
// CHECK-NEXT:   %19 = load i32, i32* %discreteidx14, align 8
// CHECK-NEXT:   %20 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 4
// CHECK-NEXT:   store i32 0, i32* %20, align 4
// CHECK-NEXT:   %21 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 5
// CHECK-NEXT:   store i32 %19, i32* %21, align 4
// CHECK-NEXT:   %22 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 6
// CHECK-NEXT:   store i32 2, i32* %22, align 4
// CHECK-NEXT:   %23 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 7
// CHECK-NEXT:   store i32 1, i32* %23, align 4
// CHECK-NEXT:   br label %1
// CHECK-NEXT: }

