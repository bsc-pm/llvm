; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_depend_proper_rtcall.c'
source_filename = "task_depend_proper_rtcall.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; struct S {
;     int x;
; } s;
; int array[10][20];
; int main() {
;     int n;
;     #pragma oss task in(array, array, array)
;     {}
;     #pragma oss task in(s.x, s.x, s.x)
;     {}
;     #pragma oss task in(n, n, n)
;     {}
; }

%struct.S = type { i32 }
%struct._depend_unpack_t = type { [10 x [20 x i32]]*, i64, i64, i64, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { [10 x [20 x i32]]*, i64, i64, i64, i64, i64, i64 }
%struct._depend_unpack_t.1 = type { [10 x [20 x i32]]*, i64, i64, i64, i64, i64, i64 }
%struct._depend_unpack_t.2 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.3 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.4 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.5 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.6 = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.7 = type { i32*, i64, i64, i64 }

@array = common dso_local global [10 x [20 x i32]] zeroinitializer, align 16
@s = common dso_local global %struct.S zeroinitializer, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %n = alloca i32, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, %struct._depend_unpack_t ([10 x [20 x i32]]*)* @compute_dep, [10 x [20 x i32]]* @array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, %struct._depend_unpack_t.0 ([10 x [20 x i32]]*)* @compute_dep.1, [10 x [20 x i32]]* @array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, %struct._depend_unpack_t.1 ([10 x [20 x i32]]*)* @compute_dep.2, [10 x [20 x i32]]* @array) ], !dbg !8
  call void @llvm.directive.region.exit(token %0), !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* @s), "QUAL.OSS.DEP.IN"(%struct.S* @s, %struct._depend_unpack_t.2 (%struct.S*)* @compute_dep.3, %struct.S* @s), "QUAL.OSS.DEP.IN"(%struct.S* @s, %struct._depend_unpack_t.3 (%struct.S*)* @compute_dep.4, %struct.S* @s), "QUAL.OSS.DEP.IN"(%struct.S* @s, %struct._depend_unpack_t.4 (%struct.S*)* @compute_dep.5, %struct.S* @s) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %n), "QUAL.OSS.DEP.IN"(i32* %n, %struct._depend_unpack_t.5 (i32*)* @compute_dep.6, i32* %n), "QUAL.OSS.DEP.IN"(i32* %n, %struct._depend_unpack_t.6 (i32*)* @compute_dep.7, i32* %n), "QUAL.OSS.DEP.IN"(i32* %n, %struct._depend_unpack_t.7 (i32*)* @compute_dep.8, i32* %n) ], !dbg !12
  call void @llvm.directive.region.exit(token %2), !dbg !13
  ret i32 0, !dbg !14
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep([10 x [20 x i32]]* %array) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store [10 x [20 x i32]]* %array, [10 x [20 x i32]]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 80, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 80, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 4
  store i64 10, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 6
  store i64 10, i64* %6, align 8
  %7 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %7
}

define internal %struct._depend_unpack_t.0 @compute_dep.1([10 x [20 x i32]]* %array) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store [10 x [20 x i32]]* %array, [10 x [20 x i32]]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 80, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 80, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 4
  store i64 10, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 6
  store i64 10, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %7
}

define internal %struct._depend_unpack_t.1 @compute_dep.2([10 x [20 x i32]]* %array) {
entry:
  %return.val = alloca %struct._depend_unpack_t.1, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 0
  store [10 x [20 x i32]]* %array, [10 x [20 x i32]]** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 1
  store i64 80, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 3
  store i64 80, i64* %3, align 8
  %4 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 4
  store i64 10, i64* %4, align 8
  %5 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 5
  store i64 0, i64* %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, i32 0, i32 6
  store i64 10, i64* %6, align 8
  %7 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %return.val, align 8
  ret %struct._depend_unpack_t.1 %7
}

define internal %struct._depend_unpack_t.2 @compute_dep.3(%struct.S* %s) {
entry:
  %return.val = alloca %struct._depend_unpack_t.2, align 8
  %x = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %0 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 0
  store i32* %x, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %return.val, align 8
  ret %struct._depend_unpack_t.2 %4
}

define internal %struct._depend_unpack_t.3 @compute_dep.4(%struct.S* %s) {
entry:
  %return.val = alloca %struct._depend_unpack_t.3, align 8
  %x = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %0 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 0
  store i32* %x, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %return.val, align 8
  ret %struct._depend_unpack_t.3 %4
}

define internal %struct._depend_unpack_t.4 @compute_dep.5(%struct.S* %s) {
entry:
  %return.val = alloca %struct._depend_unpack_t.4, align 8
  %x = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %0 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %return.val, i32 0, i32 0
  store i32* %x, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %return.val, align 8
  ret %struct._depend_unpack_t.4 %4
}

define internal %struct._depend_unpack_t.5 @compute_dep.6(i32* %n) {
entry:
  %return.val = alloca %struct._depend_unpack_t.5, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 0
  store i32* %n, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %return.val, align 8
  ret %struct._depend_unpack_t.5 %4
}

define internal %struct._depend_unpack_t.6 @compute_dep.7(i32* %n) {
entry:
  %return.val = alloca %struct._depend_unpack_t.6, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %return.val, i32 0, i32 0
  store i32* %n, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %return.val, align 8
  ret %struct._depend_unpack_t.6 %4
}

define internal %struct._depend_unpack_t.7 @compute_dep.8(i32* %n) {
entry:
  %return.val = alloca %struct._depend_unpack_t.7, align 8
  %0 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 0
  store i32* %n, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 1
  store i64 4, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 2
  store i64 0, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, i32 0, i32 3
  store i64 4, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %return.val, align 8
  ret %struct._depend_unpack_t.7 %4
}

; Check we access properly a local variable, a global variable, a constant expression

; CHECK: define internal void @nanos6_unpacked_deps_main0([10 x [20 x i32]]* %array, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t @compute_dep([10 x [20 x i32]]* %array)
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t @compute_dep([10 x [20 x i32]]* %array)
; CHECK-NEXT:   %2 = extractvalue %struct._depend_unpack_t %0, 0
; CHECK-NEXT:   %3 = bitcast [10 x [20 x i32]]* %2 to i8*
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t %0, 1
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t %0, 2
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t %1, 3
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t %0, 4
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t %0, 5
; CHECK-NEXT:   %9 = extractvalue %struct._depend_unpack_t %1, 6
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9)
; CHECK-NEXT:   %10 = call %struct._depend_unpack_t.0 @compute_dep.1([10 x [20 x i32]]* %array)
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t.0 @compute_dep.1([10 x [20 x i32]]* %array)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.0 %10, 0
; CHECK-NEXT:   %13 = bitcast [10 x [20 x i32]]* %12 to i8*
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t.0 %10, 1
; CHECK-NEXT:   %15 = extractvalue %struct._depend_unpack_t.0 %10, 2
; CHECK-NEXT:   %16 = extractvalue %struct._depend_unpack_t.0 %11, 3
; CHECK-NEXT:   %17 = extractvalue %struct._depend_unpack_t.0 %10, 4
; CHECK-NEXT:   %18 = extractvalue %struct._depend_unpack_t.0 %10, 5
; CHECK-NEXT:   %19 = extractvalue %struct._depend_unpack_t.0 %11, 6
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19)
; CHECK-NEXT:   %20 = call %struct._depend_unpack_t.1 @compute_dep.2([10 x [20 x i32]]* %array)
; CHECK-NEXT:   %21 = call %struct._depend_unpack_t.1 @compute_dep.2([10 x [20 x i32]]* %array)
; CHECK-NEXT:   %22 = extractvalue %struct._depend_unpack_t.1 %20, 0
; CHECK-NEXT:   %23 = bitcast [10 x [20 x i32]]* %22 to i8*
; CHECK-NEXT:   %24 = extractvalue %struct._depend_unpack_t.1 %20, 1
; CHECK-NEXT:   %25 = extractvalue %struct._depend_unpack_t.1 %20, 2
; CHECK-NEXT:   %26 = extractvalue %struct._depend_unpack_t.1 %21, 3
; CHECK-NEXT:   %27 = extractvalue %struct._depend_unpack_t.1 %20, 4
; CHECK-NEXT:   %28 = extractvalue %struct._depend_unpack_t.1 %20, 5
; CHECK-NEXT:   %29 = extractvalue %struct._depend_unpack_t.1 %21, 6
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main1(%struct.S* %s, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t.2 @compute_dep.3(%struct.S* %s)
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t.2 @compute_dep.3(%struct.S* %s)
; CHECK-NEXT:   %2 = extractvalue %struct._depend_unpack_t.2 %0, 0
; CHECK-NEXT:   %3 = bitcast i32* %2 to i8*
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t.2 %0, 1
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t.2 %0, 2
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t.2 %1, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %3, i64 %4, i64 %5, i64 %6)
; CHECK-NEXT:   %7 = call %struct._depend_unpack_t.3 @compute_dep.4(%struct.S* %s)
; CHECK-NEXT:   %8 = call %struct._depend_unpack_t.3 @compute_dep.4(%struct.S* %s)
; CHECK-NEXT:   %9 = extractvalue %struct._depend_unpack_t.3 %7, 0
; CHECK-NEXT:   %10 = bitcast i32* %9 to i8*
; CHECK-NEXT:   %11 = extractvalue %struct._depend_unpack_t.3 %7, 1
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.3 %7, 2
; CHECK-NEXT:   %13 = extractvalue %struct._depend_unpack_t.3 %8, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %10, i64 %11, i64 %12, i64 %13)
; CHECK-NEXT:   %14 = call %struct._depend_unpack_t.4 @compute_dep.5(%struct.S* %s)
; CHECK-NEXT:   %15 = call %struct._depend_unpack_t.4 @compute_dep.5(%struct.S* %s)
; CHECK-NEXT:   %16 = extractvalue %struct._depend_unpack_t.4 %14, 0
; CHECK-NEXT:   %17 = bitcast i32* %16 to i8*
; CHECK-NEXT:   %18 = extractvalue %struct._depend_unpack_t.4 %14, 1
; CHECK-NEXT:   %19 = extractvalue %struct._depend_unpack_t.4 %14, 2
; CHECK-NEXT:   %20 = extractvalue %struct._depend_unpack_t.4 %15, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %17, i64 %18, i64 %19, i64 %20)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main2(i32* %n, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call %struct._depend_unpack_t.5 @compute_dep.6(i32* %n)
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t.5 @compute_dep.6(i32* %n)
; CHECK-NEXT:   %2 = extractvalue %struct._depend_unpack_t.5 %0, 0
; CHECK-NEXT:   %3 = bitcast i32* %2 to i8*
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t.5 %0, 1
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t.5 %0, 2
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t.5 %1, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %3, i64 %4, i64 %5, i64 %6)
; CHECK-NEXT:   %7 = call %struct._depend_unpack_t.6 @compute_dep.7(i32* %n)
; CHECK-NEXT:   %8 = call %struct._depend_unpack_t.6 @compute_dep.7(i32* %n)
; CHECK-NEXT:   %9 = extractvalue %struct._depend_unpack_t.6 %7, 0
; CHECK-NEXT:   %10 = bitcast i32* %9 to i8*
; CHECK-NEXT:   %11 = extractvalue %struct._depend_unpack_t.6 %7, 1
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.6 %7, 2
; CHECK-NEXT:   %13 = extractvalue %struct._depend_unpack_t.6 %8, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %10, i64 %11, i64 %12, i64 %13)
; CHECK-NEXT:   %14 = call %struct._depend_unpack_t.7 @compute_dep.8(i32* %n)
; CHECK-NEXT:   %15 = call %struct._depend_unpack_t.7 @compute_dep.8(i32* %n)
; CHECK-NEXT:   %16 = extractvalue %struct._depend_unpack_t.7 %14, 0
; CHECK-NEXT:   %17 = bitcast i32* %16 to i8*
; CHECK-NEXT:   %18 = extractvalue %struct._depend_unpack_t.7 %14, 1
; CHECK-NEXT:   %19 = extractvalue %struct._depend_unpack_t.7 %14, 2
; CHECK-NEXT:   %20 = extractvalue %struct._depend_unpack_t.7 %15, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %17, i64 %18, i64 %19, i64 %20)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_depend_proper_rtcall.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !7, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 8, column: 13, scope: !6)
!9 = !DILocation(line: 9, column: 6, scope: !6)
!10 = !DILocation(line: 10, column: 13, scope: !6)
!11 = !DILocation(line: 11, column: 6, scope: !6)
!12 = !DILocation(line: 12, column: 13, scope: !6)
!13 = !DILocation(line: 13, column: 6, scope: !6)
!14 = !DILocation(line: 14, column: 1, scope: !6)
