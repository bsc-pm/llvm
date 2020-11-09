; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'taskloop_multideps.ll'
source_filename = "taskloop_multideps.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test checks we use nanos6 lower bound to build
; multidep loop and call to register dep

; int v[10];
; int main() {
;     #pragma oss taskloop out( { v[i], i=0;j } )
;     for (int j = 0; j < 10; ++j) { }
; }

%struct._depend_unpack_t = type { i32, i32, i32, i32 }
%struct._depend_unpack_t.0 = type { i32*, i64, i64, i64 }

@v = global [10 x i32] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !6 {
entry:
  %j = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %j, align 4, !dbg !9
  store i32 0, i32* %i, align 4, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([10 x i32]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.LOOP.IND.VAR"(i32* %j), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1), "QUAL.OSS.MULTIDEP.RANGE.OUT"(i32* %i, %struct._depend_unpack_t (i32*, i32*)* @compute_dep, i32* %i, i32* %j, [10 x i32]* @v, [16 x i8] c"{ v[i], i=0;j }\00", %struct._depend_unpack_t.0 (i32*, i32*, [10 x i32]*)* @compute_dep.1, i32* %i, i32* %j, [10 x i32]* @v) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !9
  ret i32 0, !dbg !11
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 4
  %0 = load i32, i32* %i, align 4, !dbg !10
  %1 = load i32, i32* %j, align 4, !dbg !10
  %2 = add i32 0, %1
  %3 = add i32 %2, -1
  %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store i32 0, i32* %4, align 4
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i32 %0, i32* %5, align 4
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i32 %3, i32* %6, align 4
  %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i32 1, i32* %7, align 4
  %8 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 4
  ret %struct._depend_unpack_t %8
}

define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %i, i32* %j, [10 x i32]* %v) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %0 = load i32, i32* %i, align 4, !dbg !10
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %v, i64 0, i64 0, !dbg !10
  %3 = mul i64 %1, 4
  %4 = mul i64 %2, 4
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store i32* %arraydecay, i32** %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 40, i64* %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 %3, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 %4, i64* %8, align 8
  %9 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %9
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([10 x i32]* %v, i32* %i, i32* %j, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %3 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %4 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %5 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub = sub i32 %5, 1
; CHECK-NEXT:   %j.lb = alloca i32, align 4
; CHECK-NEXT:   %j.ub = alloca i32, align 4
; CHECK-NEXT:   %6 = mul i32 1, %lb
; CHECK-NEXT:   %7 = add i32 %6, 0
; CHECK-NEXT:   store i32 %7, i32* %j.lb, align 4
; CHECK-NEXT:   %8 = mul i32 1, %ub
; CHECK-NEXT:   %9 = add i32 %8, 0
; CHECK-NEXT:   store i32 %9, i32* %j.ub, align 4
; CHECK-NEXT:   %i.remap = alloca i32, align 4
; CHECK-NEXT:   br label %10
; CHECK: 10:                                               ; preds = %entry
; CHECK-NEXT:   store i32 0, i32* %i, align 4
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j.lb)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t %11, 0
; CHECK-NEXT:   store i32 %12, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.cond:                                         ; preds = %for.incr, %10
; CHECK-NEXT:   %13 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j.lb)
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t %13, 2
; CHECK-NEXT:   %15 = load i32, i32* %i, align 4
; CHECK-NEXT:   %16 = icmp sle i32 %15, %14
; CHECK-NEXT:   br i1 %16, label %for.body, label %26
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %17 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j.lb)
; CHECK-NEXT:   %18 = extractvalue %struct._depend_unpack_t %17, 1
; CHECK-NEXT:   store i32 %18, i32* %i.remap, align 4
; CHECK-NEXT:   %19 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %i.remap, i32* %j.lb, [10 x i32]* %v)
; CHECK-NEXT:   %20 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %i.remap, i32* %j.ub, [10 x i32]* %v)
; CHECK-NEXT:   %21 = extractvalue %struct._depend_unpack_t.0 %19, 0
; CHECK-NEXT:   %22 = bitcast i32* %21 to i8*
; CHECK-NEXT:   %23 = extractvalue %struct._depend_unpack_t.0 %19, 1
; CHECK-NEXT:   %24 = extractvalue %struct._depend_unpack_t.0 %19, 2
; CHECK-NEXT:   %25 = extractvalue %struct._depend_unpack_t.0 %20, 3
; CHECK-NEXT:   call void @nanos6_register_region_write_depinfo1(i8* %handler, i32 0, i8* getelementptr inbounds ([16 x i8], [16 x i8]* @1, i32 0, i32 0), i8* %22, i64 %23, i64 %24, i64 %25)
; CHECK-NEXT:   br label %for.incr
; CHECK: 26:                                               ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK: for.incr:                                         ; preds = %for.body
; CHECK-NEXT:   %27 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j.lb)
; CHECK-NEXT:   %28 = extractvalue %struct._depend_unpack_t %27, 3
; CHECK-NEXT:   %29 = load i32, i32* %i, align 4
; CHECK-NEXT:   %30 = add i32 %29, %28
; CHECK-NEXT:   store i32 %30, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "taskloop_multideps.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, scope: !6)
!10 = !DILocation(line: 3, scope: !6)
!11 = !DILocation(line: 5, scope: !6)
