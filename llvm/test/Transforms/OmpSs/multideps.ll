; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'multideps.ll'
source_filename = "multideps.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int v[10][10];
; int main() {
;     #pragma oss task out( { v[i][j], i = {0, 1, 2}, j = i:10-1 } )
;     {}
; }

%struct._depend_unpack_t = type { i32, i32, i32, i32, i32, i32, i32, i32 }
%struct._depend_unpack_t.0 = type { [10 x i32]*, i64, i64, i64, i64, i64, i64 }

@v = dso_local global [10 x [10 x i32]] zeroinitializer, align 16
@__const.main.discrete.array = private unnamed_addr constant [3 x i32] [i32 0, i32 1, i32 2], align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %0 = load i32, i32* %i, align 4, !dbg !8
  store i32 %0, i32* %j, align 4, !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [10 x i32]]* @v), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.MULTIDEP.RANGE.OUT"(i32* %i, i32* %j, %struct._depend_unpack_t (i32*, i32*)* @compute_dep, i32* %i, i32* %j, [10 x [10 x i32]]* @v, %struct._depend_unpack_t.0 (i32*, i32*, [10 x [10 x i32]]*)* @compute_dep.1, i32* %i, i32* %j, [10 x [10 x i32]]* @v) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  ret i32 0, !dbg !12
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 4
  %discrete.array = alloca [3 x i32], align 4
  %0 = bitcast [3 x i32]* %discrete.array to i8*, !dbg !13
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 bitcast ([3 x i32]* @__const.main.discrete.array to i8*), i64 12, i1 false), !dbg !13
  %1 = load i32, i32* %i, align 4, !dbg !14
  %discreteidx = getelementptr [3 x i32], [3 x i32]* %discrete.array, i32 0, i32 %1
  %2 = load i32, i32* %discreteidx, align 8
  %3 = load i32, i32* %i, align 4, !dbg !8
  %4 = load i32, i32* %j, align 4, !dbg !9
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store i32 0, i32* %5, align 4
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i32 %2, i32* %6, align 4
  %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i32 2, i32* %7, align 4
  %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i32 1, i32* %8, align 4
  %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 4
  store i32 %3, i32* %9, align 4
  %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 5
  store i32 %4, i32* %10, align 4
  %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 6
  store i32 9, i32* %11, align 4
  %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 7
  store i32 1, i32* %12, align 4
  %13 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 4
  ret %struct._depend_unpack_t %13
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %i, i32* %j, [10 x [10 x i32]]* %v) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %0 = load i32, i32* %j, align 4, !dbg !15
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %3 = load i32, i32* %i, align 4, !dbg !16
  %4 = sext i32 %3 to i64
  %5 = add i64 %4, 1
  %arraydecay = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* %v, i64 0, i64 0, !dbg !17
  %6 = mul i64 %1, 4
  %7 = mul i64 %2, 4
  %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store [10 x i32]* %arraydecay, [10 x i32]** %8, align 8
  %9 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 40, i64* %9, align 8
  %10 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 %6, i64* %10, align 8
  %11 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 %7, i64* %11, align 8
  %12 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 4
  store i64 10, i64* %12, align 8
  %13 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 5
  store i64 %4, i64* %13, align 8
  %14 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 6
  store i64 %5, i64* %14, align 8
  %15 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %15
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([10 x [10 x i32]]* %v, i32* %i, i32* %j, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %entry
; CHECK-NEXT:   %i.remap = alloca i32, align 4
; CHECK-NEXT:   %1 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %2 = extractvalue %struct._depend_unpack_t %1, 0
; CHECK-NEXT:   store i32 %2, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.cond:                                         ; preds = %for.incr, %0
; CHECK-NEXT:   %3 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t %3, 2
; CHECK-NEXT:   %5 = load i32, i32* %i, align 4
; CHECK-NEXT:   %6 = icmp sle i32 %5, %4
; CHECK-NEXT:   br i1 %6, label %for.body, label %27
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %7 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t %7, 1
; CHECK-NEXT:   store i32 %8, i32* %i.remap, align 4
; CHECK-NEXT:   %j.remap = alloca i32, align 4
; CHECK-NEXT:   %9 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %10 = extractvalue %struct._depend_unpack_t %9, 4
; CHECK-NEXT:   store i32 %10, i32* %j, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK: for.cond1:                                        ; preds = %for.incr3, %for.body
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t %11, 6
; CHECK-NEXT:   %13 = load i32, i32* %j, align 4
; CHECK-NEXT:   %14 = icmp sle i32 %13, %12
; CHECK-NEXT:   br i1 %14, label %for.body2, label %for.incr
; CHECK: for.body2:                                        ; preds = %for.cond1
; CHECK-NEXT:   %15 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %16 = extractvalue %struct._depend_unpack_t %15, 5
; CHECK-NEXT:   store i32 %16, i32* %j.remap, align 4
; CHECK-NEXT:   %17 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %i.remap, i32* %j.remap, [10 x [10 x i32]]* %v)
; CHECK-NEXT:   %18 = call %struct._depend_unpack_t.0 @compute_dep.1(i32* %i.remap, i32* %j.remap, [10 x [10 x i32]]* %v)
; CHECK-NEXT:   %19 = extractvalue %struct._depend_unpack_t.0 %17, 0
; CHECK-NEXT:   %20 = bitcast [10 x i32]* %19 to i8*
; CHECK-NEXT:   %21 = extractvalue %struct._depend_unpack_t.0 %17, 1
; CHECK-NEXT:   %22 = extractvalue %struct._depend_unpack_t.0 %17, 2
; CHECK-NEXT:   %23 = extractvalue %struct._depend_unpack_t.0 %18, 3
; CHECK-NEXT:   %24 = extractvalue %struct._depend_unpack_t.0 %17, 4
; CHECK-NEXT:   %25 = extractvalue %struct._depend_unpack_t.0 %17, 5
; CHECK-NEXT:   %26 = extractvalue %struct._depend_unpack_t.0 %18, 6
; CHECK-NEXT:   call void @nanos6_register_region_write_depinfo2(i8* %handler, i32 0, i8* null, i8* %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26)
; CHECK-NEXT:   br label %for.incr3
; CHECK: 27:                                               ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK: for.incr:                                         ; preds = %for.cond1
; CHECK-NEXT:   %28 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %29 = extractvalue %struct._depend_unpack_t %28, 3
; CHECK-NEXT:   %30 = load i32, i32* %i, align 4
; CHECK-NEXT:   %31 = add i32 %30, %29
; CHECK-NEXT:   store i32 %31, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.incr3:                                        ; preds = %for.body2
; CHECK-NEXT:   %32 = call %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j)
; CHECK-NEXT:   %33 = extractvalue %struct._depend_unpack_t %32, 7
; CHECK-NEXT:   %34 = load i32, i32* %j, align 4
; CHECK-NEXT:   %35 = add i32 %34, %33
; CHECK-NEXT:   store i32 %35, i32* %j, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { argmemonly nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "multideps.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 57, scope: !6)
!9 = !DILocation(line: 3, column: 53, scope: !6)
!10 = !DILocation(line: 3, column: 13, scope: !6)
!11 = !DILocation(line: 4, column: 6, scope: !6)
!12 = !DILocation(line: 5, column: 1, scope: !6)
!13 = !DILocation(line: 0, scope: !6)
!14 = !DILocation(line: 3, column: 38, scope: !6)
!15 = !DILocation(line: 3, column: 34, scope: !6)
!16 = !DILocation(line: 3, column: 31, scope: !6)
!17 = !DILocation(line: 3, column: 29, scope: !6)
