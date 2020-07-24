; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop_clauses.ll'
source_filename = "loop_clauses.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo() {
;     #pragma oss task for chunksize(777)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss taskloop for chunksize(777)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss taskloop grainsize(777)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss taskloop for grainsize(777)
;     for (int i = 0; i < 10; ++i) {}
; }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z3foov() #0 !dbg !6 {
entry:
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %i3 = alloca i32, align 4
  store i32 0, i32* %i, align 4, !dbg !8
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.CHUNKSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 777) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !10
  store i32 0, i32* %i1, align 4, !dbg !11
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.CHUNKSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 777) ], !dbg !12
  call void @llvm.directive.region.exit(token %1), !dbg !13
  store i32 0, i32* %i2, align 4, !dbg !14
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 777) ], !dbg !15
  call void @llvm.directive.region.exit(token %2), !dbg !16
  store i32 0, i32* %i3, align 4, !dbg !17
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 777) ], !dbg !18
  call void @llvm.directive.region.exit(token %3), !dbg !19
  ret void, !dbg !20
}

; CHECK: call void @nanos6_register_loop_bounds(i8* %5, i64 0, i64 10, i64 0, i64 777)
; CHECK: call void @nanos6_register_loop_bounds(i8* %11, i64 0, i64 10, i64 0, i64 777)
; CHECK: call void @nanos6_register_loop_bounds(i8* %17, i64 0, i64 10, i64 777, i64 0)
; CHECK: call void @nanos6_register_loop_bounds(i8* %23, i64 0, i64 10, i64 777, i64 0)

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "loop_clauses.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 14, scope: !6)
!9 = !DILocation(line: 3, column: 10, scope: !6)
!10 = !DILocation(line: 3, column: 35, scope: !6)
!11 = !DILocation(line: 5, column: 14, scope: !6)
!12 = !DILocation(line: 5, column: 10, scope: !6)
!13 = !DILocation(line: 5, column: 35, scope: !6)
!14 = !DILocation(line: 7, column: 14, scope: !6)
!15 = !DILocation(line: 7, column: 10, scope: !6)
!16 = !DILocation(line: 7, column: 35, scope: !6)
!17 = !DILocation(line: 9, column: 14, scope: !6)
!18 = !DILocation(line: 9, column: 10, scope: !6)
!19 = !DILocation(line: 9, column: 35, scope: !6)
!20 = !DILocation(line: 10, column: 1, scope: !6)
