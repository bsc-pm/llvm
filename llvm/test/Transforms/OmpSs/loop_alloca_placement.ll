; RUN: opt %s -ompss-2 -S | FileCheck %s

; ModuleID = 'loop_alloca_placement.ll'
source_filename = "loop_alloca_placement.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Loop transformation should place constant allocas
; outside the loop. Otherwise the stack will grow
; every iteration and potentially raise a stack
; smashing in the execution.
; VLAs do not need this because they use
; llvm.stacksave/stackrestore

; int main() {
;     #pragma oss taskloop
;     for (int i = 0; i < 10; ++i) {
;         int array[10];
;         #pragma oss task
;         {
;             if (i > 78) {
;                 int array1[10];
;             }
;             array[0]++;
;         }
;     }
; }

; CHECK-LABEL: @main
; CHECK: final.then9:                                      ; preds = %final.cond8
; CHECK-NEXT:   %array1.clone1 = alloca [10 x i32], align 16, !dbg !11
; CHECK-NEXT:   %array.clone = alloca [10 x i32], align 16, !dbg !11

; CHECK-LABEL: @nanos6_unpacked_task_region_main1
; CHECK: 2:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %3 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %4 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub = trunc i64 %4 to i32
; CHECK-NEXT:   %array = alloca [10 x i32], align 16

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !11
  %array = alloca [10 x i32], align 16
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]* %array) ], !dbg !12
  %array1 = alloca [10 x i32], align 16
  %2 = load i32, i32* %i, align 4, !dbg !13
  %cmp = icmp sgt i32 %2, 78, !dbg !14
  br i1 %cmp, label %if.then, label %if.end, !dbg !13

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !15

if.end:                                           ; preds = %if.then, %entry
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %array, i64 0, i64 0, !dbg !16
  %3 = load i32, i32* %arrayidx, align 16, !dbg !17
  %inc = add nsw i32 %3, 1, !dbg !17
  store i32 %inc, i32* %arrayidx, align 16, !dbg !17
  call void @llvm.directive.region.exit(token %1), !dbg !18
  call void @llvm.directive.region.exit(token %0), !dbg !19
  %4 = load i32, i32* %retval, align 4, !dbg !20
  ret i32 %4, !dbg !20
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb() #2 !dbg !21 {
entry:
  ret i32 0, !dbg !22
}

define internal i32 @compute_ub() #2 !dbg !23 {
entry:
  ret i32 10, !dbg !24
}

define internal i32 @compute_step() #2 !dbg !25 {
entry:
  ret i32 1, !dbg !26
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }
attributes #2 = { "min-legal-vector-width"="0" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (git@bscpm03.bsc.es:llvm-ompss/llvm-mono.git 1a41f701f068bdfc3c8b069bb65749a03e9db4b6)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "/home/rpenacob/llvm-mono-tmp/build/bin")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 15.0.0 (git@bscpm03.bsc.es:llvm-ompss/llvm-mono.git 1a41f701f068bdfc3c8b069bb65749a03e9db4b6)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 3, column: 14, scope: !7)
!11 = !DILocation(line: 3, column: 10, scope: !7)
!12 = !DILocation(line: 5, column: 17, scope: !7)
!13 = !DILocation(line: 7, column: 17, scope: !7)
!14 = !DILocation(line: 7, column: 19, scope: !7)
!15 = !DILocation(line: 9, column: 13, scope: !7)
!16 = !DILocation(line: 10, column: 13, scope: !7)
!17 = !DILocation(line: 10, column: 21, scope: !7)
!18 = !DILocation(line: 11, column: 9, scope: !7)
!19 = !DILocation(line: 12, column: 5, scope: !7)
!20 = !DILocation(line: 13, column: 1, scope: !7)
!21 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !9)
!22 = !DILocation(line: 3, column: 18, scope: !21)
!23 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !9)
!24 = !DILocation(line: 3, column: 25, scope: !23)
!25 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !9)
!26 = !DILocation(line: 3, column: 29, scope: !25)
