; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_loop_iter.ll'
source_filename = "task_loop_iter.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test checks we do copy the iterator instructions that lead to DSA instead
; of moving them

; Function Attrs: nounwind uwtable
define dso_local void @initialize(i64 %M, i64 %TS, double* noalias %A, double %value) !dbg !6 {
entry:
  %M.addr = alloca i64, align 8
  %TS.addr = alloca i64, align 8
  %A.addr = alloca double*, align 8
  %value.addr = alloca double, align 8
  %j = alloca i64, align 8
  store i64 %M, i64* %M.addr, align 8, !tbaa !8
  store i64 %TS, i64* %TS.addr, align 8, !tbaa !8
  store double* %A, double** %A.addr, align 8, !tbaa !12
  store double %value, double* %value.addr, align 8, !tbaa !14
  %0 = load i64, i64* %M.addr, align 8, !dbg !16, !tbaa !8
  %1 = load i64, i64* %TS.addr, align 8, !dbg !17, !tbaa !8
  %div = sdiv i64 %0, %1, !dbg !18
  %2 = bitcast i64* %j to i8*, !dbg !19
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %2), !dbg !19
  store i64 0, i64* %j, align 8, !dbg !20, !tbaa !8
  br label %for.cond, !dbg !19

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load i64, i64* %j, align 8, !dbg !21, !tbaa !8
  %cmp = icmp slt i64 %3, 10, !dbg !22
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !23

for.cond.cleanup:                                 ; preds = %for.cond
  %4 = bitcast i64* %j to i8*, !dbg !23
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4), !dbg !23
  br label %for.end

for.body:                                         ; preds = %for.cond
  %5 = load i64, i64* %j, align 8, !dbg !24, !tbaa !8
  %6 = add i64 %5, 1, !dbg !25
  %7 = load double*, double** %A.addr, align 8, !dbg !26, !tbaa !12
  %8 = mul i64 %div, 8, !dbg !25
  %9 = mul i64 %5, 8, !dbg !25
  %10 = mul i64 %6, 8, !dbg !25
  %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(double** %A.addr), "QUAL.OSS.FIRSTPRIVATE"(i64* %j), "QUAL.OSS.CAPTURED"(i64 %div), "QUAL.OSS.DEP.OUT"(double* %7, i64 %8, i64 %9, i64 %10, i64 1, i64 0, i64 1) ], !dbg !25
  %12 = load double*, double** %A.addr, align 8, !dbg !27, !tbaa !12
  %13 = mul nsw i64 0, %div, !dbg !27
  %arrayidx = getelementptr inbounds double, double* %12, i64 %13, !dbg !27
  %14 = load i64, i64* %j, align 8, !dbg !28, !tbaa !8
  %arrayidx1 = getelementptr inbounds double, double* %arrayidx, i64 %14, !dbg !27
  store double 1.000000e+00, double* %arrayidx1, align 8, !dbg !29, !tbaa !14
  call void @llvm.directive.region.exit(token %11), !dbg !30
  br label %for.inc, !dbg !31

for.inc:                                          ; preds = %for.body
  %15 = load i64, i64* %j, align 8, !dbg !32, !tbaa !8
  %inc = add nsw i64 %15, 1, !dbg !32
  store i64 %inc, i64* %j, align 8, !dbg !32, !tbaa !8
  br label %for.cond, !dbg !23, !llvm.loop !33

for.end:                                          ; preds = %for.cond.cleanup
  ret void, !dbg !34
}

; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %5 = load i64, i64* %j, align 8, !dbg !24, !tbaa !8

; CHECK: %16 = bitcast double** %gep_A.addr to i8*, !dbg !25
; CHECK-NEXT: %17 = bitcast double** %A.addr to i8*, !dbg !25
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %16, i8* align 8 %17, i64 8, i1 false), !dbg !25
; CHECK-NEXT: %gep_j = getelementptr %nanos6_task_args_initialize0, %nanos6_task_args_initialize0* %14, i32 0, i32 1, !dbg !25
; CHECK-NEXT: %18 = bitcast i64* %gep_j to i8*, !dbg !25
; CHECK-NEXT: %19 = bitcast i64* %j to i8*, !dbg !25
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %18, i8* align 8 %19, i64 8, i1 false), !dbg !25
; CHECK-NEXT: %capt_gep_div = getelementptr %nanos6_task_args_initialize0, %nanos6_task_args_initialize0* %14, i32 0, i32 2, !dbg !25
; CHECK-NEXT: store i64 %div, i64* %capt_gep_div, !dbg !25

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_loop_iter.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "initialize", scope: !1, file: !1, line: 1, type: !7, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !9, i64 0}
!9 = !{!"long", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"any pointer", !10, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"double", !10, i64 0}
!16 = !DILocation(line: 2, column: 26, scope: !6)
!17 = !DILocation(line: 2, column: 30, scope: !6)
!18 = !DILocation(line: 2, column: 28, scope: !6)
!19 = !DILocation(line: 3, column: 10, scope: !6)
!20 = !DILocation(line: 3, column: 15, scope: !6)
!21 = !DILocation(line: 3, column: 22, scope: !6)
!22 = !DILocation(line: 3, column: 24, scope: !6)
!23 = !DILocation(line: 3, column: 5, scope: !6)
!24 = !DILocation(line: 4, column: 35, scope: !6)
!25 = !DILocation(line: 4, column: 17, scope: !6)
!26 = !DILocation(line: 4, column: 30, scope: !6)
!27 = !DILocation(line: 5, column: 11, scope: !6)
!28 = !DILocation(line: 5, column: 16, scope: !6)
!29 = !DILocation(line: 5, column: 19, scope: !6)
!30 = !DILocation(line: 5, column: 24, scope: !6)
!31 = !DILocation(line: 6, column: 5, scope: !6)
!32 = !DILocation(line: 3, column: 31, scope: !6)
!33 = distinct !{!33, !23, !31}
!34 = !DILocation(line: 7, column: 1, scope: !6)

