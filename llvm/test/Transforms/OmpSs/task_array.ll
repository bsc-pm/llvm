; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_array.ll'
source_filename = "task_array.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32 }

define dso_local void @_Z9pod_arrayv() !dbg !6 {
entry:
  %array = alloca [10 x i32], align 16
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"([10 x i32]* %array) ], !dbg !8
  call void @llvm.directive.region.exit(token %0), !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]* %array) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  ret void, !dbg !12

; CHECK: codeRepl:                                         ; preds = %entry
; CHECK-NEXT:   %0 = alloca %nanos6_task_args__Z9pod_arrayv0*, !dbg !8
; CHECK-NEXT:   %1 = bitcast %nanos6_task_args__Z9pod_arrayv0** %0 to i8**, !dbg !8
; CHECK-NEXT:   %2 = alloca i8*, !dbg !8
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z9pod_arrayv0, %nanos6_task_invocation_info_t* @task_invocation_info__Z9pod_arrayv0, i64 48, i8** %1, i8** %2, i64 0, i64 0), !dbg !8
; CHECK-NEXT:   %3 = load %nanos6_task_args__Z9pod_arrayv0*, %nanos6_task_args__Z9pod_arrayv0** %0, !dbg !8
; CHECK-NEXT:   %4 = bitcast %nanos6_task_args__Z9pod_arrayv0* %3 to i8*, !dbg !8
; CHECK-NEXT:   %args_end = getelementptr i8, i8* %4, i64 48, !dbg !8
; CHECK-NEXT:   %5 = load i8*, i8** %2, !dbg !8
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %5), !dbg !8
; CHECK-NEXT:   br label %6, !dbg !8
; CHECK: 6:                                                ; preds = %codeRepl
; CHECK-NEXT:   br label %codeRepl1, !dbg !9
; CHECK: codeRepl1:                                        ; preds = %6
; CHECK-NEXT:   %7 = alloca %nanos6_task_args__Z9pod_arrayv1*, !dbg !9
; CHECK-NEXT:   %8 = bitcast %nanos6_task_args__Z9pod_arrayv1** %7 to i8**, !dbg !9
; CHECK-NEXT:   %9 = alloca i8*, !dbg !9
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z9pod_arrayv1, %nanos6_task_invocation_info_t* @task_invocation_info__Z9pod_arrayv1, i64 48, i8** %8, i8** %9, i64 0, i64 0), !dbg !9
; CHECK-NEXT:   %10 = load %nanos6_task_args__Z9pod_arrayv1*, %nanos6_task_args__Z9pod_arrayv1** %7, !dbg !9
; CHECK-NEXT:   %11 = bitcast %nanos6_task_args__Z9pod_arrayv1* %10 to i8*, !dbg !9
; CHECK-NEXT:   %args_end2 = getelementptr i8, i8* %11, i64 48, !dbg !9
; CHECK-NEXT:   %gep_array = getelementptr %nanos6_task_args__Z9pod_arrayv1, %nanos6_task_args__Z9pod_arrayv1* %10, i32 0, i32 0, !dbg !9
; CHECK-NEXT:   %12 = bitcast [10 x i32]* %gep_array to i8*, !dbg !9
; CHECK-NEXT:   %13 = bitcast [10 x i32]* %array to i8*, !dbg !9
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %12, i8* align 4 %13, i64 40, i1 false), !dbg !9
; CHECK-NEXT:   %14 = load i8*, i8** %9, !dbg !9
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %14), !dbg !9
; CHECK-NEXT:   br label %15, !dbg !9
; CHECK: 15:                                               ; preds = %codeRepl1
; CHECK-NEXT:   ret void, !dbg !10
}


define dso_local void @_Z13non_pod_arrayv() !dbg !13 {
entry:
  %array = alloca [10 x %struct.S], align 16
  %array.begin = getelementptr inbounds [10 x %struct.S], [10 x %struct.S]* %array, i32 0, i32 0, !dbg !14
  %arrayctor.end = getelementptr inbounds %struct.S, %struct.S* %array.begin, i64 10, !dbg !14
  br label %arrayctor.loop, !dbg !14

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi %struct.S* [ %array.begin, %entry ], [ %arrayctor.next, %arrayctor.loop ], !dbg !14
  call void @_ZN1SC1Ev(%struct.S* %arrayctor.cur), !dbg !14
  %arrayctor.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.cur, i64 1, !dbg !14
  %arrayctor.done = icmp eq %struct.S* %arrayctor.next, %arrayctor.end, !dbg !14
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !14

arrayctor.cont:                                   ; preds = %arrayctor.loop
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"([10 x %struct.S]* %array), "QUAL.OSS.INIT"([10 x %struct.S]* %array, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x %struct.S]* %array, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ], !dbg !15
  call void @llvm.directive.region.exit(token %0), !dbg !16
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x %struct.S]* %array), "QUAL.OSS.COPY"([10 x %struct.S]* %array, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_), "QUAL.OSS.DEINIT"([10 x %struct.S]* %array, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ], !dbg !17
  call void @llvm.directive.region.exit(token %1), !dbg !18
  ret void, !dbg !19

; CHECK: codeRepl:                                         ; preds = %arrayctor.cont
; CHECK-NEXT:   %0 = alloca %nanos6_task_args__Z13non_pod_arrayv0*, !dbg !13
; CHECK-NEXT:   %1 = bitcast %nanos6_task_args__Z13non_pod_arrayv0** %0 to i8**, !dbg !13
; CHECK-NEXT:   %2 = alloca i8*, !dbg !13
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z13non_pod_arrayv0, %nanos6_task_invocation_info_t* @task_invocation_info__Z13non_pod_arrayv0, i64 48, i8** %1, i8** %2, i64 0, i64 0), !dbg !13
; CHECK-NEXT:   %3 = load %nanos6_task_args__Z13non_pod_arrayv0*, %nanos6_task_args__Z13non_pod_arrayv0** %0, !dbg !13
; CHECK-NEXT:   %4 = bitcast %nanos6_task_args__Z13non_pod_arrayv0* %3 to i8*, !dbg !13
; CHECK-NEXT:   %args_end = getelementptr i8, i8* %4, i64 48, !dbg !13
; CHECK-NEXT:   %gep_array = getelementptr %nanos6_task_args__Z13non_pod_arrayv0, %nanos6_task_args__Z13non_pod_arrayv0* %3, i32 0, i32 0, !dbg !13
; CHECK-NEXT:   %5 = bitcast [10 x %struct.S]* %gep_array to %struct.S*, !dbg !13
; CHECK-NEXT:   call void @oss_ctor_ZN1SC1Ev(%struct.S* %5, i64 10), !dbg !13
; CHECK-NEXT:   %6 = load i8*, i8** %2, !dbg !13
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %6), !dbg !13
; CHECK-NEXT:   br label %7, !dbg !13
; CHECK: 7:                                                ; preds = %codeRepl
; CHECK-NEXT:   br label %codeRepl1, !dbg !14
; CHECK: codeRepl1:                                        ; preds = %7
; CHECK-NEXT:   %8 = alloca %nanos6_task_args__Z13non_pod_arrayv1*, !dbg !14
; CHECK-NEXT:   %9 = bitcast %nanos6_task_args__Z13non_pod_arrayv1** %8 to i8**, !dbg !14
; CHECK-NEXT:   %10 = alloca i8*, !dbg !14
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z13non_pod_arrayv1, %nanos6_task_invocation_info_t* @task_invocation_info__Z13non_pod_arrayv1, i64 48, i8** %9, i8** %10, i64 0, i64 0), !dbg !14
; CHECK-NEXT:   %11 = load %nanos6_task_args__Z13non_pod_arrayv1*, %nanos6_task_args__Z13non_pod_arrayv1** %8, !dbg !14
; CHECK-NEXT:   %12 = bitcast %nanos6_task_args__Z13non_pod_arrayv1* %11 to i8*, !dbg !14
; CHECK-NEXT:   %args_end2 = getelementptr i8, i8* %12, i64 48, !dbg !14
; CHECK-NEXT:   %gep_array3 = getelementptr %nanos6_task_args__Z13non_pod_arrayv1, %nanos6_task_args__Z13non_pod_arrayv1* %11, i32 0, i32 0, !dbg !14
; CHECK-NEXT:   %13 = bitcast [10 x %struct.S]* %gep_array3 to %struct.S*, !dbg !14
; CHECK-NEXT:   %14 = bitcast [10 x %struct.S]* %array to %struct.S*, !dbg !14
; CHECK-NEXT:   call void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %14, %struct.S* %13, i64 10), !dbg !14
; CHECK-NEXT:   %15 = load i8*, i8** %10, !dbg !14
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %15), !dbg !14
; CHECK-NEXT:   br label %16, !dbg !14
; CHECK: 16:                                               ; preds = %codeRepl1
; CHECK-NEXT:   ret void, !dbg !15
}

; CHECK: define internal void @nanos6_unpacked_task_region__Z13non_pod_arrayv0([10 x %struct.S]* %array, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK:   call void @oss_dtor_ZN1SD1Ev(%struct.S* %1, i64 10)
; CHECK: }

; CHECK: define internal void @nanos6_unpacked_task_region__Z13non_pod_arrayv1([10 x %struct.S]* %array, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK:   call void @oss_dtor_ZN1SD1Ev(%struct.S* %1, i64 10)
; CHECK: }

declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare dso_local void @_ZN1SC1Ev(%struct.S*)
declare dso_local void @oss_ctor_ZN1SC1Ev(%struct.S* %0, i64 %1)
declare dso_local void @_ZN1SD1Ev(%struct.S*)
declare dso_local void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1)
declare dso_local void @_ZN1SC2ERKS_(%struct.S* %this, %struct.S* dereferenceable(4) %0)
declare dso_local void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %0, %struct.S* %1, i64 %2)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_array.ll", directory: "llvm/test/Transforms/OmpSs")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "pod_array", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 13, scope: !6)
!9 = !DILocation(line: 4, column: 6, scope: !6)
!10 = !DILocation(line: 5, column: 13, scope: !6)
!11 = !DILocation(line: 6, column: 6, scope: !6)
!12 = !DILocation(line: 7, column: 1, scope: !6)
!13 = distinct !DISubprogram(name: "non_pod_array", scope: !1, file: !1, line: 14, type: !7, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 15, column: 7, scope: !13)
!15 = !DILocation(line: 16, column: 13, scope: !13)
!16 = !DILocation(line: 17, column: 6, scope: !13)
!17 = !DILocation(line: 18, column: 13, scope: !13)
!18 = !DILocation(line: 19, column: 6, scope: !13)
!19 = !DILocation(line: 20, column: 1, scope: !13)

