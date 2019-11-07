; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_vla.ll'
source_filename = "task_vla.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32 }

define dso_local void @_Z9pod_arrayi(i32 %n) !dbg !6 {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4, !dbg !8
  %1 = zext i32 %0 to i64, !dbg !9
  %2 = call i8* @llvm.stacksave(), !dbg !9
  store i8* %2, i8** %saved_stack, align 8, !dbg !9
  %vla = alloca i32, i64 %1, align 16, !dbg !9
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !9
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !10
  call void @llvm.directive.region.exit(token %3), !dbg !11
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !12
  call void @llvm.directive.region.exit(token %4), !dbg !13

; CHECK: codeRepl:                                         ; preds = %entry
; CHECK-NEXT:   %3 = alloca %nanos6_task_args__Z9pod_arrayi0*, !dbg !10
; CHECK-NEXT:   %4 = bitcast %nanos6_task_args__Z9pod_arrayi0** %3 to i8**, !dbg !10
; CHECK-NEXT:   %5 = alloca i8*, !dbg !10
; CHECK-NEXT:   %6 = mul nuw i64 4, %1, !dbg !10
; CHECK-NEXT:   %7 = add nuw i64 0, %6, !dbg !10
; CHECK-NEXT:   %8 = add nuw i64 16, %7, !dbg !10
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z9pod_arrayi0, %nanos6_task_invocation_info_t* @task_invocation_info__Z9pod_arrayi0, i64 %8, i8** %4, i8** %5, i64 0, i64 0), !dbg !10
; CHECK-NEXT:   %9 = load %nanos6_task_args__Z9pod_arrayi0*, %nanos6_task_args__Z9pod_arrayi0** %3, !dbg !10
; CHECK-NEXT:   %10 = bitcast %nanos6_task_args__Z9pod_arrayi0* %9 to i8*, !dbg !10
; CHECK-NEXT:   %args_end = getelementptr i8, i8* %10, i64 16, !dbg !10
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args__Z9pod_arrayi0, %nanos6_task_args__Z9pod_arrayi0* %9, i32 0, i32 0, !dbg !10
; CHECK-NEXT:   %11 = bitcast i32** %gep_vla to i8**, !dbg !10
; CHECK-NEXT:   store i8* %args_end, i8** %11, align 4, !dbg !10
; CHECK-NEXT:   %12 = mul nuw i64 4, %1, !dbg !10
; CHECK-NEXT:   %13 = getelementptr i8, i8* %args_end, i64 %12, !dbg !10
; CHECK-NEXT:   %dims_gep_ = getelementptr %nanos6_task_args__Z9pod_arrayi0, %nanos6_task_args__Z9pod_arrayi0* %9, i32 0, i32 1, !dbg !10
; CHECK-NEXT:   store i64 %1, i64* %dims_gep_, !dbg !10
; CHECK-NEXT:   %14 = load i8*, i8** %5, !dbg !10
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %14), !dbg !10
; CHECK-NEXT:   br label %15, !dbg !10
; CHECK: 15:                                               ; preds = %codeRepl
; CHECK-NEXT:   br label %codeRepl1, !dbg !11
; CHECK: codeRepl1:                                        ; preds = %15
; CHECK-NEXT:   %16 = alloca %nanos6_task_args__Z9pod_arrayi1*, !dbg !11
; CHECK-NEXT:   %17 = bitcast %nanos6_task_args__Z9pod_arrayi1** %16 to i8**, !dbg !11
; CHECK-NEXT:   %18 = alloca i8*, !dbg !11
; CHECK-NEXT:   %19 = mul nuw i64 4, %1, !dbg !11
; CHECK-NEXT:   %20 = add nuw i64 0, %19, !dbg !11
; CHECK-NEXT:   %21 = add nuw i64 16, %20, !dbg !11
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z9pod_arrayi1, %nanos6_task_invocation_info_t* @task_invocation_info__Z9pod_arrayi1, i64 %21, i8** %17, i8** %18, i64 0, i64 0), !dbg !11
; CHECK-NEXT:   %22 = load %nanos6_task_args__Z9pod_arrayi1*, %nanos6_task_args__Z9pod_arrayi1** %16, !dbg !11
; CHECK-NEXT:   %23 = bitcast %nanos6_task_args__Z9pod_arrayi1* %22 to i8*, !dbg !11
; CHECK-NEXT:   %args_end2 = getelementptr i8, i8* %23, i64 16, !dbg !11
; CHECK-NEXT:   %gep_vla3 = getelementptr %nanos6_task_args__Z9pod_arrayi1, %nanos6_task_args__Z9pod_arrayi1* %22, i32 0, i32 0, !dbg !11
; CHECK-NEXT:   %24 = bitcast i32** %gep_vla3 to i8**, !dbg !11
; CHECK-NEXT:   store i8* %args_end2, i8** %24, align 4, !dbg !11
; CHECK-NEXT:   %25 = mul nuw i64 4, %1, !dbg !11
; CHECK-NEXT:   %26 = getelementptr i8, i8* %args_end2, i64 %25, !dbg !11
; CHECK-NEXT:   %27 = mul nuw i64 1, %1, !dbg !11
; CHECK-NEXT:   %gep_vla4 = getelementptr %nanos6_task_args__Z9pod_arrayi1, %nanos6_task_args__Z9pod_arrayi1* %22, i32 0, i32 0, !dbg !11
; CHECK-NEXT:   %28 = load i32*, i32** %gep_vla4, !dbg !11
; CHECK-NEXT:   %29 = mul nuw i64 %27, 4, !dbg !11
; CHECK-NEXT:   %30 = bitcast i32* %28 to i8*, !dbg !11
; CHECK-NEXT:   %31 = bitcast i32* %vla to i8*, !dbg !11
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %30, i8* align 4 %31, i64 %29, i1 false), !dbg !11
; CHECK-NEXT:   %dims_gep_5 = getelementptr %nanos6_task_args__Z9pod_arrayi1, %nanos6_task_args__Z9pod_arrayi1* %22, i32 0, i32 1, !dbg !11
; CHECK-NEXT:   store i64 %1, i64* %dims_gep_5, !dbg !11
; CHECK-NEXT:   %32 = load i8*, i8** %18, !dbg !11
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %32), !dbg !11
; CHECK-NEXT:   br label %33, !dbg !11

  %5 = load i8*, i8** %saved_stack, align 8, !dbg !14
  call void @llvm.stackrestore(i8* %5), !dbg !14
  ret void, !dbg !14
}

define dso_local void @_Z13non_pod_arrayi(i32 %n) !dbg !15 {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4, !dbg !16
  %1 = zext i32 %0 to i64, !dbg !17
  %2 = call i8* @llvm.stacksave(), !dbg !17
  store i8* %2, i8** %saved_stack, align 8, !dbg !17
  %vla = alloca %struct.S, i64 %1, align 16, !dbg !17
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !17
  %isempty = icmp eq i64 %1, 0, !dbg !18
  br i1 %isempty, label %arrayctor.cont, label %new.ctorloop, !dbg !18

new.ctorloop:                                     ; preds = %entry
  %arrayctor.end = getelementptr inbounds %struct.S, %struct.S* %vla, i64 %1, !dbg !18
  br label %arrayctor.loop, !dbg !18

arrayctor.loop:                                   ; preds = %arrayctor.loop, %new.ctorloop
  %arrayctor.cur = phi %struct.S* [ %vla, %new.ctorloop ], [ %arrayctor.next, %arrayctor.loop ], !dbg !18
  call void @_ZN1SC1Ev(%struct.S* %arrayctor.cur), !dbg !18
  %arrayctor.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.cur, i64 1, !dbg !18
  %arrayctor.done = icmp eq %struct.S* %arrayctor.next, %arrayctor.end, !dbg !18
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !18

arrayctor.cont:                                   ; preds = %entry, %arrayctor.loop
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(%struct.S* %vla), "QUAL.OSS.VLA.DIMS"(%struct.S* %vla, i64 %1), "QUAL.OSS.INIT"(%struct.S* %vla, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %vla, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !19
  call void @llvm.directive.region.exit(token %3), !dbg !20
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %vla), "QUAL.OSS.VLA.DIMS"(%struct.S* %vla, i64 %1), "QUAL.OSS.COPY"(%struct.S* %vla, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_), "QUAL.OSS.DEINIT"(%struct.S* %vla, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !21
  call void @llvm.directive.region.exit(token %4), !dbg !22

; CHECK: codeRepl:                                         ; preds = %arrayctor.cont
; CHECK-NEXT:   %3 = alloca %nanos6_task_args__Z13non_pod_arrayi0*, !dbg !17
; CHECK-NEXT:   %4 = bitcast %nanos6_task_args__Z13non_pod_arrayi0** %3 to i8**, !dbg !17
; CHECK-NEXT:   %5 = alloca i8*, !dbg !17
; CHECK-NEXT:   %6 = mul nuw i64 4, %1, !dbg !17
; CHECK-NEXT:   %7 = add nuw i64 0, %6, !dbg !17
; CHECK-NEXT:   %8 = add nuw i64 16, %7, !dbg !17
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z13non_pod_arrayi0, %nanos6_task_invocation_info_t* @task_invocation_info__Z13non_pod_arrayi0, i64 %8, i8** %4, i8** %5, i64 0, i64 0), !dbg !17
; CHECK-NEXT:   %9 = load %nanos6_task_args__Z13non_pod_arrayi0*, %nanos6_task_args__Z13non_pod_arrayi0** %3, !dbg !17
; CHECK-NEXT:   %10 = bitcast %nanos6_task_args__Z13non_pod_arrayi0* %9 to i8*, !dbg !17
; CHECK-NEXT:   %args_end = getelementptr i8, i8* %10, i64 16, !dbg !17
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args__Z13non_pod_arrayi0, %nanos6_task_args__Z13non_pod_arrayi0* %9, i32 0, i32 0, !dbg !17
; CHECK-NEXT:   %11 = bitcast %struct.S** %gep_vla to i8**, !dbg !17
; CHECK-NEXT:   store i8* %args_end, i8** %11, align 8, !dbg !17
; CHECK-NEXT:   %12 = mul nuw i64 4, %1, !dbg !17
; CHECK-NEXT:   %13 = getelementptr i8, i8* %args_end, i64 %12, !dbg !17
; CHECK-NEXT:   %14 = mul nuw i64 1, %1, !dbg !17
; CHECK-NEXT:   %gep_vla1 = getelementptr %nanos6_task_args__Z13non_pod_arrayi0, %nanos6_task_args__Z13non_pod_arrayi0* %9, i32 0, i32 0, !dbg !17
; CHECK-NEXT:   %15 = load %struct.S*, %struct.S** %gep_vla1, !dbg !17
; CHECK-NEXT:   call void @oss_ctor_ZN1SC1Ev(%struct.S* %15, i64 %14), !dbg !17
; CHECK-NEXT:   %dims_gep_ = getelementptr %nanos6_task_args__Z13non_pod_arrayi0, %nanos6_task_args__Z13non_pod_arrayi0* %9, i32 0, i32 1, !dbg !17
; CHECK-NEXT:   store i64 %1, i64* %dims_gep_, !dbg !17
; CHECK-NEXT:   %16 = load i8*, i8** %5, !dbg !17
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %16), !dbg !17
; CHECK-NEXT:   br label %17, !dbg !17
; CHECK: 17:                                               ; preds = %codeRepl
; CHECK-NEXT:   br label %codeRepl2, !dbg !18
; CHECK: codeRepl2:                                        ; preds = %17
; CHECK-NEXT:   %18 = alloca %nanos6_task_args__Z13non_pod_arrayi1*, !dbg !18
; CHECK-NEXT:   %19 = bitcast %nanos6_task_args__Z13non_pod_arrayi1** %18 to i8**, !dbg !18
; CHECK-NEXT:   %20 = alloca i8*, !dbg !18
; CHECK-NEXT:   %21 = mul nuw i64 4, %1, !dbg !18
; CHECK-NEXT:   %22 = add nuw i64 0, %21, !dbg !18
; CHECK-NEXT:   %23 = add nuw i64 16, %22, !dbg !18
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var__Z13non_pod_arrayi1, %nanos6_task_invocation_info_t* @task_invocation_info__Z13non_pod_arrayi1, i64 %23, i8** %19, i8** %20, i64 0, i64 0), !dbg !18
; CHECK-NEXT:   %24 = load %nanos6_task_args__Z13non_pod_arrayi1*, %nanos6_task_args__Z13non_pod_arrayi1** %18, !dbg !18
; CHECK-NEXT:   %25 = bitcast %nanos6_task_args__Z13non_pod_arrayi1* %24 to i8*, !dbg !18
; CHECK-NEXT:   %args_end3 = getelementptr i8, i8* %25, i64 16, !dbg !18
; CHECK-NEXT:   %gep_vla4 = getelementptr %nanos6_task_args__Z13non_pod_arrayi1, %nanos6_task_args__Z13non_pod_arrayi1* %24, i32 0, i32 0, !dbg !18
; CHECK-NEXT:   %26 = bitcast %struct.S** %gep_vla4 to i8**, !dbg !18
; CHECK-NEXT:   store i8* %args_end3, i8** %26, align 8, !dbg !18
; CHECK-NEXT:   %27 = mul nuw i64 4, %1, !dbg !18
; CHECK-NEXT:   %28 = getelementptr i8, i8* %args_end3, i64 %27, !dbg !18
; CHECK-NEXT:   %29 = mul nuw i64 1, %1, !dbg !18
; CHECK-NEXT:   %gep_vla5 = getelementptr %nanos6_task_args__Z13non_pod_arrayi1, %nanos6_task_args__Z13non_pod_arrayi1* %24, i32 0, i32 0, !dbg !18
; CHECK-NEXT:   %30 = load %struct.S*, %struct.S** %gep_vla5, !dbg !18
; CHECK-NEXT:   call void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %vla, %struct.S* %30, i64 %29), !dbg !18
; CHECK-NEXT:   %dims_gep_6 = getelementptr %nanos6_task_args__Z13non_pod_arrayi1, %nanos6_task_args__Z13non_pod_arrayi1* %24, i32 0, i32 1, !dbg !18
; CHECK-NEXT:   store i64 %1, i64* %dims_gep_6, !dbg !18
; CHECK-NEXT:   %31 = load i8*, i8** %20, !dbg !18
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %31), !dbg !18
; CHECK-NEXT:   br label %32, !dbg !18

  %5 = getelementptr inbounds %struct.S, %struct.S* %vla, i64 %1, !dbg !23
  %arraydestroy.isempty = icmp eq %struct.S* %vla, %5, !dbg !23
  br i1 %arraydestroy.isempty, label %arraydestroy.done1, label %arraydestroy.body, !dbg !23

arraydestroy.body:                                ; preds = %arraydestroy.body, %arrayctor.cont
  %arraydestroy.elementPast = phi %struct.S* [ %5, %arrayctor.cont ], [ %arraydestroy.element, %arraydestroy.body ], !dbg !23
  %arraydestroy.element = getelementptr inbounds %struct.S, %struct.S* %arraydestroy.elementPast, i64 -1, !dbg !23
  call void @_ZN1SD1Ev(%struct.S* %arraydestroy.element) #1, !dbg !23
  %arraydestroy.done = icmp eq %struct.S* %arraydestroy.element, %vla, !dbg !23
  br i1 %arraydestroy.done, label %arraydestroy.done1, label %arraydestroy.body, !dbg !23

arraydestroy.done1:                               ; preds = %arraydestroy.body, %arrayctor.cont
  %6 = load i8*, i8** %saved_stack, align 8, !dbg !23
  call void @llvm.stackrestore(i8* %6), !dbg !23
  ret void, !dbg !23
}

; CHECK: define internal void @nanos6_unpacked_task_region__Z13non_pod_arrayi0(%struct.S* %vla, i64 %0, i8* %1, %nanos6_address_translation_entry_t* %2) {
; CHECK:   call void @oss_dtor_ZN1SD1Ev(%struct.S* %vla, i64 %4)
; CHECK: }

; CHECK: define internal void @nanos6_unpacked_task_region__Z13non_pod_arrayi1(%struct.S* %vla, i64 %0, i8* %1, %nanos6_address_translation_entry_t* %2) {
; CHECK:   call void @oss_dtor_ZN1SD1Ev(%struct.S* %vla, i64 %4)
; CHECK: }

declare i8* @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(i8*)
declare dso_local void @_ZN1SC1Ev(%struct.S*)
declare dso_local void @oss_ctor_ZN1SC1Ev(%struct.S* %0, i64 %1)
declare dso_local void @_ZN1SD1Ev(%struct.S*)
declare dso_local void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1)
declare dso_local void @_ZN1SC1ERKS_(%struct.S*, %struct.S* dereferenceable(4))
declare dso_local void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %0, %struct.S* %1, i64 %2)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_vla.ll", directory: "llvm/test/Transforms/OmpSs")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "pod_array", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 15, scope: !6)
!9 = !DILocation(line: 2, column: 5, scope: !6)
!10 = !DILocation(line: 3, column: 13, scope: !6)
!11 = !DILocation(line: 4, column: 6, scope: !6)
!12 = !DILocation(line: 5, column: 13, scope: !6)
!13 = !DILocation(line: 6, column: 6, scope: !6)
!14 = !DILocation(line: 7, column: 1, scope: !6)
!15 = distinct !DISubprogram(name: "non_pod_array", scope: !1, file: !1, line: 16, type: !7, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 17, column: 13, scope: !15)
!17 = !DILocation(line: 17, column: 5, scope: !15)
!18 = !DILocation(line: 17, column: 7, scope: !15)
!19 = !DILocation(line: 18, column: 13, scope: !15)
!20 = !DILocation(line: 19, column: 6, scope: !15)
!21 = !DILocation(line: 20, column: 13, scope: !15)
!22 = !DILocation(line: 21, column: 6, scope: !15)
!23 = !DILocation(line: 22, column: 1, scope: !15)

