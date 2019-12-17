; RUN: opt %s -ompss-2 -S | FileCheck %s

source_filename = "task_shared_vla_depend.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @vla_senction_dep(i32 %n, i32 %k, i32 %j) !dbg !6 {
entry:
  %n.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %__vla_expr1 = alloca i64, align 8
  %__vla_expr2 = alloca i64, align 8
  %size3 = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  store i32 %k, i32* %k.addr, align 4
  store i32 %j, i32* %j.addr, align 4
  %0 = load i32, i32* %n.addr, align 4, !dbg !8
  %add = add nsw i32 %0, 1, !dbg !9
  %1 = zext i32 %add to i64, !dbg !10
  %2 = load i32, i32* %k.addr, align 4, !dbg !11
  %add1 = add nsw i32 %2, 2, !dbg !12
  %3 = zext i32 %add1 to i64, !dbg !10
  %4 = load i32, i32* %j.addr, align 4, !dbg !13
  %add2 = add nsw i32 %4, 3, !dbg !14
  %5 = zext i32 %add2 to i64, !dbg !10
  %6 = call i8* @llvm.stacksave(), !dbg !10
  store i8* %6, i8** %saved_stack, align 8, !dbg !10
  %7 = mul nuw i64 %1, %3, !dbg !10
  %8 = mul nuw i64 %7, %5, !dbg !10
  %vla = alloca i32, i64 %8, align 16, !dbg !10
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !10
  store i64 %3, i64* %__vla_expr1, align 8, !dbg !10
  store i64 %5, i64* %__vla_expr2, align 8, !dbg !10
  %9 = mul i64 %5, 4, !dbg !15
  %10 = mul i64 %5, 4, !dbg !15
  %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5), "QUAL.OSS.DEP.IN"(i32* %vla, i64 %9, i64 0, i64 %10, i64 %3, i64 0, i64 %3, i64 %1, i64 0, i64 %1) ], !dbg !15
  %size = alloca i32, align 4
  %12 = mul nuw i64 %1, %3, !dbg !16
  %13 = mul nuw i64 %12, %5, !dbg !16
  %14 = mul nuw i64 4, %13, !dbg !16
  %conv = trunc i64 %14 to i32, !dbg !16
  store i32 %conv, i32* %size, align 4, !dbg !17
  call void @llvm.directive.region.exit(token %11), !dbg !18
  %15 = mul nuw i64 %1, %3, !dbg !19
  %16 = mul nuw i64 %15, %5, !dbg !19
  %17 = mul nuw i64 4, %16, !dbg !19
  %conv4 = trunc i64 %17 to i32, !dbg !19
  store i32 %conv4, i32* %size3, align 4, !dbg !20
  %18 = load i8*, i8** %saved_stack, align 8, !dbg !21
  call void @llvm.stackrestore(i8* %18), !dbg !21
  ret void, !dbg !21

; CHECK: codeRepl:                                         ; preds = %entry
; CHECK-NEXT:   %9 = alloca %nanos6_task_args_vla_senction_dep0*, !dbg !13
; CHECK-NEXT:   %10 = bitcast %nanos6_task_args_vla_senction_dep0** %9 to i8**, !dbg !13
; CHECK-NEXT:   %11 = alloca i8*, !dbg !13
; CHECK-NEXT:   %12 = mul nuw i64 4, %1, !dbg !13
; CHECK-NEXT:   %13 = mul nuw i64 %12, %3, !dbg !13
; CHECK-NEXT:   %14 = mul nuw i64 %13, %5, !dbg !13
; CHECK-NEXT:   %15 = add nuw i64 0, %14, !dbg !13
; CHECK-NEXT:   %16 = add nuw i64 32, %15, !dbg !13
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_vla_senction_dep0, %nanos6_task_invocation_info_t* @task_invocation_info_vla_senction_dep0, i64 %16, i8** %10, i8** %11, i64 0, i64 1), !dbg !13
; CHECK-NEXT:   %17 = load %nanos6_task_args_vla_senction_dep0*, %nanos6_task_args_vla_senction_dep0** %9, !dbg !13
; CHECK-NEXT:   %18 = bitcast %nanos6_task_args_vla_senction_dep0* %17 to i8*, !dbg !13
; CHECK-NEXT:   %args_end = getelementptr i8, i8* %18, i64 32, !dbg !13
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %17, i32 0, i32 0, !dbg !13
; CHECK-NEXT:   %19 = bitcast i32** %gep_vla to i8**, !dbg !13
; CHECK-NEXT:   store i8* %args_end, i8** %19, align 4, !dbg !13
; CHECK-NEXT:   %20 = mul nuw i64 4, %1, !dbg !13
; CHECK-NEXT:   %21 = mul nuw i64 %20, %3, !dbg !13
; CHECK-NEXT:   %22 = mul nuw i64 %21, %5, !dbg !13
; CHECK-NEXT:   %23 = getelementptr i8, i8* %args_end, i64 %22, !dbg !13
; CHECK-NEXT:   %gep_vla1 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %17, i32 0, i32 0, !dbg !13
; CHECK-NEXT:   store i32* %vla, i32** %gep_vla1, !dbg !13
; CHECK-NEXT:   %capt_gep_ = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %17, i32 0, i32 1, !dbg !13
; CHECK-NEXT:   store i64 %1, i64* %capt_gep_, !dbg !13
; CHECK-NEXT:   %capt_gep_2 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %17, i32 0, i32 2, !dbg !13
; CHECK-NEXT:   store i64 %3, i64* %capt_gep_2, !dbg !13
; CHECK-NEXT:   %capt_gep_3 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %17, i32 0, i32 3, !dbg !13
; CHECK-NEXT:   store i64 %5, i64* %capt_gep_3, !dbg !13
; CHECK-NEXT:   %24 = load i8*, i8** %11, !dbg !13
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %24), !dbg !13
; CHECK-NEXT:   br label %25, !dbg !13
; CHECK: 25:                                               ; preds = %codeRepl
; CHECK-NEXT:   %26 = mul nuw i64 %1, %3, !dbg !14
; CHECK-NEXT:   %27 = mul nuw i64 %26, %5, !dbg !14
; CHECK-NEXT:   %28 = mul nuw i64 4, %27, !dbg !14
; CHECK-NEXT:   %conv4 = trunc i64 %28 to i32, !dbg !14
; CHECK-NEXT:   store i32 %conv4, i32* %size3, align 4, !dbg !15
; CHECK-NEXT:   %29 = load i8*, i8** %saved_stack, align 8, !dbg !16
; CHECK-NEXT:   call void @llvm.stackrestore(i8* %29), !dbg !16
; CHECK-NEXT:   ret void, !dbg !16

}

; CHECK: define internal void @nanos6_unpacked_task_region_vla_senction_dep0(i32* %vla, i64 %0, i64 %1, i64 %2, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   br label %3, !dbg !13
; CHECK: 3:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %size = alloca i32, align 4
; CHECK-NEXT:   %4 = mul nuw i64 %0, %1, !dbg !17
; CHECK-NEXT:   %5 = mul nuw i64 %4, %2, !dbg !17
; CHECK-NEXT:   %6 = mul nuw i64 4, %5, !dbg !17
; CHECK-NEXT:   %conv = trunc i64 %6 to i32, !dbg !17
; CHECK-NEXT:   store i32 %conv, i32* %size, align 4, !dbg !18
; CHECK-NEXT:   ret void, !dbg !14
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_task_region_vla_senction_dep0(%nanos6_task_args_vla_senction_dep0* %task_args, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   %capt_gep1 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 2
; CHECK-NEXT:   %load_capt_gep1 = load i64, i64* %capt_gep1
; CHECK-NEXT:   %capt_gep2 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 3
; CHECK-NEXT:   %load_capt_gep2 = load i64, i64* %capt_gep2
; CHECK-NEXT:   call void @nanos6_unpacked_task_region_vla_senction_dep0(i32* %load_gep_vla, i64 %load_capt_gep, i64 %load_capt_gep1, i64 %load_capt_gep2, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_vla_senction_dep0(i32* %vla, i64 %0, i64 %1, i64 %2, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %3 = mul i64 %2, 4, !dbg !13
; CHECK-NEXT:   %4 = mul i64 %2, 4, !dbg !13
; CHECK-NEXT:   %5 = bitcast i32* %vla to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo3(i8* %handler, i32 0, i8* null, i8* %5, i64 %3, i64 0, i64 %4, i64 %1, i64 0, i64 %1, i64 %0, i64 0, i64 %0)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_deps_vla_senction_dep0(%nanos6_task_args_vla_senction_dep0* %task_args, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   %capt_gep1 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 2
; CHECK-NEXT:   %load_capt_gep1 = load i64, i64* %capt_gep1
; CHECK-NEXT:   %capt_gep2 = getelementptr %nanos6_task_args_vla_senction_dep0, %nanos6_task_args_vla_senction_dep0* %task_args, i32 0, i32 3
; CHECK-NEXT:   %load_capt_gep2 = load i64, i64* %capt_gep2
; CHECK-NEXT:   call void @nanos6_unpacked_deps_vla_senction_dep0(i32* %load_gep_vla, i64 %load_capt_gep, i64 %load_capt_gep1, i64 %load_capt_gep2, i8* %handler)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

declare i8* @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(i8*)

!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_shared_vla_depend.c", directory: "")!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !7, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 15, scope: !6)
!9 = !DILocation(line: 3, column: 17, scope: !6)
!10 = !DILocation(line: 3, column: 5, scope: !6)
!11 = !DILocation(line: 3, column: 22, scope: !6)
!12 = !DILocation(line: 3, column: 24, scope: !6)
!13 = !DILocation(line: 3, column: 29, scope: !6)
!14 = !DILocation(line: 3, column: 31, scope: !6)
!15 = !DILocation(line: 4, column: 13, scope: !6)
!16 = !DILocation(line: 6, column: 20, scope: !6)
!17 = !DILocation(line: 6, column: 13, scope: !6)
!18 = !DILocation(line: 7, column: 5, scope: !6)
!19 = !DILocation(line: 8, column: 16, scope: !6)
!20 = !DILocation(line: 8, column: 9, scope: !6)
!21 = !DILocation(line: 9, column: 1, scope: !6)
