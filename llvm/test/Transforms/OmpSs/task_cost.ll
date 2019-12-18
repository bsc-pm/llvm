; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_cost.ll'
source_filename = "task_cost.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z3fooIiET_v = comdat any

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z3bari(i32 %n) !dbg !6 {
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
  %call = call i32 @_Z3fooIiET_v(), !dbg !10
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.COST"(i32 %call) ], !dbg !11
  call void @llvm.directive.region.exit(token %3), !dbg !12
  %4 = load i32, i32* %n.addr, align 4, !dbg !13
  %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %n.addr), "QUAL.OSS.COST"(i32 %4) ], !dbg !14
  call void @llvm.directive.region.exit(token %5), !dbg !15
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 1, !dbg !16
  %6 = load i32, i32* %arrayidx, align 4, !dbg !16
  %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1), "QUAL.OSS.COST"(i32 %6) ], !dbg !17
  call void @llvm.directive.region.exit(token %7), !dbg !18
  %8 = load i8*, i8** %saved_stack, align 8, !dbg !19
  call void @llvm.stackrestore(i8* %8), !dbg !19
  ret void, !dbg !19
}

declare i8* @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare i32 @_Z3fooIiET_v()
declare void @llvm.stackrestore(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_cost.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 4, column: 13, scope: !6)
!9 = !DILocation(line: 4, column: 5, scope: !6)
!10 = !DILocation(line: 5, column: 27, scope: !6)
!11 = !DILocation(line: 5, column: 13, scope: !6)
!12 = !DILocation(line: 6, column: 6, scope: !6)
!13 = !DILocation(line: 7, column: 27, scope: !6)
!14 = !DILocation(line: 7, column: 13, scope: !6)
!15 = !DILocation(line: 8, column: 6, scope: !6)
!16 = !DILocation(line: 9, column: 27, scope: !6)
!17 = !DILocation(line: 9, column: 13, scope: !6)
!18 = !DILocation(line: 10, column: 6, scope: !6)
!19 = !DILocation(line: 11, column: 1, scope: !6)
!20 = distinct !DISubprogram(name: "foo<int>", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 1, column: 32, scope: !20)

; CHECK: define internal void @nanos6_unpacked_constraints__Z3bari0(%nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call i32 @_Z3fooIiET_v(), !dbg !15
; CHECK-NEXT:   %gep_constraints = getelementptr %nanos6_task_constraints_t, %nanos6_task_constraints_t* %constraints, i32 0, i32 0
; CHECK-NEXT:   %0 = zext i32 %call to i64
; CHECK-NEXT:   store i64 %0, i64* %gep_constraints
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_constraints__Z3bari0(%nanos6_task_args__Z3bari0* %task_args, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @nanos6_unpacked_constraints__Z3bari0(%nanos6_task_constraints_t* %constraints)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_constraints__Z3bari1(i32* %n.addr, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* %n.addr, align 4, !dbg !14
; CHECK-NEXT:   %gep_constraints = getelementptr %nanos6_task_constraints_t, %nanos6_task_constraints_t* %constraints, i32 0, i32 0
; CHECK-NEXT:   %1 = zext i32 %0 to i64
; CHECK-NEXT:   store i64 %1, i64* %gep_constraints
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_constraints__Z3bari1(%nanos6_task_args__Z3bari1* %task_args, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_n.addr = getelementptr %nanos6_task_args__Z3bari1, %nanos6_task_args__Z3bari1* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_n.addr = load i32*, i32** %gep_n.addr
; CHECK-NEXT:   call void @nanos6_unpacked_constraints__Z3bari1(i32* %load_gep_n.addr, %nanos6_task_constraints_t* %constraints)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_constraints__Z3bari2(i32* %vla, i64 %0, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i32, i32* %vla, i64 1, !dbg !16
; CHECK-NEXT:   %1 = load i32, i32* %arrayidx, align 4, !dbg !16
; CHECK-NEXT:   %gep_constraints = getelementptr %nanos6_task_constraints_t, %nanos6_task_constraints_t* %constraints, i32 0, i32 0
; CHECK-NEXT:   %2 = zext i32 %1 to i64
; CHECK-NEXT:   store i64 %2, i64* %gep_constraints
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_constraints__Z3bari2(%nanos6_task_args__Z3bari2* %task_args, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args__Z3bari2, %nanos6_task_args__Z3bari2* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args__Z3bari2, %nanos6_task_args__Z3bari2* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   call void @nanos6_unpacked_constraints__Z3bari2(i32* %load_gep_vla, i64 %load_capt_gep, %nanos6_task_constraints_t* %constraints)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

