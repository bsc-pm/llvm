; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_cost.ll'
source_filename = "task_cost.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z3fooIiET_v = comdat any

; template<typename T> T foo() { return T(); }
; void bar(int n) {
;     int vla[n];
;     #pragma oss task cost(foo<int>())
;     {}
;     #pragma oss task cost(n)
;     {}
;     #pragma oss task cost(vla[1])
;     {}
; }

; Function Attrs: noinline nounwind optnone mustprogress
define dso_local void @_Z3bari(i32 %n) #0 !dbg !6 {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4, !dbg !9
  %1 = zext i32 %0 to i64, !dbg !10
  %2 = call i8* @llvm.stacksave(), !dbg !10
  store i8* %2, i8** %saved_stack, align 8, !dbg !10
  %vla = alloca i32, i64 %1, align 16, !dbg !10
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !10
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.COST"(i32 ()* @compute_cost) ], !dbg !11
  call void @llvm.directive.region.exit(token %3), !dbg !12
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.COST"(i32 (i32*)* @compute_cost.1, i32* %n.addr) ], !dbg !13
  call void @llvm.directive.region.exit(token %4), !dbg !14
  %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.COST"(i32 (i32*, i64)* @compute_cost.2, i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !15
  call void @llvm.directive.region.exit(token %5), !dbg !16
  %6 = load i8*, i8** %saved_stack, align 8, !dbg !17
  call void @llvm.stackrestore(i8* %6), !dbg !17
  ret void, !dbg !17
}

; Function Attrs: nofree nosync nounwind willreturn
declare i8* @llvm.stacksave() #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

define internal i32 @compute_cost() !dbg !18 {
entry:
  %call = call i32 @_Z3fooIiET_v(), !dbg !19
  ret i32 %call, !dbg !19
}

; Function Attrs: noinline nounwind optnone mustprogress
define linkonce_odr i32 @_Z3fooIiET_v() #0 comdat !dbg !21 {
entry:
  ret i32 0, !dbg !22
}

define internal i32 @compute_cost.1(i32* %n) !dbg !23 {
entry:
  %n.addr = alloca i32*, align 8
  store i32* %n, i32** %n.addr, align 8
  %0 = load i32, i32* %n, align 4, !dbg !24
  ret i32 %0, !dbg !24
}

define internal i32 @compute_cost.2(i32* %vla, i64 %0) !dbg !26 {
entry:
  %vla.addr = alloca i32*, align 8
  %.addr = alloca i64, align 8
  store i32* %vla, i32** %vla.addr, align 8
  store i64 %0, i64* %.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 1, !dbg !27
  %1 = load i32, i32* %arrayidx, align 4, !dbg !27
  ret i32 %1, !dbg !29
}

; CHECK: define internal void @nanos6_unpacked_constraints__Z3bari0(%nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_constraints = getelementptr %nanos6_task_constraints_t, %nanos6_task_constraints_t* %constraints, i32 0, i32 0
; CHECK-NEXT:   %0 = call i32 @compute_cost()
; CHECK-NEXT:   %1 = zext i32 %0 to i64
; CHECK-NEXT:   store i64 %1, i64* %gep_constraints, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_constraints__Z3bari0(%nanos6_task_args__Z3bari0* %task_args, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @nanos6_unpacked_constraints__Z3bari0(%nanos6_task_constraints_t* %constraints)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_constraints__Z3bari1(i32* %n.addr, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_constraints = getelementptr %nanos6_task_constraints_t, %nanos6_task_constraints_t* %constraints, i32 0, i32 0
; CHECK-NEXT:   %0 = call i32 @compute_cost.1(i32* %n.addr)
; CHECK-NEXT:   %1 = zext i32 %0 to i64
; CHECK-NEXT:   store i64 %1, i64* %gep_constraints, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_constraints__Z3bari1(%nanos6_task_args__Z3bari1* %task_args, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_n.addr = getelementptr %nanos6_task_args__Z3bari1, %nanos6_task_args__Z3bari1* %task_args, i32 0, i32 0
; CHECK-NEXT:   call void @nanos6_unpacked_constraints__Z3bari1(i32* %gep_n.addr, %nanos6_task_constraints_t* %constraints)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_constraints__Z3bari2(i32* %vla, i64 %0, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_constraints = getelementptr %nanos6_task_constraints_t, %nanos6_task_constraints_t* %constraints, i32 0, i32 0
; CHECK-NEXT:   %1 = call i32 @compute_cost.2(i32* %vla, i64 %0)
; CHECK-NEXT:   %2 = zext i32 %1 to i64
; CHECK-NEXT:   store i64 %2, i64* %gep_constraints, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_constraints__Z3bari2(%nanos6_task_args__Z3bari2* %task_args, %nanos6_task_constraints_t* %constraints) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args__Z3bari2, %nanos6_task_args__Z3bari2* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla, align 8
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args__Z3bari2, %nanos6_task_args__Z3bari2* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep, align 8
; CHECK-NEXT:   call void @nanos6_unpacked_constraints__Z3bari2(i32* %load_gep_vla, i64 %load_capt_gep, %nanos6_task_constraints_t* %constraints)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.stackrestore(i8*) #1

attributes #0 = { noinline nounwind optnone mustprogress "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nofree nosync nounwind willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 13.0.0"}
!6 = distinct !DISubprogram(name: "bar", scope: !7, file: !7, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_cost.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 13, scope: !6)
!10 = !DILocation(line: 3, column: 5, scope: !6)
!11 = !DILocation(line: 4, column: 13, scope: !6)
!12 = !DILocation(line: 5, column: 6, scope: !6)
!13 = !DILocation(line: 6, column: 13, scope: !6)
!14 = !DILocation(line: 7, column: 6, scope: !6)
!15 = !DILocation(line: 8, column: 13, scope: !6)
!16 = !DILocation(line: 9, column: 6, scope: !6)
!17 = !DILocation(line: 10, column: 1, scope: !6)
!18 = distinct !DISubprogram(linkageName: "compute_cost", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!19 = !DILocation(line: 4, column: 27, scope: !20)
!20 = !DILexicalBlockFile(scope: !18, file: !7, discriminator: 0)
!21 = distinct !DISubprogram(name: "foo<int>", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 1, column: 32, scope: !21)
!23 = distinct !DISubprogram(linkageName: "compute_cost.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 6, column: 27, scope: !25)
!25 = !DILexicalBlockFile(scope: !23, file: !7, discriminator: 0)
!26 = distinct !DISubprogram(linkageName: "compute_cost.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 8, column: 27, scope: !28)
!28 = !DILexicalBlockFile(scope: !26, file: !7, discriminator: 0)
!29 = !DILocation(line: 8, column: 31, scope: !28)
