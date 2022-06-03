; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_onready.ll'
source_filename = "task_onready.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo(int a, int b) {}
;
; int main() {
;     int x, y;
;     #pragma oss task onready(foo(x, y))
;     {
;     }
; }

; Function Attrs: noinline nounwind optnone
define dso_local void @foo(i32 %a, i32 %b) #0 !dbg !6 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  ret void, !dbg !9
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !10 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %x), "QUAL.OSS.FIRSTPRIVATE"(i32* %y), "QUAL.OSS.ONREADY"(void (i32*, i32*)* @compute_onready, i32* %x, i32* %y) ], !dbg !11
  call void @llvm.directive.region.exit(token %0), !dbg !12
  ret i32 0, !dbg !13
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal void @compute_onready(i32* %x, i32* %y) !dbg !14 {
entry:
  %x.addr = alloca i32*, align 8
  %y.addr = alloca i32*, align 8
  store i32* %x, i32** %x.addr, align 8
  store i32* %y, i32** %y.addr, align 8
  %0 = load i32, i32* %x, align 4, !dbg !15
  %1 = load i32, i32* %y, align 4, !dbg !17
  call void @foo(i32 %0, i32 %1), !dbg !18
  ret void, !dbg !17
}

; CHECK: define internal void @nanos6_unpacked_onready_main0(i32* %x, i32* %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @compute_onready(i32* %x, i32* %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_priority_main0(%nanos6_task_args_main0* %task_args) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_x = getelementptr %nanos6_task_args_main0, %nanos6_task_args_main0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %gep_y = getelementptr %nanos6_task_args_main0, %nanos6_task_args_main0* %task_args, i32 0, i32 1
; CHECK-NEXT:   call void @nanos6_unpacked_onready_main0(i32* %gep_x, i32* %gep_y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 13.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_onready.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 1, column: 25, scope: !6)
!10 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 3, type: !8, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 5, column: 13, scope: !10)
!12 = !DILocation(line: 7, column: 5, scope: !10)
!13 = !DILocation(line: 8, column: 1, scope: !10)
!14 = distinct !DISubprogram(linkageName: "compute_onready", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 5, column: 34, scope: !16)
!16 = !DILexicalBlockFile(scope: !14, file: !7, discriminator: 0)
!17 = !DILocation(line: 5, column: 37, scope: !16)
!18 = !DILocation(line: 5, column: 30, scope: !16)
