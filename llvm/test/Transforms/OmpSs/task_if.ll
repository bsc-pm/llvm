; RUN: opt %s -ompss-2 -S | FileCheck %s
source_filename = "task_if.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.IF"(i1 false) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !10
  ret void, !dbg !11
}

; task_flags | (!If << 1)
; CHECK: call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_foo0, %nanos6_task_invocation_info_t* @task_invocation_info_foo0, i64 0, i8** %1, i8** %2, i64 2, i64 %3)

declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)

!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_if.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_if.c", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, scope: !6)
!10 = !DILocation(line: 3, scope: !6)
!11 = !DILocation(line: 4, scope: !6)
