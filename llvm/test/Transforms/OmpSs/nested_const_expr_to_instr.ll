; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'nested_const_expr_to_instr.ll'
source_filename = "nested_const_expr_to_instr.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@array = dso_local global [10 x i32] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !7 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x i32]* @array) ], !dbg !10
  store i32 0, i32* inttoptr (i32 ptrtoint ([10 x i32]* @array to i32) to i32*), align 4
  call void @llvm.directive.region.exit(token %0), !dbg !12
  ret void, !dbg !13
}

; CHECK: define internal void @nanos6_unpacked_task_region_foo0
; CHECK:   %1 = ptrtoint [10 x i32]* %array to i32
; CHECK:   %2 = inttoptr i32 %1 to i32*
; CHECK:   store i32 0, i32* %2, align 4

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "nested_const_expr_to_instr.ll", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 3, column: 13, scope: !7)
!11 = !DILocation(line: 5, column: 17, scope: !7)
!12 = !DILocation(line: 6, column: 5, scope: !7)
!13 = !DILocation(line: 7, column: 1, scope: !7)
