; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'assert.ll'
source_filename = "assert.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; #pragma oss assert("a == b")
; #pragma oss assert("a == c")
; #pragma oss assert("a == d")
;
; int main() {
; }

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !11 {
entry:
  ret i32 0, !dbg !14
}

; CHECK: @0 = private unnamed_addr constant [7 x i8] c"a == b\00", align 1
; CHECK: @1 = private unnamed_addr constant [7 x i8] c"a == c\00", align 1
; CHECK: @2 = private unnamed_addr constant [7 x i8] c"a == d\00", align 1
; CHECK: declare void @nanos6_config_assert(i8*)

; CHECK: define internal void @nanos6_constructor_register_assert() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @nanos6_config_assert(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT:   call void @nanos6_config_assert(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @1, i32 0, i32 0))
; CHECK-NEXT:   call void @nanos6_config_assert(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @2, i32 0, i32 0))
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"OmpSs-2 Metadata", !6}
!6 = !{!7, !8, !9}
!7 = !{!"assert", !"a == b"}
!8 = !{!"assert", !"a == c"}
!9 = !{!"assert", !"a == d"}
!10 = !{!""}
!11 = distinct !DISubprogram(name: "main", scope: !12, file: !12, line: 5, type: !13, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DIFile(filename: "assert.ll", directory: "")
!13 = !DISubroutineType(types: !2)
!14 = !DILocation(line: 6, column: 1, scope: !11)
