; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'constexpr_to_instr.ll'
source_filename = "constexpr_to_instr.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; check that constant gep is converted to instruction

; int array[10];
; void foo() {
;     #pragma oss task firstprivate(array)
;     {
;         array[2] = 2;
;     }
; }

@array = dso_local global [10 x i32] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone
define dso_local void @foo() #0 !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]* @array) ], !dbg !9
  store i32 2, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array, i64 0, i64 2), align 8, !dbg !10
  call void @llvm.directive.region.exit(token %0), !dbg !11
  ret void, !dbg !12
}

; CHECK-LABEL @nanos6_unpacked_task_region_foo0(
; CHECK: [[VAL:%.*]] = getelementptr inbounds [10 x i32], [10 x i32]* %array, i64 0, i64 2


; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 12.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "constexpr_to_instr.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 13, scope: !6)
!10 = !DILocation(line: 5, column: 18, scope: !6)
!11 = !DILocation(line: 6, column: 5, scope: !6)
!12 = !DILocation(line: 7, column: 1, scope: !6)
