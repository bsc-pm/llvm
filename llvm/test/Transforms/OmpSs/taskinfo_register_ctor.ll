; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'taskinfo_register_ctor.c'
source_filename = "taskinfo_register_ctor.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int main() {
;     #pragma oss task
;     {}
;     #pragma oss task
;     {}
; }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ], !dbg !8
  call void @llvm.directive.region.exit(token %0), !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  ret i32 0, !dbg !12
}

; CHECK: define internal void @nanos6_constructor_register_task_info() {
; CHECK: entry:
; CHECK:   call void @nanos6_register_task_info(%nanos6_task_info_t* @task_info_var_main0)
; CHECK:   call void @nanos6_register_task_info(%nanos6_task_info_t* @task_info_var_main1)
; CHECK:   ret void
; CHECK: }

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "taskinfo_register_ctor.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 13, scope: !6)
!9 = !DILocation(line: 3, column: 6, scope: !6)
!10 = !DILocation(line: 4, column: 13, scope: !6)
!11 = !DILocation(line: 5, column: 6, scope: !6)
!12 = !DILocation(line: 6, column: 1, scope: !6)
