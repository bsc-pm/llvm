; RUN: opt %s -ompss-2 -enable-new-pm=0 -S | FileCheck %s
; RUN: opt %s -ompss-2 -enable-new-pm -S | FileCheck -check-prefix=CHECK-NEW %s
; ModuleID = 'new-pm.ll'
source_filename = "new-pm.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int main() {
;     #pragma oss task
;     {}
; }

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !10
  ret i32 0, !dbg !11
}

; CHECK-NOT: call token @llvm.directive.region.entry
; CHECK-NOT: call void @llvm.directive.region.exit
; CHECK-NEW-NOT: call token @llvm.directive.region.entry
; CHECK-NEW-NOT: call void @llvm.directive.region.exit

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
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "new-pm.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 13, scope: !6)
!10 = !DILocation(line: 3, column: 6, scope: !6)
!11 = !DILocation(line: 4, column: 1, scope: !6)
