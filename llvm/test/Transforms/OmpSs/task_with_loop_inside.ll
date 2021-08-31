; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop.ll'
source_filename = "loop.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ], !dbg !8
  %i = alloca i32, align 4
  store i32 0, i32* %i, align 4, !dbg !9
  br label %for.cond, !dbg !10

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4, !dbg !11
  %cmp = icmp slt i32 %1, 10, !dbg !12
  br i1 %cmp, label %for.body, label %for.end, !dbg !13

for.body:                                         ; preds = %for.cond
  br label %for.inc, !dbg !14

for.inc:                                          ; preds = %for.body
  %2 = load i32, i32* %i, align 4, !dbg !15
  %inc = add nsw i32 %2, 1, !dbg !15
  store i32 %inc, i32* %i, align 4, !dbg !15
  br label %for.cond, !dbg !13, !llvm.loop !16

for.end:                                          ; preds = %for.cond
  call void @llvm.directive.region.exit(token %0), !dbg !17
  %3 = load i32, i32* %retval, align 4, !dbg !18
  ret i32 %3, !dbg !18
}

; CHECK: define internal void @nanos6_unpacked_task_region_main0(i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %i = alloca i32, align 4
; CHECK-NEXT:   store i32 0, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.cond:                                         ; preds = %for.inc, %0
; CHECK-NEXT:   %1 = load i32, i32* %i, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %1, 10
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   br label %for.inc
; CHECK: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   br label %.exitStub
; CHECK: for.inc:                                          ; preds = %for.body
; CHECK-NEXT:   %2 = load i32, i32* %i, align 4
; CHECK-NEXT:   %inc = add nsw i32 %2, 1
; CHECK-NEXT:   store i32 %inc, i32* %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: .exitStub:                                        ; preds = %for.end
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "loop.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 13, scope: !6)
!9 = !DILocation(line: 5, column: 18, scope: !6)
!10 = !DILocation(line: 5, column: 14, scope: !6)
!11 = !DILocation(line: 5, column: 25, scope: !6)
!12 = !DILocation(line: 5, column: 27, scope: !6)
!13 = !DILocation(line: 5, column: 9, scope: !6)
!14 = !DILocation(line: 6, column: 10, scope: !6)
!15 = !DILocation(line: 5, column: 33, scope: !6)
!16 = distinct !{!16, !13, !14}
!17 = !DILocation(line: 7, column: 5, scope: !6)
!18 = !DILocation(line: 8, column: 1, scope: !6)


