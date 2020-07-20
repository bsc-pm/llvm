; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 't1.c'
source_filename = "t1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; NOTE: this test has been typed by hand from an 'oss task' example

; int sum = 0;
;
; void taskloop(int lb, int ub, int step) {
;     int i;
;     #pragma oss task
;     sum += i;
; }
;
; int main() {
;     taskloop(-55, 31, 2);
; }

@sum = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @taskloop(i32 %lb, i32 %ub, i32 %step) #0 !dbg !6 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4

  %lb.value = load i32, i32* %lb.addr, align 4
  %ub.value = load i32, i32* %ub.addr, align 4
  %step.value = load i32, i32* %step.addr, align 4
  %i.addr = alloca i32, align 4
  %new.ub.addr = alloca i32, align 4

  %region = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"),
            "QUAL.OSS.PRIVATE"(i32* %i.addr),
            "QUAL.OSS.CAPTURED"(i32 %lb.value, i32 %ub.value, i32 %step.value),
            "QUAL.OSS.LOOP.TYPE"(i32 0),
            "QUAL.OSS.LOOP.IND.VAR"(i32* %i.addr),
            "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %lb.value),
            "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %ub.value),
            "QUAL.OSS.LOOP.STEP"(i32 %step.value) ], !dbg !8

  %0 = load i32, i32* @sum, align 4, !dbg !9
  %i = load i32, i32* %i.addr, align 4, !dbg !9
  %add = add nsw i32 %0, %i, !dbg !9
  store i32 %add, i32* @sum, align 4, !dbg !9

  call void @llvm.directive.region.exit(token %region), !dbg !10

  ret void, !dbg !11
}

; CHECK: define internal void @nanos6_unpacked_task_region_taskloop0(i32* %i.addr, i32 %lb.value, i32 %ub.value, i32 %step.value, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table) !dbg !14 {
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0, !dbg !15
; CHECK: .exitStub:                                        ; preds = %for.cond1
; CHECK-NEXT:   ret void
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0, !dbg !15
; CHECK-NEXT:   %1 = load i64, i64* %lb_gep, align 8, !dbg !15
; CHECK-NEXT:   %lb4 = trunc i64 %1 to i32, !dbg !15
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1, !dbg !15
; CHECK-NEXT:   %2 = load i64, i64* %ub_gep, align 8, !dbg !15
; CHECK-NEXT:   %ub5 = trunc i64 %2 to i32, !dbg !15
; CHECK-NEXT:   %lb.tmp.addr = alloca i32, align 4, !dbg !15
; CHECK-NEXT:   store i32 %lb.value, i32* %lb.tmp.addr, align 4, !dbg !15
; CHECK-NEXT:   %3 = load i32, i32* %lb.tmp.addr, align 4, !dbg !15
; CHECK-NEXT:   %ub.tmp.addr = alloca i32, align 4, !dbg !15
; CHECK-NEXT:   store i32 %ub.value, i32* %ub.tmp.addr, align 4, !dbg !15
; CHECK-NEXT:   %4 = load i32, i32* %ub.tmp.addr, align 4, !dbg !15
; CHECK-NEXT:   %step.tmp.addr = alloca i32, align 4, !dbg !15
; CHECK-NEXT:   store i32 1, i32* %step.tmp.addr, align 4, !dbg !15
; CHECK-NEXT:   %5 = load i32, i32* %step.tmp.addr, align 4, !dbg !15
; CHECK-NEXT:   %loop.i.addr = alloca i32, align 4, !dbg !15
; CHECK-NEXT:   store i32 %lb4, i32* %loop.i.addr, align 4, !dbg !15
; CHECK-NEXT:   br label %for.cond1, !dbg !15
; CHECK: for.cond1:                                        ; preds = %for.incr3, %0
; CHECK-NEXT:   %6 = load i32, i32* %loop.i.addr, align 4, !dbg !15
; CHECK-NEXT:   %7 = icmp slt i32 %6, %ub5, !dbg !15
; CHECK-NEXT:   br i1 %7, label %for.body2, label %.exitStub, !dbg !15
; CHECK: for.body2:                                        ; preds = %for.cond1
; CHECK-NEXT:   %8 = load i32, i32* %loop.i.addr, align 4, !dbg !16
; CHECK-NEXT:   %9 = mul i32 %8, %5, !dbg !16
; CHECK-NEXT:   %10 = add i32 %9, %3, !dbg !16
; CHECK-NEXT:   store i32 %10, i32* %i.addr, align 4, !dbg !16
; CHECK-NEXT:   %11 = load i32, i32* @sum, align 4, !dbg !16
; CHECK-NEXT:   %i = load i32, i32* %i.addr, align 4, !dbg !16
; CHECK-NEXT:   %add = add nsw i32 %11, %i, !dbg !16
; CHECK-NEXT:   store i32 %add, i32* @sum, align 4, !dbg !16
; CHECK-NEXT:   br label %for.incr3, !dbg !17
; CHECK: for.incr3:                                        ; preds = %for.body2
; CHECK-NEXT:   %12 = load i32, i32* %loop.i.addr, align 4, !dbg !15
; CHECK-NEXT:   %13 = add i32 %12, %5, !dbg !15
; CHECK-NEXT:   store i32 %13, i32* %loop.i.addr, align 4, !dbg !15
; CHECK-NEXT:   br label %for.cond1, !dbg !15
; CHECK-NEXT: }

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !12 {
entry:
  call void @taskloop(i32 -55, i32 31, i32 2), !dbg !13
  ret i32 0, !dbg !14
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "taskloop", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, column: 13, scope: !6)
!9 = !DILocation(line: 4, column: 9, scope: !6)
!10 = !DILocation(line: 4, column: 5, scope: !6)
!11 = !DILocation(line: 5, column: 1, scope: !6)
!12 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !7, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 7, column: 5, scope: !12)
!14 = !DILocation(line: 8, column: 1, scope: !12)
