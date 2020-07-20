; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_for_unpack_loop.ll'
source_filename = "task_for_unpack_loop.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo(int lb, int ub, int step) {
;     #pragma oss task for
;     for (int i = 5; i < 10; i += 1) { i; }
;     #pragma oss task for
;     for (int i = 5; i < 10; i += 3) { i; }
;     #pragma oss task for
;     for (int i = 10; i > 0; i -= 1) { i; }
;     #pragma oss task for
;     for (int i = 10; i > 0; i -= 3) { i; }
; }

; Function Attrs: noinline nounwind optnone
define void @foo(i32 %lb, i32 %ub, i32 %step) #0 !dbg !6 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %i3 = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  store i32 5, i32* %i, align 4, !dbg !9
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 5), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 5, i32 10, i32 1) ], !dbg !9
  %1 = load i32, i32* %i, align 4, !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !9
  store i32 5, i32* %i1, align 4, !dbg !10
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 5), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 3), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 5, i32 10, i32 3) ], !dbg !10
  %3 = load i32, i32* %i1, align 4, !dbg !10
  call void @llvm.directive.region.exit(token %2), !dbg !10
  store i32 10, i32* %i2, align 4, !dbg !11
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 10), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 0), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 2), "QUAL.OSS.CAPTURED"(i32 10, i32 0, i32 1) ], !dbg !11
  %5 = load i32, i32* %i2, align 4, !dbg !11
  call void @llvm.directive.region.exit(token %4), !dbg !11
  store i32 10, i32* %i3, align 4, !dbg !12
  %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 10), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 0), "QUAL.OSS.LOOP.STEP"(i32 3), "QUAL.OSS.LOOP.TYPE"(i64 2), "QUAL.OSS.CAPTURED"(i32 10, i32 0, i32 3) ], !dbg !12
  %7 = load i32, i32* %i3, align 4, !dbg !12
  call void @llvm.directive.region.exit(token %6), !dbg !12
  ret void, !dbg !13
}

; CHECK: define internal void @nanos6_unpacked_task_region_foo0(i32* %i, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table) !dbg !14 {
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %3
; CHECK: .exitStub:                                        ; preds = %for.cond1
; CHECK-NEXT:   ret void
; CHECK: 3:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %4 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb4 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %5 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub5 = trunc i64 %5 to i32
; CHECK-NEXT:   %lb.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 5, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %6 = load i32, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %ub.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 10, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %7 = load i32, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %step.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 1, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %8 = load i32, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %loop.i = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb4, i32* %loop.i, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK: for.cond1:                                        ; preds = %for.incr3, %3
; CHECK-NEXT:   %9 = load i32, i32* %loop.i, align 4
; CHECK-NEXT:   %10 = icmp slt i32 %9, %ub5
; CHECK-NEXT:   br i1 %10, label %for.body2, label %.exitStub
; CHECK: for.body2:                                        ; preds = %for.cond1
; CHECK-NEXT:   %11 = load i32, i32* %loop.i, align 4
; CHECK-NEXT:   %12 = mul i32 %11, %8
; CHECK-NEXT:   %13 = add i32 %12, %6
; CHECK-NEXT:   store i32 %13, i32* %i, align 4
; CHECK-NEXT:   %14 = load i32, i32* %i, align 4
; CHECK-NEXT:   br label %for.incr3
; CHECK: for.incr3:                                        ; preds = %for.body2
; CHECK-NEXT:   %15 = load i32, i32* %loop.i, align 4
; CHECK-NEXT:   %16 = add i32 %15, %8
; CHECK-NEXT:   store i32 %16, i32* %loop.i, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_foo1(i32* %i1, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table) !dbg !17 {
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %3
; CHECK: .exitStub:                                        ; preds = %for.cond14
; CHECK-NEXT:   ret void
; CHECK: 3:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %4 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb18 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %5 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub19 = trunc i64 %5 to i32
; CHECK-NEXT:   %lb.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 5, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %6 = load i32, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %ub.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 10, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %7 = load i32, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %step.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 1, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %8 = load i32, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %loop.i1 = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb18, i32* %loop.i1, align 4
; CHECK-NEXT:   br label %for.cond14
; CHECK: for.cond14:                                       ; preds = %for.incr16, %3
; CHECK-NEXT:   %9 = load i32, i32* %loop.i1, align 4
; CHECK-NEXT:   %10 = icmp slt i32 %9, %ub19
; CHECK-NEXT:   br i1 %10, label %for.body15, label %.exitStub
; CHECK: for.body15:                                       ; preds = %for.cond14
; CHECK-NEXT:   %11 = load i32, i32* %loop.i1, align 4
; CHECK-NEXT:   %12 = mul i32 %11, %8
; CHECK-NEXT:   %13 = add i32 %12, %6
; CHECK-NEXT:   store i32 %13, i32* %i1, align 4
; CHECK-NEXT:   %14 = load i32, i32* %i1, align 4
; CHECK-NEXT:   br label %for.incr16
; CHECK: for.incr16:                                       ; preds = %for.body15
; CHECK-NEXT:   %15 = load i32, i32* %loop.i1, align 4
; CHECK-NEXT:   %16 = add i32 %15, %8
; CHECK-NEXT:   store i32 %16, i32* %loop.i1, align 4
; CHECK-NEXT:   br label %for.cond14
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_foo2(i32* %i2, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table) !dbg !20 {
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %3
; CHECK: .exitStub:                                        ; preds = %for.cond30
; CHECK-NEXT:   ret void
; CHECK: 3:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %4 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb34 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %5 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub35 = trunc i64 %5 to i32
; CHECK-NEXT:   %lb.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 10, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %6 = load i32, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %ub.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 0, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %7 = load i32, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %step.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 1, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %8 = load i32, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %loop.i2 = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb34, i32* %loop.i2, align 4
; CHECK-NEXT:   br label %for.cond30
; CHECK: for.cond30:                                       ; preds = %for.incr32, %3
; CHECK-NEXT:   %9 = load i32, i32* %loop.i2, align 4
; CHECK-NEXT:   %10 = icmp slt i32 %9, %ub35
; CHECK-NEXT:   br i1 %10, label %for.body31, label %.exitStub
; CHECK: for.body31:                                       ; preds = %for.cond30
; CHECK-NEXT:   %11 = load i32, i32* %loop.i2, align 4
; CHECK-NEXT:   %12 = mul i32 %11, %8
; CHECK-NEXT:   %13 = add i32 %12, %6
; CHECK-NEXT:   store i32 %13, i32* %i2, align 4
; CHECK-NEXT:   %14 = load i32, i32* %i2, align 4
; CHECK-NEXT:   br label %for.incr32
; CHECK: for.incr32:                                       ; preds = %for.body31
; CHECK-NEXT:   %15 = load i32, i32* %loop.i2, align 4
; CHECK-NEXT:   %16 = add i32 %15, %8
; CHECK-NEXT:   store i32 %16, i32* %loop.i2, align 4
; CHECK-NEXT:   br label %for.cond30
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_foo3(i32* %i3, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table) !dbg !23 {
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %3
; CHECK: .exitStub:                                        ; preds = %for.cond46
; CHECK-NEXT:   ret void
; CHECK: 3:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %4 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb50 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %5 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub51 = trunc i64 %5 to i32
; CHECK-NEXT:   %lb.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 10, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %6 = load i32, i32* %lb.tmp.addr, align 4
; CHECK-NEXT:   %ub.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 0, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %7 = load i32, i32* %ub.tmp.addr, align 4
; CHECK-NEXT:   %step.tmp.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 1, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %8 = load i32, i32* %step.tmp.addr, align 4
; CHECK-NEXT:   %loop.i3 = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb50, i32* %loop.i3, align 4
; CHECK-NEXT:   br label %for.cond46
; CHECK: for.cond46:                                       ; preds = %for.incr48, %3
; CHECK-NEXT:   %9 = load i32, i32* %loop.i3, align 4
; CHECK-NEXT:   %10 = icmp slt i32 %9, %ub51
; CHECK-NEXT:   br i1 %10, label %for.body47, label %.exitStub
; CHECK: for.body47:                                       ; preds = %for.cond46
; CHECK-NEXT:   %11 = load i32, i32* %loop.i3, align 4
; CHECK-NEXT:   %12 = mul i32 %11, %8
; CHECK-NEXT:   %13 = add i32 %12, %6
; CHECK-NEXT:   store i32 %13, i32* %i3, align 4
; CHECK-NEXT:   %14 = load i32, i32* %i3, align 4
; CHECK-NEXT:   br label %for.incr48
; CHECK: for.incr48:                                       ; preds = %for.body47
; CHECK-NEXT:   %15 = load i32, i32* %loop.i3, align 4
; CHECK-NEXT:   %16 = add i32 %15, %8
; CHECK-NEXT:   store i32 %16, i32* %loop.i3, align 4
; CHECK-NEXT:   br label %for.cond46
; CHECK-NEXT: }

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_for_unpack_loop.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, scope: !6)
!10 = !DILocation(line: 5, scope: !6)
!11 = !DILocation(line: 7, scope: !6)
!12 = !DILocation(line: 9, scope: !6)
!13 = !DILocation(line: 10, scope: !6)
