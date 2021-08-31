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
define dso_local void @foo(i32 %lb, i32 %ub, i32 %step) #0 !dbg !6 {
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
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !10
  %1 = load i32, i32* %i, align 4, !dbg !11
  call void @llvm.directive.region.exit(token %0), !dbg !12
  store i32 5, i32* %i1, align 4, !dbg !13
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !14
  %3 = load i32, i32* %i1, align 4, !dbg !15
  call void @llvm.directive.region.exit(token %2), !dbg !16
  store i32 10, i32* %i2, align 4, !dbg !17
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.4), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.5), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.6), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 1, i64 1, i64 1, i64 1) ], !dbg !18
  %5 = load i32, i32* %i2, align 4, !dbg !19
  call void @llvm.directive.region.exit(token %4), !dbg !20
  store i32 10, i32* %i3, align 4, !dbg !21
  %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.7), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.8), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.9), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 1, i64 1, i64 1, i64 1) ], !dbg !22
  %7 = load i32, i32* %i3, align 4, !dbg !23
  call void @llvm.directive.region.exit(token %6), !dbg !24
  ret void, !dbg !25
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb() #2 !dbg !26 {
entry:
  ret i32 5, !dbg !27
}

define internal i32 @compute_ub() #2 !dbg !29 {
entry:
  ret i32 10, !dbg !30
}

define internal i32 @compute_step() #2 !dbg !32 {
entry:
  ret i32 1, !dbg !33
}

define internal i32 @compute_lb.1() #2 !dbg !35 {
entry:
  ret i32 5, !dbg !36
}

define internal i32 @compute_ub.2() #2 !dbg !38 {
entry:
  ret i32 10, !dbg !39
}

define internal i32 @compute_step.3() #2 !dbg !41 {
entry:
  ret i32 3, !dbg !42
}

define internal i32 @compute_lb.4() #2 !dbg !44 {
entry:
  ret i32 10, !dbg !45
}

define internal i32 @compute_ub.5() #2 !dbg !47 {
entry:
  ret i32 0, !dbg !48
}

define internal i32 @compute_step.6() #2 !dbg !50 {
entry:
  ret i32 -1, !dbg !51
}

define internal i32 @compute_lb.7() #2 !dbg !53 {
entry:
  ret i32 10, !dbg !54
}

define internal i32 @compute_ub.8() #2 !dbg !56 {
entry:
  ret i32 0, !dbg !57
}

define internal i32 @compute_step.9() #2 !dbg !59 {
entry:
  ret i32 -3, !dbg !60
}

; CHECK: define internal void @nanos6_unpacked_task_region_foo0(i32* %i, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %1 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb26 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %2 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub27 = trunc i64 %2 to i32
; CHECK-NEXT:   %3 = call i32 @compute_lb()
; CHECK-NEXT:   %4 = call i32 @compute_ub()
; CHECK-NEXT:   %5 = call i32 @compute_step()
; CHECK-NEXT:   %6 = sub i32 %ub27, %lb26
; CHECK-NEXT:   %7 = sub i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %loop = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb26, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond22
; CHECK: for.cond22:                                       ; preds = %for.incr25, %0
; CHECK-NEXT:   %11 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %12 = icmp slt i32 %11, %ub27
; CHECK-NEXT:   br i1 %12, label %13, label %.exitStub
; CHECK: 13:                                               ; preds = %for.cond22
; CHECK-NEXT:   %14 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %15 = sext i32 %14 to i64
; CHECK-NEXT:   %16 = udiv i64 %15, 1
; CHECK-NEXT:   %17 = sext i32 %5 to i64
; CHECK-NEXT:   %18 = mul i64 %16, %17
; CHECK-NEXT:   %19 = sext i32 %3 to i64
; CHECK-NEXT:   %20 = add i64 %18, %19
; CHECK-NEXT:   %21 = mul i64 %16, 1
; CHECK-NEXT:   %22 = sext i32 %14 to i64
; CHECK-NEXT:   %23 = sub i64 %22, %21
; CHECK-NEXT:   %24 = trunc i64 %20 to i32
; CHECK-NEXT:   store i32 %24, i32* %i, align 4
; CHECK-NEXT:   br label %for.body23
; CHECK: for.body23:                                       ; preds = %13
; CHECK-NEXT:   %25 = load i32, i32* %i, align 4
; CHECK-NEXT:   br label %for.incr25
; CHECK: for.incr25:                                       ; preds = %for.body23
; CHECK-NEXT:   %26 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %27 = add i32 %26, 1
; CHECK-NEXT:   store i32 %27, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond22
; CHECK: .exitStub:                                        ; preds = %for.cond22
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_foo1(i32* %i1, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %1 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb33 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %2 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub34 = trunc i64 %2 to i32
; CHECK-NEXT:   %3 = call i32 @compute_lb.1()
; CHECK-NEXT:   %4 = call i32 @compute_ub.2()
; CHECK-NEXT:   %5 = call i32 @compute_step.3()
; CHECK-NEXT:   %6 = sub i32 %ub34, %lb33
; CHECK-NEXT:   %7 = sub i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %loop = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb33, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond28
; CHECK: for.cond28:                                       ; preds = %for.incr31, %0
; CHECK-NEXT:   %11 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %12 = icmp slt i32 %11, %ub34
; CHECK-NEXT:   br i1 %12, label %13, label %.exitStub
; CHECK: 13:                                               ; preds = %for.cond28
; CHECK-NEXT:   %14 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %15 = sext i32 %14 to i64
; CHECK-NEXT:   %16 = udiv i64 %15, 1
; CHECK-NEXT:   %17 = sext i32 %5 to i64
; CHECK-NEXT:   %18 = mul i64 %16, %17
; CHECK-NEXT:   %19 = sext i32 %3 to i64
; CHECK-NEXT:   %20 = add i64 %18, %19
; CHECK-NEXT:   %21 = mul i64 %16, 1
; CHECK-NEXT:   %22 = sext i32 %14 to i64
; CHECK-NEXT:   %23 = sub i64 %22, %21
; CHECK-NEXT:   %24 = trunc i64 %20 to i32
; CHECK-NEXT:   store i32 %24, i32* %i1, align 4
; CHECK-NEXT:   br label %for.body29
; CHECK: for.body29:                                       ; preds = %13
; CHECK-NEXT:   %25 = load i32, i32* %i1, align 4
; CHECK-NEXT:   br label %for.incr31
; CHECK: for.incr31:                                       ; preds = %for.body29
; CHECK-NEXT:   %26 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %27 = add i32 %26, 1
; CHECK-NEXT:   store i32 %27, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond28
; CHECK: .exitStub:                                        ; preds = %for.cond28
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_foo2(i32* %i2, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %1 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb42 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %2 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub43 = trunc i64 %2 to i32
; CHECK-NEXT:   %3 = call i32 @compute_lb.4()
; CHECK-NEXT:   %4 = call i32 @compute_ub.5()
; CHECK-NEXT:   %5 = call i32 @compute_step.6()
; CHECK-NEXT:   %6 = sub i32 %ub43, %lb42
; CHECK-NEXT:   %7 = add i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %loop = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb42, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond37
; CHECK: for.cond37:                                       ; preds = %for.incr40, %0
; CHECK-NEXT:   %11 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %12 = icmp slt i32 %11, %ub43
; CHECK-NEXT:   br i1 %12, label %13, label %.exitStub
; CHECK: 13:                                               ; preds = %for.cond37
; CHECK-NEXT:   %14 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %15 = sext i32 %14 to i64
; CHECK-NEXT:   %16 = udiv i64 %15, 1
; CHECK-NEXT:   %17 = sext i32 %5 to i64
; CHECK-NEXT:   %18 = mul i64 %16, %17
; CHECK-NEXT:   %19 = sext i32 %3 to i64
; CHECK-NEXT:   %20 = add i64 %18, %19
; CHECK-NEXT:   %21 = mul i64 %16, 1
; CHECK-NEXT:   %22 = sext i32 %14 to i64
; CHECK-NEXT:   %23 = sub i64 %22, %21
; CHECK-NEXT:   %24 = trunc i64 %20 to i32
; CHECK-NEXT:   store i32 %24, i32* %i2, align 4
; CHECK-NEXT:   br label %for.body38
; CHECK: for.body38:                                       ; preds = %13
; CHECK-NEXT:   %25 = load i32, i32* %i2, align 4
; CHECK-NEXT:   br label %for.incr40
; CHECK: for.incr40:                                       ; preds = %for.body38
; CHECK-NEXT:   %26 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %27 = add i32 %26, 1
; CHECK-NEXT:   store i32 %27, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond37
; CHECK: .exitStub:                                        ; preds = %for.cond37
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_foo3(i32* %i3, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %1 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb51 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %2 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub52 = trunc i64 %2 to i32
; CHECK-NEXT:   %3 = call i32 @compute_lb.7()
; CHECK-NEXT:   %4 = call i32 @compute_ub.8()
; CHECK-NEXT:   %5 = call i32 @compute_step.9()
; CHECK-NEXT:   %6 = sub i32 %ub52, %lb51
; CHECK-NEXT:   %7 = add i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %loop = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb51, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond46
; CHECK: for.cond46:                                       ; preds = %for.incr49, %0
; CHECK-NEXT:   %11 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %12 = icmp slt i32 %11, %ub52
; CHECK-NEXT:   br i1 %12, label %13, label %.exitStub
; CHECK: 13:                                               ; preds = %for.cond46
; CHECK-NEXT:   %14 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %15 = sext i32 %14 to i64
; CHECK-NEXT:   %16 = udiv i64 %15, 1
; CHECK-NEXT:   %17 = sext i32 %5 to i64
; CHECK-NEXT:   %18 = mul i64 %16, %17
; CHECK-NEXT:   %19 = sext i32 %3 to i64
; CHECK-NEXT:   %20 = add i64 %18, %19
; CHECK-NEXT:   %21 = mul i64 %16, 1
; CHECK-NEXT:   %22 = sext i32 %14 to i64
; CHECK-NEXT:   %23 = sub i64 %22, %21
; CHECK-NEXT:   %24 = trunc i64 %20 to i32
; CHECK-NEXT:   store i32 %24, i32* %i3, align 4
; CHECK-NEXT:   br label %for.body47
; CHECK: for.body47:                                       ; preds = %13
; CHECK-NEXT:   %25 = load i32, i32* %i3, align 4
; CHECK-NEXT:   br label %for.incr49
; CHECK: for.incr49:                                       ; preds = %for.body47
; CHECK-NEXT:   %26 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %27 = add i32 %26, 1
; CHECK-NEXT:   store i32 %27, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond46
; CHECK: .exitStub:                                        ; preds = %for.cond46
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind }
attributes #2 = { "min-legal-vector-width"="0" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_for_unpack_loop.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 14, scope: !6)
!10 = !DILocation(line: 3, column: 10, scope: !6)
!11 = !DILocation(line: 3, column: 39, scope: !6)
!12 = !DILocation(line: 3, column: 42, scope: !6)
!13 = !DILocation(line: 5, column: 14, scope: !6)
!14 = !DILocation(line: 5, column: 10, scope: !6)
!15 = !DILocation(line: 5, column: 39, scope: !6)
!16 = !DILocation(line: 5, column: 42, scope: !6)
!17 = !DILocation(line: 7, column: 14, scope: !6)
!18 = !DILocation(line: 7, column: 10, scope: !6)
!19 = !DILocation(line: 7, column: 39, scope: !6)
!20 = !DILocation(line: 7, column: 42, scope: !6)
!21 = !DILocation(line: 9, column: 14, scope: !6)
!22 = !DILocation(line: 9, column: 10, scope: !6)
!23 = !DILocation(line: 9, column: 39, scope: !6)
!24 = !DILocation(line: 9, column: 42, scope: !6)
!25 = !DILocation(line: 10, column: 1, scope: !6)
!26 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 3, column: 18, scope: !28)
!28 = !DILexicalBlockFile(scope: !26, file: !7, discriminator: 0)
!29 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 3, column: 25, scope: !31)
!31 = !DILexicalBlockFile(scope: !29, file: !7, discriminator: 0)
!32 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 3, column: 34, scope: !34)
!34 = !DILexicalBlockFile(scope: !32, file: !7, discriminator: 0)
!35 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!36 = !DILocation(line: 5, column: 18, scope: !37)
!37 = !DILexicalBlockFile(scope: !35, file: !7, discriminator: 0)
!38 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!39 = !DILocation(line: 5, column: 25, scope: !40)
!40 = !DILexicalBlockFile(scope: !38, file: !7, discriminator: 0)
!41 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!42 = !DILocation(line: 5, column: 34, scope: !43)
!43 = !DILexicalBlockFile(scope: !41, file: !7, discriminator: 0)
!44 = distinct !DISubprogram(linkageName: "compute_lb.4", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!45 = !DILocation(line: 7, column: 18, scope: !46)
!46 = !DILexicalBlockFile(scope: !44, file: !7, discriminator: 0)
!47 = distinct !DISubprogram(linkageName: "compute_ub.5", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!48 = !DILocation(line: 7, column: 26, scope: !49)
!49 = !DILexicalBlockFile(scope: !47, file: !7, discriminator: 0)
!50 = distinct !DISubprogram(linkageName: "compute_step.6", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!51 = !DILocation(line: 7, column: 34, scope: !52)
!52 = !DILexicalBlockFile(scope: !50, file: !7, discriminator: 0)
!53 = distinct !DISubprogram(linkageName: "compute_lb.7", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!54 = !DILocation(line: 9, column: 18, scope: !55)
!55 = !DILexicalBlockFile(scope: !53, file: !7, discriminator: 0)
!56 = distinct !DISubprogram(linkageName: "compute_ub.8", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!57 = !DILocation(line: 9, column: 26, scope: !58)
!58 = !DILexicalBlockFile(scope: !56, file: !7, discriminator: 0)
!59 = distinct !DISubprogram(linkageName: "compute_step.9", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!60 = !DILocation(line: 9, column: 34, scope: !61)
!61 = !DILexicalBlockFile(scope: !59, file: !7, discriminator: 0)
