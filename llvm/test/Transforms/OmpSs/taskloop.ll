; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'taskloop.ll'
source_filename = "taskloop.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int sum = 0;
;
; void taskloop(int lb, int ub, int step) {
;   int i;
;   #pragma oss taskloop
;   for (i = lb; i < ub; i += step)
;     sum += i;
; }
;
; int main() {
;   taskloop(-55, 31, 2);
; }

@sum = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone
define dso_local void @taskloop(i32 %lb, i32 %ub, i32 %step) #0 !dbg !6 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %lb.addr, align 4, !dbg !9
  store i32 %0, i32* %i, align 4, !dbg !10
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb, i32* %lb.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub, i32* %ub.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !11
  %2 = load i32, i32* %i, align 4, !dbg !12
  %3 = load i32, i32* @sum, align 4, !dbg !13
  %add = add nsw i32 %3, %2, !dbg !13
  store i32 %add, i32* @sum, align 4, !dbg !13
  call void @llvm.directive.region.exit(token %1), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb(i32* %lb) #2 !dbg !16 {
entry:
  %lb.addr = alloca i32*, align 8
  store i32* %lb, i32** %lb.addr, align 8
  %0 = load i32, i32* %lb, align 4, !dbg !17
  ret i32 %0, !dbg !17
}

define internal i32 @compute_ub(i32* %ub) #2 !dbg !19 {
entry:
  %ub.addr = alloca i32*, align 8
  store i32* %ub, i32** %ub.addr, align 8
  %0 = load i32, i32* %ub, align 4, !dbg !20
  ret i32 %0, !dbg !20
}

define internal i32 @compute_step(i32* %step) #2 !dbg !22 {
entry:
  %step.addr = alloca i32*, align 8
  store i32* %step, i32** %step.addr, align 8
  %0 = load i32, i32* %step, align 4, !dbg !23
  ret i32 %0, !dbg !23
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !25 {
entry:
  call void @taskloop(i32 -55, i32 31, i32 2), !dbg !26
  ret i32 0, !dbg !27
}

; CHECK: define internal void @nanos6_unpacked_task_region_taskloop0(i32* %sum, i32* %i, i32* %lb.addr, i32* %ub.addr, i32* %step.addr, %nanos6_loop_bounds_t* %loop_bounds, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %0
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %1 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb5 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %2 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %ub6 = trunc i64 %2 to i32
; CHECK-NEXT:   %3 = call i32 @compute_lb(i32* %lb.addr)
; CHECK-NEXT:   %4 = call i32 @compute_ub(i32* %ub.addr)
; CHECK-NEXT:   %5 = call i32 @compute_step(i32* %step.addr)
; CHECK-NEXT:   %6 = sub i32 %ub6, %lb5
; CHECK-NEXT:   %7 = sub i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %loop = alloca i32, align 4
; CHECK-NEXT:   store i32 %lb5, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK: for.cond1:                                        ; preds = %for.incr4, %0
; CHECK-NEXT:   %11 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %12 = icmp slt i32 %11, %ub6
; CHECK-NEXT:   br i1 %12, label %13, label %.exitStub
; CHECK: 13:                                               ; preds = %for.cond1
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
; CHECK-NEXT:   br label %for.body2
; CHECK: for.body2:                                        ; preds = %13
; CHECK-NEXT:   %25 = load i32, i32* %i, align 4
; CHECK-NEXT:   %26 = load i32, i32* %sum, align 4
; CHECK-NEXT:   %add = add nsw i32 %26, %25
; CHECK-NEXT:   store i32 %add, i32* %sum, align 4
; CHECK-NEXT:   br label %for.incr4
; CHECK: for.incr4:                                        ; preds = %for.body2
; CHECK-NEXT:   %27 = load i32, i32* %loop, align 4
; CHECK-NEXT:   %28 = add i32 %27, 1
; CHECK-NEXT:   store i32 %28, i32* %loop, align 4
; CHECK-NEXT:   br label %for.cond1
; CHECK: .exitStub:                                        ; preds = %for.cond1
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind }
attributes #2 = { "min-legal-vector-width"="0" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "taskloop", scope: !7, file: !7, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "taskloop.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 6, column: 15, scope: !6)
!10 = !DILocation(line: 6, column: 13, scope: !6)
!11 = !DILocation(line: 6, column: 11, scope: !6)
!12 = !DILocation(line: 7, column: 13, scope: !6)
!13 = !DILocation(line: 7, column: 10, scope: !6)
!14 = !DILocation(line: 7, column: 6, scope: !6)
!15 = !DILocation(line: 8, column: 2, scope: !6)
!16 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!17 = !DILocation(line: 6, column: 15, scope: !18)
!18 = !DILexicalBlockFile(scope: !16, file: !7, discriminator: 0)
!19 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 6, column: 23, scope: !21)
!21 = !DILexicalBlockFile(scope: !19, file: !7, discriminator: 0)
!22 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!23 = !DILocation(line: 6, column: 32, scope: !24)
!24 = !DILexicalBlockFile(scope: !22, file: !7, discriminator: 0)
!25 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 10, type: !8, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 11, column: 6, scope: !25)
!27 = !DILocation(line: 12, column: 2, scope: !25)
