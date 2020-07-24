; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_for_cond_and_register_check.ll'
source_filename = ""
target datalayout = ""
target triple = "x86_64-unknown-linux-gnu"

; void signed_loop_slt(int lb, int ub, int step) {
;     #pragma oss task for
;     for (int i = lb; i < ub; i += step) {}
; }
; void signed_loop_sle(int lb, int ub, int step) {
;     #pragma oss task for
;     for (int i = lb; i <= ub; i += step) {}
; }
; void signed_loop_sgt(int lb, int ub, int step) {
;     #pragma oss task for
;     for (int i = ub; i > lb; i -= step) {}
; }
; void signed_loop_sge(int lb, int ub, int step) {
;     #pragma oss task for
;     for (int i = ub; i >= lb; i -= step) {}
; }
; void unsigned_loop_slt(unsigned lb, unsigned ub, unsigned step) {
;     #pragma oss task for
;     for (unsigned i = lb; i < ub; i += step) {}
; }
; void unsigned_loop_sle(unsigned lb, unsigned ub, unsigned step) {
;     #pragma oss task for
;     for (unsigned i = lb; i <= ub; i += step) {}
; }
; void unsigned_loop_sgt(unsigned lb, unsigned ub, unsigned step) {
;     #pragma oss task for
;     for (unsigned i = ub; i > lb; i -= step) {}
; }
; void unsigned_loop_sge(unsigned lb, unsigned ub, unsigned step) {
;     #pragma oss task for
;     for (unsigned i = ub; i >= lb; i -= step) {}
; }
; void constants_loop() {
;     #pragma oss task for
;     for (int i = 0; i < 10; i += 1) {}
; }

; Function Attrs: noinline nounwind optnone
define void @signed_loop_slt(i32 %lb, i32 %ub, i32 %step) #0 !dbg !6 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %lb.addr, align 4, !dbg !9
  store i32 %0, i32* %i, align 4, !dbg !9
  %1 = load i32, i32* %lb.addr, align 4, !dbg !9
  %2 = load i32, i32* %ub.addr, align 4, !dbg !9
  %3 = load i32, i32* %step.addr, align 4, !dbg !9
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !9
  call void @llvm.directive.region.exit(token %4), !dbg !9
  ret void, !dbg !10
; CHECK-LABEL: @signed_loop_slt(
; CHECK: %15 = load i8*, i8** %6, align 8
; CHECK-NEXT: %16 = sub i32 %2, %1
; CHECK-NEXT: %17 = sub i32 %16, 1
; CHECK-NEXT: %18 = sdiv i32 %17, %3
; CHECK-NEXT: %19 = add i32 %18, 1
; CHECK-NEXT: %20 = sext i32 %19 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %15, i64 0, i64 %20, i64 0, i64 0)
; CHECK: %22 = icmp slt i32 %21, %2
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

; Function Attrs: noinline nounwind optnone
define void @signed_loop_sle(i32 %lb, i32 %ub, i32 %step) #0 !dbg !11 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %lb.addr, align 4, !dbg !12
  store i32 %0, i32* %i, align 4, !dbg !12
  %1 = load i32, i32* %lb.addr, align 4, !dbg !12
  %2 = load i32, i32* %ub.addr, align 4, !dbg !12
  %3 = load i32, i32* %step.addr, align 4, !dbg !12
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !12
  call void @llvm.directive.region.exit(token %4), !dbg !12
  ret void, !dbg !13
; CHECK-LABEL: @signed_loop_sle(
; CHECK: %15 = load i8*, i8** %6, align 8
; CHECK-NEXT: %16 = sub i32 %2, %1
; CHECK-NEXT: %17 = sdiv i32 %16, %3
; CHECK-NEXT: %18 = add i32 %17, 1
; CHECK-NEXT: %19 = sext i32 %18 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %15, i64 0, i64 %19, i64 0, i64 0)
; CHECK: %21 = icmp sle i32 %20, %2
}

; Function Attrs: noinline nounwind optnone
define void @signed_loop_sgt(i32 %lb, i32 %ub, i32 %step) #0 !dbg !14 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %ub.addr, align 4, !dbg !15
  store i32 %0, i32* %i, align 4, !dbg !15
  %1 = load i32, i32* %ub.addr, align 4, !dbg !15
  %2 = load i32, i32* %ub.addr, align 4, !dbg !15
  %3 = load i32, i32* %step.addr, align 4, !dbg !15
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !15
  call void @llvm.directive.region.exit(token %4), !dbg !15
  ret void, !dbg !16
; CHECK-LABEL: @signed_loop_sgt(
; CHECK: %13 = load i8*, i8** %6, align 8
; CHECK-NEXT: %14 = sub i32 %2, %1
; CHECK-NEXT: %15 = add i32 %14, 1
; CHECK-NEXT: %16 = sdiv i32 %15, %3
; CHECK-NEXT: %17 = add i32 %16, 1
; CHECK-NEXT: %18 = sext i32 %17 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %13, i64 0, i64 %18, i64 0, i64 0)
; CHECK: %20 = icmp sgt i32 %19, %2
}

; Function Attrs: noinline nounwind optnone
define void @signed_loop_sge(i32 %lb, i32 %ub, i32 %step) #0 !dbg !17 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %ub.addr, align 4, !dbg !18
  store i32 %0, i32* %i, align 4, !dbg !18
  %1 = load i32, i32* %ub.addr, align 4, !dbg !18
  %2 = load i32, i32* %ub.addr, align 4, !dbg !18
  %3 = load i32, i32* %step.addr, align 4, !dbg !18
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !18
  call void @llvm.directive.region.exit(token %4), !dbg !18
  ret void, !dbg !19
; CHECK-LABEL: @signed_loop_sge(
; CHECK: %13 = load i8*, i8** %6, align 8
; CHECK-NEXT: %14 = sub i32 %2, %1
; CHECK-NEXT: %15 = sdiv i32 %14, %3
; CHECK-NEXT: %16 = add i32 %15, 1
; CHECK-NEXT: %17 = sext i32 %16 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %13, i64 0, i64 %17, i64 0, i64 0)
; CHECK: %19 = icmp sge i32 %18, %2
}

; Function Attrs: noinline nounwind optnone
define void @unsigned_loop_slt(i32 %lb, i32 %ub, i32 %step) #0 !dbg !20 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %lb.addr, align 4, !dbg !21
  store i32 %0, i32* %i, align 4, !dbg !21
  %1 = load i32, i32* %lb.addr, align 4, !dbg !21
  %2 = load i32, i32* %ub.addr, align 4, !dbg !21
  %3 = load i32, i32* %step.addr, align 4, !dbg !21
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !21
  call void @llvm.directive.region.exit(token %4), !dbg !21
  ret void, !dbg !22
; CHECK-LABEL: @unsigned_loop_slt(
; CHECK: %15 = load i8*, i8** %6, align 8
; CHECK-NEXT: %16 = sub i32 %2, %1
; CHECK-NEXT: %17 = sub i32 %16, 1
; CHECK-NEXT: %18 = udiv i32 %17, %3
; CHECK-NEXT: %19 = add i32 %18, 1
; CHECK-NEXT: %20 = zext i32 %19 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %15, i64 0, i64 %20, i64 0, i64 0)
; CHECK: %22 = icmp ult i32 %21, %2
}

; Function Attrs: noinline nounwind optnone
define void @unsigned_loop_sle(i32 %lb, i32 %ub, i32 %step) #0 !dbg !23 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %lb.addr, align 4, !dbg !24
  store i32 %0, i32* %i, align 4, !dbg !24
  %1 = load i32, i32* %lb.addr, align 4, !dbg !24
  %2 = load i32, i32* %ub.addr, align 4, !dbg !24
  %3 = load i32, i32* %step.addr, align 4, !dbg !24
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !24
  call void @llvm.directive.region.exit(token %4), !dbg !24
  ret void, !dbg !25
; CHECK-LABEL: @unsigned_loop_sle(
; CHECK: %15 = load i8*, i8** %6, align 8
; CHECK-NEXT: %16 = sub i32 %2, %1
; CHECK-NEXT: %17 = udiv i32 %16, %3
; CHECK-NEXT: %18 = add i32 %17, 1
; CHECK-NEXT: %19 = zext i32 %18 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %15, i64 0, i64 %19, i64 0, i64 0)
; CHECK: %21 = icmp ule i32 %20, %2
}

; Function Attrs: noinline nounwind optnone
define void @unsigned_loop_sgt(i32 %lb, i32 %ub, i32 %step) #0 !dbg !26 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %ub.addr, align 4, !dbg !27
  store i32 %0, i32* %i, align 4, !dbg !27
  %1 = load i32, i32* %ub.addr, align 4, !dbg !27
  %2 = load i32, i32* %ub.addr, align 4, !dbg !27
  %3 = load i32, i32* %step.addr, align 4, !dbg !27
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !27
  call void @llvm.directive.region.exit(token %4), !dbg !27
  ret void, !dbg !28
; CHECK-LABEL: @unsigned_loop_sgt(
; CHECK: %13 = load i8*, i8** %6, align 8
; CHECK-NEXT: %14 = sub i32 %2, %1
; CHECK-NEXT: %15 = add i32 %14, 1
; CHECK-NEXT: %16 = udiv i32 %15, %3
; CHECK-NEXT: %17 = add i32 %16, 1
; CHECK-NEXT: %18 = zext i32 %17 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %13, i64 0, i64 %18, i64 0, i64 0)
; CHECK: %20 = icmp ugt i32 %19, %2
}

; Function Attrs: noinline nounwind optnone
define void @unsigned_loop_sge(i32 %lb, i32 %ub, i32 %step) #0 !dbg !29 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  %0 = load i32, i32* %ub.addr, align 4, !dbg !30
  store i32 %0, i32* %i, align 4, !dbg !30
  %1 = load i32, i32* %ub.addr, align 4, !dbg !30
  %2 = load i32, i32* %ub.addr, align 4, !dbg !30
  %3 = load i32, i32* %step.addr, align 4, !dbg !30
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !30
  call void @llvm.directive.region.exit(token %4), !dbg !30
  ret void, !dbg !31
; CHECK-LABEL: @unsigned_loop_sge(
; CHECK: %13 = load i8*, i8** %6, align 8
; CHECK-NEXT: %14 = sub i32 %2, %1
; CHECK-NEXT: %15 = udiv i32 %14, %3
; CHECK-NEXT: %16 = add i32 %15, 1
; CHECK-NEXT: %17 = zext i32 %16 to i64
; CHECK-NEXT: call void @nanos6_register_loop_bounds(i8* %13, i64 0, i64 %17, i64 0, i64 0)
; CHECK: %19 = icmp uge i32 %18, %2
}

; Function Attrs: noinline nounwind optnone
define void @constants_loop() #0 !dbg !32 {
entry:
  %i = alloca i32, align 4
  store i32 0, i32* %i, align 4, !dbg !33
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !33
  call void @llvm.directive.region.exit(token %0), !dbg !33
  ret void, !dbg !34
; CHECK-LABEL: @constants_loop(
; CHECK: call void @nanos6_register_loop_bounds(i8* %5, i64 0, i64 10, i64 0, i64 0)
; CHECK: %7 = icmp slt i32 %6, 10
}

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
!6 = distinct !DISubprogram(name: "signed_loop_slt", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_for_cond_and_register_check.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, scope: !6)
!10 = !DILocation(line: 4, scope: !6)
!11 = distinct !DISubprogram(name: "signed_loop_sle", scope: !7, file: !7, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 7, scope: !11)
!13 = !DILocation(line: 8, scope: !11)
!14 = distinct !DISubprogram(name: "signed_loop_sgt", scope: !7, file: !7, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 11, scope: !14)
!16 = !DILocation(line: 12, scope: !14)
!17 = distinct !DISubprogram(name: "signed_loop_sge", scope: !7, file: !7, line: 13, type: !8, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 15, scope: !17)
!19 = !DILocation(line: 16, scope: !17)
!20 = distinct !DISubprogram(name: "unsigned_loop_slt", scope: !7, file: !7, line: 18, type: !8, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 20, scope: !20)
!22 = !DILocation(line: 21, scope: !20)
!23 = distinct !DISubprogram(name: "unsigned_loop_sle", scope: !7, file: !7, line: 22, type: !8, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 24, scope: !23)
!25 = !DILocation(line: 25, scope: !23)
!26 = distinct !DISubprogram(name: "unsigned_loop_sgt", scope: !7, file: !7, line: 26, type: !8, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 28, scope: !26)
!28 = !DILocation(line: 29, scope: !26)
!29 = distinct !DISubprogram(name: "unsigned_loop_sge", scope: !7, file: !7, line: 30, type: !8, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 32, scope: !29)
!31 = !DILocation(line: 33, scope: !29)
!32 = distinct !DISubprogram(name: "constants_loop", scope: !7, file: !7, line: 35, type: !8, scopeLine: 35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 37, scope: !32)
!34 = !DILocation(line: 38, scope: !32)
