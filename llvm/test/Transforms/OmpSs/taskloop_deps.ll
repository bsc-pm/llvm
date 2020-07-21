; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'taskloop_deps.ll'
source_filename = "taskloop_deps.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; NOTE: Since we call two times compute_dep we can replace uses of induction
; variable to use (nanos6.lb*step + lb) and (nanos6.ub*step + ub)

; #include <assert.h>
; int main() {
;     int array[100];
;     int i;
;     #pragma oss taskloop out(array[i])
;     for (i = 0; i < 100; ++i) {
;         array[i] = i;
;     }
;     #pragma oss taskloop in(array[i])
;     for (i = 0; i < 100; ++i) {
;         assert(array[i] == i);
;     }
;     #pragma oss taskwait
; }

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { i32*, i64, i64, i64 }

@.str = private unnamed_addr constant [14 x i8] c"array[i] == i\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"taskloop_deps.ll\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %retval = alloca i32, align 4
  %array = alloca [100 x i32], align 16
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4, !dbg !8
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([100 x i32]* %array), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 100), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 100, i32 1), "QUAL.OSS.DEP.OUT"([100 x i32]* %array, %struct._depend_unpack_t ([100 x i32]*, i32*)* @compute_dep, [100 x i32]* %array, i32* %i) ], !dbg !9
  %1 = load i32, i32* %i, align 4, !dbg !10
  %2 = load i32, i32* %i, align 4, !dbg !11
  %idxprom = sext i32 %2 to i64, !dbg !12
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 %idxprom, !dbg !12
  store i32 %1, i32* %arrayidx, align 4, !dbg !13
  call void @llvm.directive.region.exit(token %0), !dbg !14
  store i32 0, i32* %i, align 4, !dbg !15
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([100 x i32]* %array), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 100), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 100, i32 1), "QUAL.OSS.DEP.IN"([100 x i32]* %array, %struct._depend_unpack_t.0 ([100 x i32]*, i32*)* @compute_dep.1, [100 x i32]* %array, i32* %i) ], !dbg !16
  %4 = load i32, i32* %i, align 4, !dbg !17
  %idxprom1 = sext i32 %4 to i64, !dbg !17
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 %idxprom1, !dbg !17
  %5 = load i32, i32* %arrayidx2, align 4, !dbg !17
  %6 = load i32, i32* %i, align 4, !dbg !17
  %cmp = icmp eq i32 %5, %6, !dbg !17
  br i1 %cmp, label %if.then, label %if.else, !dbg !17

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !17

if.else:                                          ; preds = %entry
  call void @__assert_fail(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.2, i64 0, i64 0), i32 11, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #3, !dbg !17
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.directive.region.exit(token %3), !dbg !18
  %7 = call i1 @llvm.directive.marker() [ "DIR.OSS"([9 x i8] c"TASKWAIT\00") ], !dbg !19
  %8 = load i32, i32* %retval, align 4, !dbg !20
  ret i32 %8, !dbg !20
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32* %i) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %0 = load i32, i32* %i, align 4, !dbg !21
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 0, !dbg !22
  %3 = mul i64 %1, 4
  %4 = mul i64 %2, 4
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store i32* %arraydecay, i32** %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 400, i64* %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 %3, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 %4, i64* %8, align 8
  %9 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %9
}

define internal %struct._depend_unpack_t.0 @compute_dep.1([100 x i32]* %array, i32* %i) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %0 = load i32, i32* %i, align 4, !dbg !23
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 0, !dbg !24
  %3 = mul i64 %1, 4
  %4 = mul i64 %2, 4
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store i32* %arraydecay, i32** %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 400, i64* %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 %3, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 %4, i64* %8, align 8
  %9 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %9
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([100 x i32]* %array, i32* %i, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %3 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %4 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %5 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub = sub i32 %5, 1
; CHECK-NEXT:   %i.lb = alloca i32, align 4
; CHECK-NEXT:   %i.ub = alloca i32, align 4
; CHECK-NEXT:   %6 = mul nuw i32 1, %lb
; CHECK-NEXT:   %7 = add nuw i32 %6, 0
; CHECK-NEXT:   store i32 %7, i32* %i.lb, align 4
; CHECK-NEXT:   %8 = mul nuw i32 1, %ub
; CHECK-NEXT:   %9 = add nuw i32 %8, 0
; CHECK-NEXT:   store i32 %9, i32* %i.ub, align 4
; CHECK-NEXT:   %10 = call %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32* %i.lb)
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32* %i.ub)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t %10, 0
; CHECK-NEXT:   %13 = bitcast i32* %12 to i8*
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t %10, 1
; CHECK-NEXT:   %15 = extractvalue %struct._depend_unpack_t %10, 2
; CHECK-NEXT:   %16 = extractvalue %struct._depend_unpack_t %11, 3
; CHECK-NEXT:   call void @nanos6_register_region_write_depinfo1(i8* %handler, i32 0, i8* null, i8* %13, i64 %14, i64 %15, i64 %16)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main1([100 x i32]* %array, i32* %i, i32 %0, i32 %1, i32 %2, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %3 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %4 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %5 = trunc i64 %4 to i32
; CHECK-NEXT:   %ub = sub i32 %5, 1
; CHECK-NEXT:   %i.lb = alloca i32, align 4
; CHECK-NEXT:   %i.ub = alloca i32, align 4
; CHECK-NEXT:   %6 = mul nuw i32 1, %lb
; CHECK-NEXT:   %7 = add nuw i32 %6, 0
; CHECK-NEXT:   store i32 %7, i32* %i.lb, align 4
; CHECK-NEXT:   %8 = mul nuw i32 1, %ub
; CHECK-NEXT:   %9 = add nuw i32 %8, 0
; CHECK-NEXT:   store i32 %9, i32* %i.ub, align 4
; CHECK-NEXT:   %10 = call %struct._depend_unpack_t.0 @compute_dep.1([100 x i32]* %array, i32* %i.lb)
; CHECK-NEXT:   %11 = call %struct._depend_unpack_t.0 @compute_dep.1([100 x i32]* %array, i32* %i.ub)
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t.0 %10, 0
; CHECK-NEXT:   %13 = bitcast i32* %12 to i8*
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t.0 %10, 1
; CHECK-NEXT:   %15 = extractvalue %struct._depend_unpack_t.0 %10, 2
; CHECK-NEXT:   %16 = extractvalue %struct._depend_unpack_t.0 %11, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %13, i64 %14, i64 %15, i64 %16)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) #2

; Function Attrs: nounwind
declare i1 @llvm.directive.marker() #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "taskloop_deps.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 6, column: 12, scope: !6)
!9 = !DILocation(line: 6, column: 10, scope: !6)
!10 = !DILocation(line: 7, column: 20, scope: !6)
!11 = !DILocation(line: 7, column: 15, scope: !6)
!12 = !DILocation(line: 7, column: 9, scope: !6)
!13 = !DILocation(line: 7, column: 18, scope: !6)
!14 = !DILocation(line: 8, column: 5, scope: !6)
!15 = !DILocation(line: 10, column: 12, scope: !6)
!16 = !DILocation(line: 10, column: 10, scope: !6)
!17 = !DILocation(line: 11, column: 9, scope: !6)
!18 = !DILocation(line: 12, column: 5, scope: !6)
!19 = !DILocation(line: 13, column: 13, scope: !6)
!20 = !DILocation(line: 14, column: 1, scope: !6)
!21 = !DILocation(line: 5, column: 36, scope: !6)
!22 = !DILocation(line: 5, column: 30, scope: !6)
!23 = !DILocation(line: 9, column: 35, scope: !6)
!24 = !DILocation(line: 9, column: 29, scope: !6)
