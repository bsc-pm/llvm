; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 't2.c'
source_filename = "t2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Checking only dependencies

; NOTE: this test has been typed by hand from an 'oss task' example
; NOTE: compute_deps has been typed by hand to accept lb and ub
; #include <assert.h>
; int main() {
;     int array[100];
;     int i;
;     #pragma oss task out(array[i])
;     // for (i = 0; i < 100; ++i) {
;         array[i] = i;
;     // }
;     #pragma oss task in(array[i])
;     // for (i = 0; i < 100; ++i) {
;         assert(array[i] == i);
;     // }
;     #pragma oss taskwait
; }

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }
%struct._depend_unpack_t.0 = type { i32*, i64, i64, i64 }

@.str = private unnamed_addr constant [14 x i8] c"array[i] == i\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"t2.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %retval = alloca i32, align 4
  %array = alloca [100 x i32], align 16
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4

  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4

  store i32 0, i32* %lb.addr, align 4
  store i32 100, i32* %ub.addr, align 4
  store i32 1, i32* %step.addr, align 4

  %lb.value = load i32, i32* %lb.addr, align 4
  %ub.value = load i32, i32* %ub.addr, align 4
  %step.value = load i32, i32* %step.addr, align 4

  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"),
      "QUAL.OSS.SHARED"([100 x i32]* %array),
        "QUAL.OSS.PRIVATE"(i32* %i),
        "QUAL.OSS.CAPTURED"(i32 %lb.value, i32 %ub.value, i32 %step.value),
        "QUAL.OSS.LOOP.TYPE"(i32 0),
        "QUAL.OSS.LOOP.IND.VAR"(i32* %i),
        "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %lb.value),
        "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %ub.value),
        "QUAL.OSS.LOOP.STEP"(i32 %step.value),
      "QUAL.OSS.DEP.OUT"([100 x i32]* %array, %struct._depend_unpack_t ([100 x i32]*, i32, i32)* @compute_dep, [100 x i32]* %array, i32 %lb.value, i32 %ub.value) ], !dbg !8
  %1 = load i32, i32* %i, align 4, !dbg !9
  %2 = load i32, i32* %i, align 4, !dbg !10
  %idxprom = sext i32 %2 to i64, !dbg !11
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 %idxprom, !dbg !11
  store i32 %1, i32* %arrayidx, align 4, !dbg !12

  call void @llvm.directive.region.exit(token %0), !dbg !11


  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"),
      "QUAL.OSS.SHARED"([100 x i32]* %array),
        "QUAL.OSS.PRIVATE"(i32* %i),
        "QUAL.OSS.CAPTURED"(i32 %lb.value, i32 %ub.value, i32 %step.value),
        "QUAL.OSS.LOOP.TYPE"(i32 0),
        "QUAL.OSS.LOOP.IND.VAR"(i32* %i),
        "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %lb.value),
        "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %ub.value),
        "QUAL.OSS.LOOP.STEP"(i32 %step.value),
      "QUAL.OSS.DEP.IN"([100 x i32]* %array, %struct._depend_unpack_t.0 ([100 x i32]*, i32, i32)* @compute_dep.1, [100 x i32]* %array, i32 %lb.value, i32 %ub.value) ], !dbg !13
  %4 = load i32, i32* %i, align 4, !dbg !14
  %idxprom1 = sext i32 %4 to i64, !dbg !14
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 %idxprom1, !dbg !14
  %5 = load i32, i32* %arrayidx2, align 4, !dbg !14
  %6 = load i32, i32* %i, align 4, !dbg !14
  %cmp = icmp eq i32 %5, %6, !dbg !14
  br i1 %cmp, label %if.then, label %if.else, !dbg !14

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !14

if.else:                                          ; preds = %entry
  call void @__assert_fail(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.2, i64 0, i64 0), i32 11, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #3, !dbg !14
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.directive.region.exit(token %3), !dbg !14


  %7 = call i1 @llvm.directive.marker() [ "DIR.OSS"([9 x i8] c"TASKWAIT\00") ], !dbg !15
  %8 = load i32, i32* %retval, align 4, !dbg !16
  ret i32 %8, !dbg !16
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32 %lb, i32 %ub) {
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %lb.1 = sext i32 %lb to i64
  %ub.1 = add i32 %ub, 1
  %ub.2 = sext i32 %ub.1 to i64
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 0, !dbg !13
  %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store i32* %arraydecay, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 400, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 %lb.1, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 %ub.2, i64* %3, align 8
  %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %4
}

define internal %struct._depend_unpack_t.0 @compute_dep.1([100 x i32]* %array, i32 %lb, i32 %ub) {
entry:
  %return.val = alloca %struct._depend_unpack_t.0, align 8
  %lb.1 = sext i32 %lb to i64
  %ub.1 = add i32 %ub, 1
  %ub.2 = sext i32 %ub.1 to i64
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 0, !dbg !13
  %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 0
  store i32* %arraydecay, i32** %0, align 8
  %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 1
  store i64 400, i64* %1, align 8
  %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 2
  store i64 %lb.1, i64* %2, align 8
  %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, i32 0, i32 3
  store i64 %ub.2, i64* %3, align 8
  %4 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %return.val, align 8
  ret %struct._depend_unpack_t.0 %4
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([100 x i32]* %array, i32* %i, i32 %lb.value, i32 %ub.value, i32 %step.value, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %0 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %0 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %1 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %2 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub = sub i32 %2, 1
; CHECK-NEXT:   %3 = call %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32 %lb, i32 %ub)
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t %3, 0
; CHECK-NEXT:   %5 = bitcast i32* %4 to i8*
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t %3, 1
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t %3, 2
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t %3, 3
; CHECK-NEXT:   call void @nanos6_register_region_write_depinfo1(i8* %handler, i32 0, i8* null, i8* %5, i64 %6, i64 %7, i64 %8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; CHECK: define internal void @nanos6_unpacked_deps_main1([100 x i32]* %array, i32* %i, i32 %lb.value, i32 %ub.value, i32 %step.value, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %0 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %0 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %1 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %2 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub = sub i32 %2, 1
; CHECK-NEXT:   %3 = call %struct._depend_unpack_t.0 @compute_dep.1([100 x i32]* %array, i32 %lb, i32 %ub)
; CHECK-NEXT:   %4 = extractvalue %struct._depend_unpack_t.0 %3, 0
; CHECK-NEXT:   %5 = bitcast i32* %4 to i8*
; CHECK-NEXT:   %6 = extractvalue %struct._depend_unpack_t.0 %3, 1
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t.0 %3, 2
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t.0 %3, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %5, i64 %6, i64 %7, i64 %8)
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

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t2.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 5, column: 13, scope: !6)
!9 = !DILocation(line: 7, column: 20, scope: !6)
!10 = !DILocation(line: 7, column: 15, scope: !6)
!11 = !DILocation(line: 7, column: 9, scope: !6)
!12 = !DILocation(line: 7, column: 18, scope: !6)
!13 = !DILocation(line: 9, column: 13, scope: !6)
!14 = !DILocation(line: 11, column: 9, scope: !6)
!15 = !DILocation(line: 13, column: 13, scope: !6)
!16 = !DILocation(line: 14, column: 1, scope: !6)
!17 = !DILocation(line: 5, column: 32, scope: !6)
!18 = !DILocation(line: 5, column: 26, scope: !6)
!19 = !DILocation(line: 9, column: 31, scope: !6)
!20 = !DILocation(line: 9, column: 25, scope: !6)
