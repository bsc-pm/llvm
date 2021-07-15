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
@.str.5 = private unnamed_addr constant [17 x i8] c"taskloop_deps.ll\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %array = alloca [100 x i32], align 16
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([100 x i32]* %array), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.DEP.OUT"([100 x i32]* %array, [9 x i8] c"array[i]\00", %struct._depend_unpack_t ([100 x i32]*, i32*)* @compute_dep, [100 x i32]* %array, i32* %i) ], !dbg !11
  %1 = load i32, i32* %i, align 4, !dbg !12
  %2 = load i32, i32* %i, align 4, !dbg !13
  %idxprom = sext i32 %2 to i64, !dbg !14
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 %idxprom, !dbg !14
  store i32 %1, i32* %arrayidx, align 4, !dbg !15
  call void @llvm.directive.region.exit(token %0), !dbg !16
  store i32 0, i32* %i, align 4, !dbg !17
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([100 x i32]* %array), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.DEP.IN"([100 x i32]* %array, [9 x i8] c"array[i]\00", %struct._depend_unpack_t.0 ([100 x i32]*, i32*)* @compute_dep.4, [100 x i32]* %array, i32* %i) ], !dbg !18
  %4 = load i32, i32* %i, align 4, !dbg !19
  %idxprom1 = sext i32 %4 to i64, !dbg !19
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 %idxprom1, !dbg !19
  %5 = load i32, i32* %arrayidx2, align 4, !dbg !19
  %6 = load i32, i32* %i, align 4, !dbg !19
  %cmp = icmp eq i32 %5, %6, !dbg !19
  br i1 %cmp, label %if.then, label %if.else, !dbg !19

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !19

if.else:                                          ; preds = %entry
  call void @__assert_fail(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.5, i64 0, i64 0), i32 11, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #4, !dbg !19
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.directive.region.exit(token %3), !dbg !20
  %7 = call i1 @llvm.directive.marker() [ "DIR.OSS"([9 x i8] c"TASKWAIT\00") ], !dbg !21
  %8 = load i32, i32* %retval, align 4, !dbg !22
  ret i32 %8, !dbg !22
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb() #2 !dbg !23 {
entry:
  ret i32 0, !dbg !24
}

define internal i32 @compute_ub() #2 !dbg !25 {
entry:
  ret i32 100, !dbg !26
}

define internal i32 @compute_step() #2 !dbg !27 {
entry:
  ret i32 1, !dbg !28
}

define internal %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32* %i) #2 !dbg !29 {
entry:
  %retval = alloca %struct._depend_unpack_t, align 8
  %array.addr = alloca [100 x i32]*, align 8
  %i.addr = alloca i32*, align 8
  store [100 x i32]* %array, [100 x i32]** %array.addr, align 8
  store i32* %i, i32** %i.addr, align 8
  %0 = load i32, i32* %i, align 4, !dbg !30
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 0, !dbg !31
  %3 = mul i64 %1, 4
  %4 = mul i64 %2, 4
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
  store i32* %arraydecay, i32** %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
  store i64 400, i64* %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
  store i64 %3, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
  store i64 %4, i64* %8, align 8
  %9 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8, !dbg !31
  ret %struct._depend_unpack_t %9, !dbg !31
}

define internal i32 @compute_lb.1() #2 !dbg !32 {
entry:
  ret i32 0, !dbg !33
}

define internal i32 @compute_ub.2() #2 !dbg !34 {
entry:
  ret i32 100, !dbg !35
}

define internal i32 @compute_step.3() #2 !dbg !36 {
entry:
  ret i32 1, !dbg !37
}

define internal %struct._depend_unpack_t.0 @compute_dep.4([100 x i32]* %array, i32* %i) #2 !dbg !38 {
entry:
  %retval = alloca %struct._depend_unpack_t.0, align 8
  %array.addr = alloca [100 x i32]*, align 8
  %i.addr = alloca i32*, align 8
  store [100 x i32]* %array, [100 x i32]** %array.addr, align 8
  store i32* %i, i32** %i.addr, align 8
  %0 = load i32, i32* %i, align 4, !dbg !39
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, 1
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %array, i64 0, i64 0, !dbg !40
  %3 = mul i64 %1, 4
  %4 = mul i64 %2, 4
  %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
  store i32* %arraydecay, i32** %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
  store i64 400, i64* %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
  store i64 %3, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
  store i64 %4, i64* %8, align 8
  %9 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8, !dbg !40
  ret %struct._depend_unpack_t.0 %9, !dbg !40
}

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) #3

; Function Attrs: nounwind
declare i1 @llvm.directive.marker() #1

; CHECK: define internal void @nanos6_unpacked_deps_main0([100 x i32]* %array, i32* %i, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %0 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %0 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %1 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %2 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub = sub i32 %2, 1
; CHECK-NEXT:   %3 = call i32 @compute_lb()
; CHECK-NEXT:   %4 = call i32 @compute_ub()
; CHECK-NEXT:   %5 = call i32 @compute_step()
; CHECK-NEXT:   %6 = sub i32 %4, %3
; CHECK-NEXT:   %7 = sub i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %i.lb = alloca i32, align 4
; CHECK-NEXT:   %i.ub = alloca i32, align 4
; CHECK-NEXT:   %11 = sext i32 %lb to i64
; CHECK-NEXT:   %12 = udiv i64 %11, 1
; CHECK-NEXT:   %13 = sext i32 %5 to i64
; CHECK-NEXT:   %14 = mul i64 %12, %13
; CHECK-NEXT:   %15 = sext i32 %3 to i64
; CHECK-NEXT:   %16 = add i64 %14, %15
; CHECK-NEXT:   %17 = mul i64 %12, 1
; CHECK-NEXT:   %18 = sext i32 %lb to i64
; CHECK-NEXT:   %19 = sub i64 %18, %17
; CHECK-NEXT:   %20 = trunc i64 %16 to i32
; CHECK-NEXT:   store i32 %20, i32* %i.lb, align 4
; CHECK-NEXT:   %21 = sext i32 %ub to i64
; CHECK-NEXT:   %22 = udiv i64 %21, 1
; CHECK-NEXT:   %23 = sext i32 %5 to i64
; CHECK-NEXT:   %24 = mul i64 %22, %23
; CHECK-NEXT:   %25 = sext i32 %3 to i64
; CHECK-NEXT:   %26 = add i64 %24, %25
; CHECK-NEXT:   %27 = mul i64 %22, 1
; CHECK-NEXT:   %28 = sext i32 %ub to i64
; CHECK-NEXT:   %29 = sub i64 %28, %27
; CHECK-NEXT:   %30 = trunc i64 %26 to i32
; CHECK-NEXT:   store i32 %30, i32* %i.ub, align 4
; CHECK-NEXT:   %31 = call %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32* %i.lb)
; CHECK-NEXT:   %32 = call %struct._depend_unpack_t @compute_dep([100 x i32]* %array, i32* %i.ub)
; CHECK-NEXT:   %33 = extractvalue %struct._depend_unpack_t %31, 0
; CHECK-NEXT:   %34 = bitcast i32* %33 to i8*
; CHECK-NEXT:   %35 = extractvalue %struct._depend_unpack_t %31, 1
; CHECK-NEXT:   %36 = extractvalue %struct._depend_unpack_t %31, 2
; CHECK-NEXT:   %37 = extractvalue %struct._depend_unpack_t %32, 3
; CHECK-NEXT:   call void @nanos6_register_region_write_depinfo1(i8* %handler, i32 0, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @1, i32 0, i32 0), i8* %34, i64 %35, i64 %36, i64 %37)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main1([100 x i32]* %array, i32* %i, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK: entry:
; CHECK-NEXT:   %lb_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 0
; CHECK-NEXT:   %0 = load i64, i64* %lb_gep, align 8
; CHECK-NEXT:   %lb = trunc i64 %0 to i32
; CHECK-NEXT:   %ub_gep = getelementptr %nanos6_loop_bounds_t, %nanos6_loop_bounds_t* %loop_bounds, i32 0, i32 1
; CHECK-NEXT:   %1 = load i64, i64* %ub_gep, align 8
; CHECK-NEXT:   %2 = trunc i64 %1 to i32
; CHECK-NEXT:   %ub = sub i32 %2, 1
; CHECK-NEXT:   %3 = call i32 @compute_lb.1()
; CHECK-NEXT:   %4 = call i32 @compute_ub.2()
; CHECK-NEXT:   %5 = call i32 @compute_step.3()
; CHECK-NEXT:   %6 = sub i32 %4, %3
; CHECK-NEXT:   %7 = sub i32 %6, 1
; CHECK-NEXT:   %8 = sdiv i32 %7, %5
; CHECK-NEXT:   %9 = add i32 %8, 1
; CHECK-NEXT:   %10 = sext i32 %9 to i64
; CHECK-NEXT:   %i.lb = alloca i32, align 4
; CHECK-NEXT:   %i.ub = alloca i32, align 4
; CHECK-NEXT:   %11 = sext i32 %lb to i64
; CHECK-NEXT:   %12 = udiv i64 %11, 1
; CHECK-NEXT:   %13 = sext i32 %5 to i64
; CHECK-NEXT:   %14 = mul i64 %12, %13
; CHECK-NEXT:   %15 = sext i32 %3 to i64
; CHECK-NEXT:   %16 = add i64 %14, %15
; CHECK-NEXT:   %17 = mul i64 %12, 1
; CHECK-NEXT:   %18 = sext i32 %lb to i64
; CHECK-NEXT:   %19 = sub i64 %18, %17
; CHECK-NEXT:   %20 = trunc i64 %16 to i32
; CHECK-NEXT:   store i32 %20, i32* %i.lb, align 4
; CHECK-NEXT:   %21 = sext i32 %ub to i64
; CHECK-NEXT:   %22 = udiv i64 %21, 1
; CHECK-NEXT:   %23 = sext i32 %5 to i64
; CHECK-NEXT:   %24 = mul i64 %22, %23
; CHECK-NEXT:   %25 = sext i32 %3 to i64
; CHECK-NEXT:   %26 = add i64 %24, %25
; CHECK-NEXT:   %27 = mul i64 %22, 1
; CHECK-NEXT:   %28 = sext i32 %ub to i64
; CHECK-NEXT:   %29 = sub i64 %28, %27
; CHECK-NEXT:   %30 = trunc i64 %26 to i32
; CHECK-NEXT:   store i32 %30, i32* %i.ub, align 4
; CHECK-NEXT:   %31 = call %struct._depend_unpack_t.0 @compute_dep.4([100 x i32]* %array, i32* %i.lb)
; CHECK-NEXT:   %32 = call %struct._depend_unpack_t.0 @compute_dep.4([100 x i32]* %array, i32* %i.ub)
; CHECK-NEXT:   %33 = extractvalue %struct._depend_unpack_t.0 %31, 0
; CHECK-NEXT:   %34 = bitcast i32* %33 to i8*
; CHECK-NEXT:   %35 = extractvalue %struct._depend_unpack_t.0 %31, 1
; CHECK-NEXT:   %36 = extractvalue %struct._depend_unpack_t.0 %31, 2
; CHECK-NEXT:   %37 = extractvalue %struct._depend_unpack_t.0 %32, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @3, i32 0, i32 0), i8* %34, i64 %35, i64 %36, i64 %37)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }
attributes #2 = { "min-legal-vector-width"="0" }
attributes #3 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "taskloop_deps.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!""}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 6, column: 12, scope: !8)
!11 = !DILocation(line: 6, column: 10, scope: !8)
!12 = !DILocation(line: 7, column: 20, scope: !8)
!13 = !DILocation(line: 7, column: 15, scope: !8)
!14 = !DILocation(line: 7, column: 9, scope: !8)
!15 = !DILocation(line: 7, column: 18, scope: !8)
!16 = !DILocation(line: 8, column: 5, scope: !8)
!17 = !DILocation(line: 10, column: 12, scope: !8)
!18 = !DILocation(line: 10, column: 10, scope: !8)
!19 = !DILocation(line: 11, column: 9, scope: !8)
!20 = !DILocation(line: 12, column: 5, scope: !8)
!21 = !DILocation(line: 13, column: 13, scope: !8)
!22 = !DILocation(line: 14, column: 1, scope: !8)
!23 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 6, column: 14, scope: !23)
!25 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 6, column: 21, scope: !25)
!27 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!28 = !DILocation(line: 6, column: 26, scope: !27)
!29 = distinct !DISubprogram(linkageName: "compute_dep", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 5, column: 36, scope: !29)
!31 = !DILocation(line: 5, column: 30, scope: !29)
!32 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 10, column: 14, scope: !32)
!34 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!35 = !DILocation(line: 10, column: 21, scope: !34)
!36 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!37 = !DILocation(line: 10, column: 26, scope: !36)
!38 = distinct !DISubprogram(linkageName: "compute_dep.4", scope: !1, file: !1, type: !9, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!39 = !DILocation(line: 9, column: 35, scope: !38)
!40 = !DILocation(line: 9, column: 29, scope: !38)
