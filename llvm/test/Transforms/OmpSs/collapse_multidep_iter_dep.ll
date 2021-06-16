; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'collapse_multidep_iter_dep.ll'
source_filename = "collapse_multidep_iter_dep.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Checking we use lowerbound only to compute the dependency

; int array[50];
; int main() {
;     #pragma oss taskloop collapse(2) in( { array[i + j + k], k=0;j } )
;     for (int i = 0; i < 10; ++i) {
;         for (int j = 0; j < 10; ++j) {
;         }
;     }
; } 

%struct._depend_unpack_t = type { i32, i32, i32, i32 }
%struct._depend_unpack_t.0 = type { i32*, i64, i64, i64 }

@array = global [50 x i32] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, i32* %i, align 4, !dbg !9
  store i32 0, i32* %j, align 4, !dbg !10
  store i32 0, i32* %k, align 4, !dbg !11
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([50 x i32]* @array), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.PRIVATE"(i32* %k), "QUAL.OSS.LOOP.IND.VAR"(i32* %i, i32* %j), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb, i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub, i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step, i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1, i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.MULTIDEP.RANGE.IN"(i32* %k, %struct._depend_unpack_t (i32*, i32*, i64)* @compute_dep, i32* %k, i32* %j, [50 x i32]* @array, [28 x i8] c"{ array[i + j + k], k=0;j }\00", %struct._depend_unpack_t.0 (i32*, i32*, i32*, [50 x i32]*)* @compute_dep.4, i32* %i, i32* %j, i32* %k, [50 x i32]* @array) ], !dbg !12
  call void @llvm.directive.region.exit(token %0), !dbg !13
  ret i32 0, !dbg !14
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb() #2 !dbg !15 {
entry:
  ret i32 0, !dbg !16
}

define internal i32 @compute_ub() #2 !dbg !18 {
entry:
  ret i32 10, !dbg !19
}

define internal i32 @compute_step() #2 !dbg !21 {
entry:
  ret i32 1, !dbg !22
}

define internal i32 @compute_lb.1() #2 !dbg !24 {
entry:
  ret i32 0, !dbg !25
}

define internal i32 @compute_ub.2() #2 !dbg !27 {
entry:
  ret i32 10, !dbg !28
}

define internal i32 @compute_step.3() #2 !dbg !30 {
entry:
  ret i32 1, !dbg !31
}

define internal %struct._depend_unpack_t @compute_dep(i32* %k, i32* %j, i64 %0) #2 !dbg !33 {
entry:
  %retval = alloca %struct._depend_unpack_t, align 4
  %k.addr = alloca i32*, align 8
  %j.addr = alloca i32*, align 8
  %.addr = alloca i64, align 8
  store i32* %k, i32** %k.addr, align 8
  store i32* %j, i32** %j.addr, align 8
  store i64 %0, i64* %.addr, align 8
  switch i64 %0, label %3 [
    i64 0, label %4
  ]

1:                                                ; preds = %4, %3
  %2 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 4, !dbg !34
  ret %struct._depend_unpack_t %2, !dbg !34

3:                                                ; preds = %entry
  br label %1

4:                                                ; preds = %entry
  %5 = load i32, i32* %k, align 4, !dbg !36
  %6 = load i32, i32* %j, align 4, !dbg !34
  %7 = add i32 0, %6
  %8 = add i32 %7, -1
  %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
  store i32 0, i32* %9, align 4
  %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
  store i32 %5, i32* %10, align 4
  %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
  store i32 %8, i32* %11, align 4
  %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
  store i32 1, i32* %12, align 4
  br label %1
}

define internal %struct._depend_unpack_t.0 @compute_dep.4(i32* %i, i32* %j, i32* %k, [50 x i32]* %array) #2 !dbg !37 {
entry:
  %retval = alloca %struct._depend_unpack_t.0, align 8
  %i.addr = alloca i32*, align 8
  %j.addr = alloca i32*, align 8
  %k.addr = alloca i32*, align 8
  %array.addr = alloca [50 x i32]*, align 8
  store i32* %i, i32** %i.addr, align 8
  store i32* %j, i32** %j.addr, align 8
  store i32* %k, i32** %k.addr, align 8
  store [50 x i32]* %array, [50 x i32]** %array.addr, align 8
  %0 = load i32, i32* %i, align 4, !dbg !38
  %1 = load i32, i32* %j, align 4, !dbg !40
  %add = add nsw i32 %0, %1, !dbg !41
  %2 = load i32, i32* %k, align 4, !dbg !42
  %add1 = add nsw i32 %add, %2, !dbg !43
  %3 = sext i32 %add1 to i64
  %4 = add i64 %3, 1
  %arraydecay = getelementptr inbounds [50 x i32], [50 x i32]* %array, i64 0, i64 0, !dbg !44
  %5 = mul i64 %3, 4
  %6 = mul i64 %4, 4
  %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
  store i32* %arraydecay, i32** %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
  store i64 200, i64* %8, align 8
  %9 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
  store i64 %5, i64* %9, align 8
  %10 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
  store i64 %6, i64* %10, align 8
  %11 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8, !dbg !44
  ret %struct._depend_unpack_t.0 %11, !dbg !44
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([50 x i32]* %array, i32* %i, i32* %j, i32* %k, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
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
; CHECK-NEXT:   %11 = sub i32 %4, 1
; CHECK-NEXT:   store i32 %3, i32* %i, align 4
; CHECK-NEXT:   %12 = call i32 @compute_lb.1()
; CHECK-NEXT:   store i32 %11, i32* %i, align 4
; CHECK-NEXT:   %13 = call i32 @compute_ub.2()
; CHECK-NEXT:   %14 = call i32 @compute_step.3()
; CHECK-NEXT:   %15 = sub i32 %13, %12
; CHECK-NEXT:   %16 = sub i32 %15, 1
; CHECK-NEXT:   %17 = sdiv i32 %16, %14
; CHECK-NEXT:   %18 = add i32 %17, 1
; CHECK-NEXT:   %19 = sext i32 %18 to i64
; CHECK-NEXT:   %i.lb = alloca i32, align 4
; CHECK-NEXT:   %i.ub = alloca i32, align 4
; CHECK-NEXT:   %20 = mul i64 1, %19
; CHECK-NEXT:   %21 = sext i32 %lb to i64
; CHECK-NEXT:   %22 = udiv i64 %21, %20
; CHECK-NEXT:   %23 = sext i32 %5 to i64
; CHECK-NEXT:   %24 = mul i64 %22, %23
; CHECK-NEXT:   %25 = sext i32 %3 to i64
; CHECK-NEXT:   %26 = add i64 %24, %25
; CHECK-NEXT:   %27 = mul i64 %22, %20
; CHECK-NEXT:   %28 = sext i32 %lb to i64
; CHECK-NEXT:   %29 = sub i64 %28, %27
; CHECK-NEXT:   %30 = trunc i64 %26 to i32
; CHECK-NEXT:   store i32 %30, i32* %i.lb, align 4
; CHECK-NEXT:   %31 = mul i64 1, %19
; CHECK-NEXT:   %32 = sext i32 %ub to i64
; CHECK-NEXT:   %33 = udiv i64 %32, %31
; CHECK-NEXT:   %34 = sext i32 %5 to i64
; CHECK-NEXT:   %35 = mul i64 %33, %34
; CHECK-NEXT:   %36 = sext i32 %3 to i64
; CHECK-NEXT:   %37 = add i64 %35, %36
; CHECK-NEXT:   %38 = mul i64 %33, %31
; CHECK-NEXT:   %39 = sext i32 %ub to i64
; CHECK-NEXT:   %40 = sub i64 %39, %38
; CHECK-NEXT:   %41 = trunc i64 %37 to i32
; CHECK-NEXT:   store i32 %41, i32* %i.ub, align 4
; CHECK-NEXT:   %j.lb = alloca i32, align 4
; CHECK-NEXT:   %j.ub = alloca i32, align 4
; CHECK-NEXT:   %42 = udiv i64 %29, 1
; CHECK-NEXT:   %43 = sext i32 %14 to i64
; CHECK-NEXT:   %44 = mul i64 %42, %43
; CHECK-NEXT:   %45 = sext i32 %12 to i64
; CHECK-NEXT:   %46 = add i64 %44, %45
; CHECK-NEXT:   %47 = mul i64 %42, 1
; CHECK-NEXT:   %48 = sub i64 %29, %47
; CHECK-NEXT:   %49 = trunc i64 %46 to i32
; CHECK-NEXT:   store i32 %49, i32* %j.lb, align 4
; CHECK-NEXT:   %50 = udiv i64 %40, 1
; CHECK-NEXT:   %51 = sext i32 %14 to i64
; CHECK-NEXT:   %52 = mul i64 %50, %51
; CHECK-NEXT:   %53 = sext i32 %12 to i64
; CHECK-NEXT:   %54 = add i64 %52, %53
; CHECK-NEXT:   %55 = mul i64 %50, 1
; CHECK-NEXT:   %56 = sub i64 %40, %55
; CHECK-NEXT:   %57 = trunc i64 %54 to i32
; CHECK-NEXT:   store i32 %57, i32* %j.ub, align 4
; CHECK-NEXT:   %k.remap = alloca i32, align 4
; CHECK-NEXT:   br label %58
; CHECK: 58:                                               ; preds = %entry
; CHECK-NEXT:   store i32 0, i32* %k, align 4
; CHECK-NEXT:   %59 = call %struct._depend_unpack_t @compute_dep(i32* %k, i32* %j.lb, i64 0)
; CHECK-NEXT:   %60 = extractvalue %struct._depend_unpack_t %59, 0
; CHECK-NEXT:   %61 = extractvalue %struct._depend_unpack_t %59, 2
; CHECK-NEXT:   %62 = extractvalue %struct._depend_unpack_t %59, 3
; CHECK-NEXT:   store i32 %60, i32* %k, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.cond:                                         ; preds = %for.incr, %58
; CHECK-NEXT:   %63 = load i32, i32* %k, align 4
; CHECK-NEXT:   %64 = icmp sle i32 %63, %61
; CHECK-NEXT:   br i1 %64, label %for.body, label %74
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %65 = call %struct._depend_unpack_t @compute_dep(i32* %k, i32* %j.lb, i64 0)
; CHECK-NEXT:   %66 = extractvalue %struct._depend_unpack_t %65, 1
; CHECK-NEXT:   store i32 %66, i32* %k.remap, align 4
; CHECK-NEXT:   %67 = call %struct._depend_unpack_t.0 @compute_dep.4(i32* %i.lb, i32* %j.lb, i32* %k.remap, [50 x i32]* %array)
; CHECK-NEXT:   %68 = call %struct._depend_unpack_t.0 @compute_dep.4(i32* %i.lb, i32* %j.lb, i32* %k.remap, [50 x i32]* %array)
; CHECK-NEXT:   %69 = extractvalue %struct._depend_unpack_t.0 %67, 0
; CHECK-NEXT:   %70 = bitcast i32* %69 to i8*
; CHECK-NEXT:   %71 = extractvalue %struct._depend_unpack_t.0 %67, 1
; CHECK-NEXT:   %72 = extractvalue %struct._depend_unpack_t.0 %67, 2
; CHECK-NEXT:   %73 = extractvalue %struct._depend_unpack_t.0 %68, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* getelementptr inbounds ([28 x i8], [28 x i8]* @1, i32 0, i32 0), i8* %70, i64 %71, i64 %72, i64 %73)
; CHECK-NEXT:   br label %for.incr
; CHECK: 74:                                               ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK: for.incr:                                         ; preds = %for.body
; CHECK-NEXT:   %75 = load i32, i32* %k, align 4
; CHECK-NEXT:   %76 = add i32 %75, %62
; CHECK-NEXT:   store i32 %76, i32* %k, align 4
; CHECK-NEXT:   br label %for.cond
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
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "collapse_multidep_iter_dep.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 14, scope: !6)
!10 = !DILocation(line: 5, column: 18, scope: !6)
!11 = !DILocation(line: 3, column: 62, scope: !6)
!12 = !DILocation(line: 5, column: 14, scope: !6)
!13 = !DILocation(line: 6, column: 9, scope: !6)
!14 = !DILocation(line: 8, column: 1, scope: !6)
!15 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 4, column: 18, scope: !17)
!17 = !DILexicalBlockFile(scope: !15, file: !7, discriminator: 0)
!18 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!19 = !DILocation(line: 4, column: 25, scope: !20)
!20 = !DILexicalBlockFile(scope: !18, file: !7, discriminator: 0)
!21 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 4, column: 29, scope: !23)
!23 = !DILexicalBlockFile(scope: !21, file: !7, discriminator: 0)
!24 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!25 = !DILocation(line: 5, column: 22, scope: !26)
!26 = !DILexicalBlockFile(scope: !24, file: !7, discriminator: 0)
!27 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!28 = !DILocation(line: 5, column: 29, scope: !29)
!29 = !DILexicalBlockFile(scope: !27, file: !7, discriminator: 0)
!30 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!31 = !DILocation(line: 5, column: 33, scope: !32)
!32 = !DILexicalBlockFile(scope: !30, file: !7, discriminator: 0)
!33 = distinct !DISubprogram(linkageName: "compute_dep", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!34 = !DILocation(line: 3, column: 66, scope: !35)
!35 = !DILexicalBlockFile(scope: !33, file: !7, discriminator: 0)
!36 = !DILocation(line: 3, column: 62, scope: !35)
!37 = distinct !DISubprogram(linkageName: "compute_dep.4", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!38 = !DILocation(line: 3, column: 50, scope: !39)
!39 = !DILexicalBlockFile(scope: !37, file: !7, discriminator: 0)
!40 = !DILocation(line: 3, column: 54, scope: !39)
!41 = !DILocation(line: 3, column: 52, scope: !39)
!42 = !DILocation(line: 3, column: 58, scope: !39)
!43 = !DILocation(line: 3, column: 56, scope: !39)
!44 = !DILocation(line: 3, column: 44, scope: !39)
