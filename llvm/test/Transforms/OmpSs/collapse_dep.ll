; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'collapse_dep.ll'
source_filename = "collapse_dep.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int array[50];
; int main() {
;     #pragma oss taskloop collapse(2) in( array[i + j] )
;     for (int i = 0; i < 10; ++i) {
;         for (int j = 0; j < 10; ++j) {
;         }
;     }
; }

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }

@array = global [50 x i32] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, i32* %i, align 4, !dbg !9
  store i32 0, i32* %j, align 4, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"([50 x i32]* @array), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.LOOP.IND.VAR"(i32* %i, i32* %j), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb, i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub, i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step, i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1, i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.DEP.IN"([50 x i32]* @array, [13 x i8] c"array[i + j]\00", %struct._depend_unpack_t (i32*, i32*, [50 x i32]*)* @compute_dep, i32* %i, i32* %j, [50 x i32]* @array) ], !dbg !11
  call void @llvm.directive.region.exit(token %0), !dbg !12
  ret i32 0, !dbg !13
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb() #2 !dbg !14 {
entry:
  ret i32 0, !dbg !15
}

define internal i32 @compute_ub() #2 !dbg !17 {
entry:
  ret i32 10, !dbg !18
}

define internal i32 @compute_step() #2 !dbg !20 {
entry:
  ret i32 1, !dbg !21
}

define internal i32 @compute_lb.1() #2 !dbg !23 {
entry:
  ret i32 0, !dbg !24
}

define internal i32 @compute_ub.2() #2 !dbg !26 {
entry:
  ret i32 10, !dbg !27
}

define internal i32 @compute_step.3() #2 !dbg !29 {
entry:
  ret i32 1, !dbg !30
}

define internal %struct._depend_unpack_t @compute_dep(i32* %i, i32* %j, [50 x i32]* %array) #2 !dbg !32 {
entry:
  %retval = alloca %struct._depend_unpack_t, align 8
  %i.addr = alloca i32*, align 8
  %j.addr = alloca i32*, align 8
  %array.addr = alloca [50 x i32]*, align 8
  store i32* %i, i32** %i.addr, align 8
  store i32* %j, i32** %j.addr, align 8
  store [50 x i32]* %array, [50 x i32]** %array.addr, align 8
  %0 = load i32, i32* %i, align 4, !dbg !33
  %1 = load i32, i32* %j, align 4, !dbg !35
  %add = add nsw i32 %0, %1, !dbg !36
  %2 = sext i32 %add to i64
  %3 = add i64 %2, 1
  %arraydecay = getelementptr inbounds [50 x i32], [50 x i32]* %array, i64 0, i64 0, !dbg !37
  %4 = mul i64 %2, 4
  %5 = mul i64 %3, 4
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
  store i32* %arraydecay, i32** %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
  store i64 200, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
  store i64 %4, i64* %8, align 8
  %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
  store i64 %5, i64* %9, align 8
  %10 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8, !dbg !37
  ret %struct._depend_unpack_t %10, !dbg !37
}

; CHECK: define internal void @nanos6_unpacked_deps_main0([50 x i32]* %array, i32* %i, i32* %j, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
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
; CHECK-NEXT:   %58 = call %struct._depend_unpack_t @compute_dep(i32* %i.lb, i32* %j.lb, [50 x i32]* %array)
; CHECK-NEXT:   %59 = call %struct._depend_unpack_t @compute_dep(i32* %i.ub, i32* %j.ub, [50 x i32]* %array)
; CHECK-NEXT:   %60 = extractvalue %struct._depend_unpack_t %58, 0
; CHECK-NEXT:   %61 = bitcast i32* %60 to i8*
; CHECK-NEXT:   %62 = extractvalue %struct._depend_unpack_t %58, 1
; CHECK-NEXT:   %63 = extractvalue %struct._depend_unpack_t %58, 2
; CHECK-NEXT:   %64 = extractvalue %struct._depend_unpack_t %59, 3
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @1, i32 0, i32 0), i8* %61, i64 %62, i64 %63, i64 %64)
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
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "collapse_dep.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 14, scope: !6)
!10 = !DILocation(line: 5, column: 18, scope: !6)
!11 = !DILocation(line: 5, column: 14, scope: !6)
!12 = !DILocation(line: 6, column: 9, scope: !6)
!13 = !DILocation(line: 8, column: 1, scope: !6)
!14 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 4, column: 18, scope: !16)
!16 = !DILexicalBlockFile(scope: !14, file: !7, discriminator: 0)
!17 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 4, column: 25, scope: !19)
!19 = !DILexicalBlockFile(scope: !17, file: !7, discriminator: 0)
!20 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 4, column: 29, scope: !22)
!22 = !DILexicalBlockFile(scope: !20, file: !7, discriminator: 0)
!23 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 5, column: 22, scope: !25)
!25 = !DILexicalBlockFile(scope: !23, file: !7, discriminator: 0)
!26 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 5, column: 29, scope: !28)
!28 = !DILexicalBlockFile(scope: !26, file: !7, discriminator: 0)
!29 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 5, column: 33, scope: !31)
!31 = !DILexicalBlockFile(scope: !29, file: !7, discriminator: 0)
!32 = distinct !DISubprogram(linkageName: "compute_dep", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 3, column: 48, scope: !34)
!34 = !DILexicalBlockFile(scope: !32, file: !7, discriminator: 0)
!35 = !DILocation(line: 3, column: 52, scope: !34)
!36 = !DILocation(line: 3, column: 50, scope: !34)
!37 = !DILocation(line: 3, column: 42, scope: !34)
