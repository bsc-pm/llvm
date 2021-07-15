; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop_clauses.ll'
source_filename = "loop_clauses.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo() {
;     #pragma oss task for chunksize(777)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss taskloop for chunksize(777)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss taskloop grainsize(777)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss taskloop for grainsize(777)
;     for (int i = 0; i < 10; ++i) {}
; }

; Function Attrs: noinline nounwind optnone
define dso_local void @foo() #0 !dbg !6 {
entry:
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %i3 = alloca i32, align 4
  store i32 0, i32* %i, align 4, !dbg !9
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.CHUNKSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 777) ], !dbg !10
  call void @llvm.directive.region.exit(token %0), !dbg !11
  store i32 0, i32* %i1, align 4, !dbg !12
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.CHUNKSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 777) ], !dbg !13
  call void @llvm.directive.region.exit(token %1), !dbg !14
  store i32 0, i32* %i2, align 4, !dbg !15
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.4), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.5), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.6), "QUAL.OSS.LOOP.GRAINSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 777) ], !dbg !16
  call void @llvm.directive.region.exit(token %2), !dbg !17
  store i32 0, i32* %i3, align 4, !dbg !18
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.7), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.8), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.9), "QUAL.OSS.LOOP.GRAINSIZE"(i32 777), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 777) ], !dbg !19
  call void @llvm.directive.region.exit(token %3), !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_lb() #2 !dbg !22 {
entry:
  ret i32 0, !dbg !23
}

define internal i32 @compute_ub() #2 !dbg !25 {
entry:
  ret i32 10, !dbg !26
}

define internal i32 @compute_step() #2 !dbg !28 {
entry:
  ret i32 1, !dbg !29
}

define internal i32 @compute_lb.1() #2 !dbg !31 {
entry:
  ret i32 0, !dbg !32
}

define internal i32 @compute_ub.2() #2 !dbg !34 {
entry:
  ret i32 10, !dbg !35
}

define internal i32 @compute_step.3() #2 !dbg !37 {
entry:
  ret i32 1, !dbg !38
}

define internal i32 @compute_lb.4() #2 !dbg !40 {
entry:
  ret i32 0, !dbg !41
}

define internal i32 @compute_ub.5() #2 !dbg !43 {
entry:
  ret i32 10, !dbg !44
}

define internal i32 @compute_step.6() #2 !dbg !46 {
entry:
  ret i32 1, !dbg !47
}

define internal i32 @compute_lb.7() #2 !dbg !49 {
entry:
  ret i32 0, !dbg !50
}

define internal i32 @compute_ub.8() #2 !dbg !52 {
entry:
  ret i32 10, !dbg !53
}

define internal i32 @compute_step.9() #2 !dbg !55 {
entry:
  ret i32 1, !dbg !56
}

; CHECK: call void @nanos6_create_loop(%nanos6_task_info_t* @task_info_var_foo0, %nanos6_task_invocation_info_t* @task_invocation_info_foo0, i64 16, i8** %8, i8** %1, i64 8, i64 %9, i64 0, i64 %18, i64 0, i64 777)
; CHECK: call void @nanos6_create_loop(%nanos6_task_info_t* @task_info_var_foo1, %nanos6_task_invocation_info_t* @task_invocation_info_foo1, i64 16, i8** %22, i8** %3, i64 12, i64 %23, i64 0, i64 %32, i64 0, i64 777)
; CHECK: call void @nanos6_create_loop(%nanos6_task_info_t* @task_info_var_foo2, %nanos6_task_invocation_info_t* @task_invocation_info_foo2, i64 16, i8** %36, i8** %5, i64 4, i64 %37, i64 0, i64 %46, i64 777, i64 0)
; CHECK: call void @nanos6_create_loop(%nanos6_task_info_t* @task_info_var_foo3, %nanos6_task_invocation_info_t* @task_invocation_info_foo3, i64 16, i8** %50, i8** %7, i64 12, i64 %51, i64 0, i64 %60, i64 777, i64 0)

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
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "loop_clauses.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 14, scope: !6)
!10 = !DILocation(line: 3, column: 10, scope: !6)
!11 = !DILocation(line: 3, column: 35, scope: !6)
!12 = !DILocation(line: 5, column: 14, scope: !6)
!13 = !DILocation(line: 5, column: 10, scope: !6)
!14 = !DILocation(line: 5, column: 35, scope: !6)
!15 = !DILocation(line: 7, column: 14, scope: !6)
!16 = !DILocation(line: 7, column: 10, scope: !6)
!17 = !DILocation(line: 7, column: 35, scope: !6)
!18 = !DILocation(line: 9, column: 14, scope: !6)
!19 = !DILocation(line: 9, column: 10, scope: !6)
!20 = !DILocation(line: 9, column: 35, scope: !6)
!21 = !DILocation(line: 10, column: 1, scope: !6)
!22 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!23 = !DILocation(line: 3, column: 18, scope: !24)
!24 = !DILexicalBlockFile(scope: !22, file: !7, discriminator: 0)
!25 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 3, column: 25, scope: !27)
!27 = !DILexicalBlockFile(scope: !25, file: !7, discriminator: 0)
!28 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!29 = !DILocation(line: 3, column: 29, scope: !30)
!30 = !DILexicalBlockFile(scope: !28, file: !7, discriminator: 0)
!31 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!32 = !DILocation(line: 5, column: 18, scope: !33)
!33 = !DILexicalBlockFile(scope: !31, file: !7, discriminator: 0)
!34 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!35 = !DILocation(line: 5, column: 25, scope: !36)
!36 = !DILexicalBlockFile(scope: !34, file: !7, discriminator: 0)
!37 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!38 = !DILocation(line: 5, column: 29, scope: !39)
!39 = !DILexicalBlockFile(scope: !37, file: !7, discriminator: 0)
!40 = distinct !DISubprogram(linkageName: "compute_lb.4", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!41 = !DILocation(line: 7, column: 18, scope: !42)
!42 = !DILexicalBlockFile(scope: !40, file: !7, discriminator: 0)
!43 = distinct !DISubprogram(linkageName: "compute_ub.5", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!44 = !DILocation(line: 7, column: 25, scope: !45)
!45 = !DILexicalBlockFile(scope: !43, file: !7, discriminator: 0)
!46 = distinct !DISubprogram(linkageName: "compute_step.6", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!47 = !DILocation(line: 7, column: 29, scope: !48)
!48 = !DILexicalBlockFile(scope: !46, file: !7, discriminator: 0)
!49 = distinct !DISubprogram(linkageName: "compute_lb.7", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!50 = !DILocation(line: 9, column: 18, scope: !51)
!51 = !DILexicalBlockFile(scope: !49, file: !7, discriminator: 0)
!52 = distinct !DISubprogram(linkageName: "compute_ub.8", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!53 = !DILocation(line: 9, column: 25, scope: !54)
!54 = !DILexicalBlockFile(scope: !52, file: !7, discriminator: 0)
!55 = distinct !DISubprogram(linkageName: "compute_step.9", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!56 = !DILocation(line: 9, column: 29, scope: !57)
!57 = !DILexicalBlockFile(scope: !55, file: !7, discriminator: 0)



