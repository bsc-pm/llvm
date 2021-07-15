; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop_duplicate_args_array.ll'
source_filename = "loop_duplicate_args_array.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i8 }

; This test checks task_args duplicate shared/private/firstprivate arrays

; struct S {
;     S();
;     S(const S&);
;     ~S();
; };
;
; void foo() {
;     S s[2];
;     #pragma oss task for shared(s)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss task for private(s)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss task for firstprivate(s)
;     for (int i = 0; i < 10; ++i) {}
; }

; Function Attrs: noinline nounwind optnone mustprogress
define dso_local void @_Z3foov() #0 !dbg !6 {
entry:
  %s = alloca [2 x %struct.S], align 1
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %array.begin = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* %s, i32 0, i32 0, !dbg !9
  %arrayctor.end = getelementptr inbounds %struct.S, %struct.S* %array.begin, i64 2, !dbg !9
  br label %arrayctor.loop, !dbg !9

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi %struct.S* [ %array.begin, %entry ], [ %arrayctor.next, %arrayctor.loop ], !dbg !9
  call void @_ZN1SC1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arrayctor.cur), !dbg !9
  %arrayctor.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.cur, i64 1, !dbg !9
  %arrayctor.done = icmp eq %struct.S* %arrayctor.next, %arrayctor.end, !dbg !9
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !9

arrayctor.cont:                                   ; preds = %arrayctor.loop
  store i32 0, i32* %i, align 4, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"([2 x %struct.S]* %s), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !11
  call void @llvm.directive.region.exit(token %0), !dbg !12
  store i32 0, i32* %i1, align 4, !dbg !13
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"([2 x %struct.S]* %s), "QUAL.OSS.INIT"([2 x %struct.S]* %s, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([2 x %struct.S]* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !14
  call void @llvm.directive.region.exit(token %1), !dbg !15
  store i32 0, i32* %i2, align 4, !dbg !16
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.FIRSTPRIVATE"([2 x %struct.S]* %s), "QUAL.OSS.COPY"([2 x %struct.S]* %s, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_), "QUAL.OSS.DEINIT"([2 x %struct.S]* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.4), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.5), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.6), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg !17
  call void @llvm.directive.region.exit(token %2), !dbg !18
  %array.begin3 = getelementptr inbounds [2 x %struct.S], [2 x %struct.S]* %s, i32 0, i32 0, !dbg !19
  %3 = getelementptr inbounds %struct.S, %struct.S* %array.begin3, i64 2, !dbg !19
  br label %arraydestroy.body, !dbg !19

arraydestroy.body:                                ; preds = %arraydestroy.body, %arrayctor.cont
  %arraydestroy.elementPast = phi %struct.S* [ %3, %arrayctor.cont ], [ %arraydestroy.element, %arraydestroy.body ], !dbg !19
  %arraydestroy.element = getelementptr inbounds %struct.S, %struct.S* %arraydestroy.elementPast, i64 -1, !dbg !19
  call void @_ZN1SD1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arraydestroy.element) #2, !dbg !19
  %arraydestroy.done = icmp eq %struct.S* %arraydestroy.element, %array.begin3, !dbg !19
  br i1 %arraydestroy.done, label %arraydestroy.done4, label %arraydestroy.body, !dbg !19

arraydestroy.done4:                               ; preds = %arraydestroy.body
  ret void, !dbg !19
}

declare void @_ZN1SC1Ev(%struct.S* nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

define internal i32 @compute_lb() #3 !dbg !20 {
entry:
  ret i32 0, !dbg !21
}

define internal i32 @compute_ub() #3 !dbg !23 {
entry:
  ret i32 10, !dbg !24
}

define internal i32 @compute_step() #3 !dbg !26 {
entry:
  ret i32 1, !dbg !27
}

; Function Attrs: noinline norecurse nounwind
define internal void @oss_ctor_ZN1SC1Ev(%struct.S* %0, i64 %1) #4 !dbg !29 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store i64 %1, i64* %.addr1, align 8
  %2 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !30
  %3 = load i64, i64* %.addr1, align 8, !dbg !30
  %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3, !dbg !30
  br label %arrayctor.loop, !dbg !30

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ], !dbg !30
  call void @_ZN1SC1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arrayctor.dst.cur), !dbg !31
  %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1, !dbg !30
  %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end, !dbg !30
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !30

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !31
}

; Function Attrs: nounwind
declare void @_ZN1SD1Ev(%struct.S* nonnull align 1 dereferenceable(1)) unnamed_addr #5

; Function Attrs: noinline norecurse nounwind
define internal void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1) #4 !dbg !33 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store i64 %1, i64* %.addr1, align 8
  %2 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !34
  %3 = load i64, i64* %.addr1, align 8, !dbg !34
  %arraydtor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3, !dbg !34
  br label %arraydtor.loop, !dbg !34

arraydtor.loop:                                   ; preds = %arraydtor.loop, %entry
  %arraydtor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arraydtor.dst.next, %arraydtor.loop ], !dbg !34
  call void @_ZN1SD1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arraydtor.dst.cur) #2, !dbg !34
  %arraydtor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arraydtor.dst.cur, i64 1, !dbg !34
  %arraydtor.done = icmp eq %struct.S* %arraydtor.dst.next, %arraydtor.dst.end, !dbg !34
  br i1 %arraydtor.done, label %arraydtor.cont, label %arraydtor.loop, !dbg !34

arraydtor.cont:                                   ; preds = %arraydtor.loop
  ret void, !dbg !34
}

define internal i32 @compute_lb.1() #3 !dbg !35 {
entry:
  ret i32 0, !dbg !36
}

define internal i32 @compute_ub.2() #3 !dbg !38 {
entry:
  ret i32 10, !dbg !39
}

define internal i32 @compute_step.3() #3 !dbg !41 {
entry:
  ret i32 1, !dbg !42
}

declare void @_ZN1SC1ERKS_(%struct.S* nonnull align 1 dereferenceable(1), %struct.S* nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: noinline norecurse nounwind
define internal void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %0, %struct.S* %1, i64 %2) #4 !dbg !44 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca %struct.S*, align 8
  %.addr2 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store %struct.S* %1, %struct.S** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !45
  %4 = load %struct.S*, %struct.S** %.addr1, align 8, !dbg !45
  %5 = load i64, i64* %.addr2, align 8, !dbg !45
  %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %4, i64 %5, !dbg !45
  br label %arrayctor.loop, !dbg !45

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi %struct.S* [ %4, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ], !dbg !45
  %arrayctor.src.cur = phi %struct.S* [ %3, %entry ], [ %arrayctor.src.next, %arrayctor.loop ], !dbg !45
  call void @_ZN1SC1ERKS_(%struct.S* nonnull align 1 dereferenceable(1) %arrayctor.dst.cur, %struct.S* nonnull align 1 dereferenceable(1) %arrayctor.src.cur), !dbg !46
  %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1, !dbg !45
  %arrayctor.src.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.src.cur, i64 1, !dbg !45
  %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end, !dbg !45
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !45

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !46
}

define internal i32 @compute_lb.4() #3 !dbg !48 {
entry:
  ret i32 0, !dbg !49
}

define internal i32 @compute_ub.5() #3 !dbg !51 {
entry:
  ret i32 10, !dbg !52
}

define internal i32 @compute_step.6() #3 !dbg !54 {
entry:
  ret i32 1, !dbg !55
}

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov1(%nanos6_task_args__Z3foov1* %task_args_src, %nanos6_task_args__Z3foov1** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:   %0 = load %nanos6_task_args__Z3foov1*, %nanos6_task_args__Z3foov1** %task_args_dst, align 8
; CHECK:  %gep_s = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %0, i32 0, i32 0
; CHECK-NEXT:  %2 = bitcast [2 x %struct.S]* %gep_s to %struct.S*
; CHECK-NEXT:  call void @oss_ctor_ZN1SC1Ev(%struct.S* %2, i64 2)

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov2(%nanos6_task_args__Z3foov2* %task_args_src, %nanos6_task_args__Z3foov2** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:   %0 = load %nanos6_task_args__Z3foov2*, %nanos6_task_args__Z3foov2** %task_args_dst, align 8
; CHECK:   %gep_src_s = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %task_args_src, i32 0, i32 1
; CHECK-NEXT:   %gep_dst_s = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %0, i32 0, i32 1
; CHECK-NEXT:   %2 = bitcast [2 x %struct.S]* %gep_src_s to %struct.S*
; CHECK-NEXT:   %3 = bitcast [2 x %struct.S]* %gep_dst_s to %struct.S*
; CHECK-NEXT:   call void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %2, %struct.S* %3, i64 2)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "loop_duplicate_args_array.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 7, column: 7, scope: !6)
!10 = !DILocation(line: 9, column: 14, scope: !6)
!11 = !DILocation(line: 9, column: 10, scope: !6)
!12 = !DILocation(line: 9, column: 35, scope: !6)
!13 = !DILocation(line: 11, column: 14, scope: !6)
!14 = !DILocation(line: 11, column: 10, scope: !6)
!15 = !DILocation(line: 11, column: 35, scope: !6)
!16 = !DILocation(line: 13, column: 14, scope: !6)
!17 = !DILocation(line: 13, column: 10, scope: !6)
!18 = !DILocation(line: 13, column: 35, scope: !6)
!19 = !DILocation(line: 14, column: 1, scope: !6)
!20 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 9, column: 18, scope: !22)
!22 = !DILexicalBlockFile(scope: !20, file: !7, discriminator: 0)
!23 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 9, column: 25, scope: !25)
!25 = !DILexicalBlockFile(scope: !23, file: !7, discriminator: 0)
!26 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 9, column: 29, scope: !28)
!28 = !DILexicalBlockFile(scope: !26, file: !7, discriminator: 0)
!29 = distinct !DISubprogram(linkageName: "oss_ctor_ZN1SC1Ev", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 0, scope: !29)
!31 = !DILocation(line: 10, column: 34, scope: !32)
!32 = !DILexicalBlockFile(scope: !29, file: !7, discriminator: 0)
!33 = distinct !DISubprogram(linkageName: "oss_dtor_ZN1SD1Ev", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!34 = !DILocation(line: 0, scope: !33)
!35 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!36 = !DILocation(line: 11, column: 18, scope: !37)
!37 = !DILexicalBlockFile(scope: !35, file: !7, discriminator: 0)
!38 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!39 = !DILocation(line: 11, column: 25, scope: !40)
!40 = !DILexicalBlockFile(scope: !38, file: !7, discriminator: 0)
!41 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!42 = !DILocation(line: 11, column: 29, scope: !43)
!43 = !DILexicalBlockFile(scope: !41, file: !7, discriminator: 0)
!44 = distinct !DISubprogram(linkageName: "oss_copy_ctor_ZN1SC1ERKS_", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!45 = !DILocation(line: 0, scope: !44)
!46 = !DILocation(line: 12, column: 39, scope: !47)
!47 = !DILexicalBlockFile(scope: !44, file: !7, discriminator: 0)
!48 = distinct !DISubprogram(linkageName: "compute_lb.4", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!49 = !DILocation(line: 13, column: 18, scope: !50)
!50 = !DILexicalBlockFile(scope: !48, file: !7, discriminator: 0)
!51 = distinct !DISubprogram(linkageName: "compute_ub.5", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!52 = !DILocation(line: 13, column: 25, scope: !53)
!53 = !DILexicalBlockFile(scope: !51, file: !7, discriminator: 0)
!54 = distinct !DISubprogram(linkageName: "compute_step.6", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!55 = !DILocation(line: 13, column: 29, scope: !56)
!56 = !DILexicalBlockFile(scope: !54, file: !7, discriminator: 0)

