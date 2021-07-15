; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop_duplicate_args_vla.ll'
source_filename = "loop_duplicate_args_vla.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i8 }

; This test checks task_args duplicate shared/private/firstprivate vlas

; struct S {
;     S();
;     S(const S&);
;     ~S();
; };
;
; void foo() {
;     int n;
;     S s[n];
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
  %n = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %0 = load i32, i32* %n, align 4, !dbg !9
  %1 = zext i32 %0 to i64, !dbg !10
  %2 = call i8* @llvm.stacksave(), !dbg !10
  store i8* %2, i8** %saved_stack, align 8, !dbg !10
  %vla = alloca %struct.S, i64 %1, align 16, !dbg !10
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !10
  %isempty = icmp eq i64 %1, 0, !dbg !11
  br i1 %isempty, label %arrayctor.cont, label %new.ctorloop, !dbg !11

new.ctorloop:                                     ; preds = %entry
  %arrayctor.end = getelementptr inbounds %struct.S, %struct.S* %vla, i64 %1, !dbg !11
  br label %arrayctor.loop, !dbg !11

arrayctor.loop:                                   ; preds = %arrayctor.loop, %new.ctorloop
  %arrayctor.cur = phi %struct.S* [ %vla, %new.ctorloop ], [ %arrayctor.next, %arrayctor.loop ], !dbg !11
  call void @_ZN1SC1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arrayctor.cur), !dbg !11
  %arrayctor.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.cur, i64 1, !dbg !11
  %arrayctor.done = icmp eq %struct.S* %arrayctor.next, %arrayctor.end, !dbg !11
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !11

arrayctor.cont:                                   ; preds = %entry, %arrayctor.loop
  store i32 0, i32* %i, align 4, !dbg !12
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(%struct.S* %vla), "QUAL.OSS.VLA.DIMS"(%struct.S* %vla, i64 %1), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !13
  call void @llvm.directive.region.exit(token %3), !dbg !14
  store i32 0, i32* %i1, align 4, !dbg !15
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(%struct.S* %vla), "QUAL.OSS.VLA.DIMS"(%struct.S* %vla, i64 %1), "QUAL.OSS.INIT"(%struct.S* %vla, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %vla, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !16
  call void @llvm.directive.region.exit(token %4), !dbg !17
  store i32 0, i32* %i2, align 4, !dbg !18
  %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %vla), "QUAL.OSS.VLA.DIMS"(%struct.S* %vla, i64 %1), "QUAL.OSS.COPY"(%struct.S* %vla, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_), "QUAL.OSS.DEINIT"(%struct.S* %vla, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.4), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.5), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.6), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i64 %1) ], !dbg !19
  call void @llvm.directive.region.exit(token %5), !dbg !20
  %6 = getelementptr inbounds %struct.S, %struct.S* %vla, i64 %1, !dbg !21
  %arraydestroy.isempty = icmp eq %struct.S* %vla, %6, !dbg !21
  br i1 %arraydestroy.isempty, label %arraydestroy.done3, label %arraydestroy.body, !dbg !21

arraydestroy.body:                                ; preds = %arraydestroy.body, %arrayctor.cont
  %arraydestroy.elementPast = phi %struct.S* [ %6, %arrayctor.cont ], [ %arraydestroy.element, %arraydestroy.body ], !dbg !21
  %arraydestroy.element = getelementptr inbounds %struct.S, %struct.S* %arraydestroy.elementPast, i64 -1, !dbg !21
  call void @_ZN1SD1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arraydestroy.element) #3, !dbg !21
  %arraydestroy.done = icmp eq %struct.S* %arraydestroy.element, %vla, !dbg !21
  br i1 %arraydestroy.done, label %arraydestroy.done3, label %arraydestroy.body, !dbg !21

arraydestroy.done3:                               ; preds = %arraydestroy.body, %arrayctor.cont
  %7 = load i8*, i8** %saved_stack, align 8, !dbg !21
  call void @llvm.stackrestore(i8* %7), !dbg !21
  ret void, !dbg !21
}

; Function Attrs: nofree nosync nounwind willreturn
declare i8* @llvm.stacksave() #1

declare void @_ZN1SC1Ev(%struct.S* nonnull align 1 dereferenceable(1)) unnamed_addr #2

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #3

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #3

define internal i32 @compute_lb() #4 !dbg !22 {
entry:
  ret i32 0, !dbg !23
}

define internal i32 @compute_ub() #4 !dbg !25 {
entry:
  ret i32 10, !dbg !26
}

define internal i32 @compute_step() #4 !dbg !28 {
entry:
  ret i32 1, !dbg !29
}

; Function Attrs: noinline norecurse nounwind
define internal void @oss_ctor_ZN1SC1Ev(%struct.S* %0, i64 %1) #5 !dbg !31 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store i64 %1, i64* %.addr1, align 8
  %2 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !32
  %3 = load i64, i64* %.addr1, align 8, !dbg !32
  %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3, !dbg !32
  br label %arrayctor.loop, !dbg !32

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ], !dbg !32
  call void @_ZN1SC1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arrayctor.dst.cur), !dbg !33
  %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1, !dbg !32
  %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end, !dbg !32
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !32

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !33
}

; Function Attrs: nounwind
declare void @_ZN1SD1Ev(%struct.S* nonnull align 1 dereferenceable(1)) unnamed_addr #6

; Function Attrs: noinline norecurse nounwind
define internal void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1) #5 !dbg !35 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store i64 %1, i64* %.addr1, align 8
  %2 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !36
  %3 = load i64, i64* %.addr1, align 8, !dbg !36
  %arraydtor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3, !dbg !36
  br label %arraydtor.loop, !dbg !36

arraydtor.loop:                                   ; preds = %arraydtor.loop, %entry
  %arraydtor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arraydtor.dst.next, %arraydtor.loop ], !dbg !36
  call void @_ZN1SD1Ev(%struct.S* nonnull align 1 dereferenceable(1) %arraydtor.dst.cur) #3, !dbg !36
  %arraydtor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arraydtor.dst.cur, i64 1, !dbg !36
  %arraydtor.done = icmp eq %struct.S* %arraydtor.dst.next, %arraydtor.dst.end, !dbg !36
  br i1 %arraydtor.done, label %arraydtor.cont, label %arraydtor.loop, !dbg !36

arraydtor.cont:                                   ; preds = %arraydtor.loop
  ret void, !dbg !36
}

define internal i32 @compute_lb.1() #4 !dbg !37 {
entry:
  ret i32 0, !dbg !38
}

define internal i32 @compute_ub.2() #4 !dbg !40 {
entry:
  ret i32 10, !dbg !41
}

define internal i32 @compute_step.3() #4 !dbg !43 {
entry:
  ret i32 1, !dbg !44
}

declare void @_ZN1SC1ERKS_(%struct.S* nonnull align 1 dereferenceable(1), %struct.S* nonnull align 1 dereferenceable(1)) unnamed_addr #2

; Function Attrs: noinline norecurse nounwind
define internal void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %0, %struct.S* %1, i64 %2) #5 !dbg !46 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca %struct.S*, align 8
  %.addr2 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store %struct.S* %1, %struct.S** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !47
  %4 = load %struct.S*, %struct.S** %.addr1, align 8, !dbg !47
  %5 = load i64, i64* %.addr2, align 8, !dbg !47
  %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %4, i64 %5, !dbg !47
  br label %arrayctor.loop, !dbg !47

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi %struct.S* [ %4, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ], !dbg !47
  %arrayctor.src.cur = phi %struct.S* [ %3, %entry ], [ %arrayctor.src.next, %arrayctor.loop ], !dbg !47
  call void @_ZN1SC1ERKS_(%struct.S* nonnull align 1 dereferenceable(1) %arrayctor.dst.cur, %struct.S* nonnull align 1 dereferenceable(1) %arrayctor.src.cur), !dbg !48
  %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1, !dbg !47
  %arrayctor.src.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.src.cur, i64 1, !dbg !47
  %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end, !dbg !47
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !47

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !48
}

define internal i32 @compute_lb.4() #4 !dbg !50 {
entry:
  ret i32 0, !dbg !51
}

define internal i32 @compute_ub.5() #4 !dbg !53 {
entry:
  ret i32 10, !dbg !54
}

define internal i32 @compute_step.6() #4 !dbg !56 {
entry:
  ret i32 1, !dbg !57
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.stackrestore(i8*) #1

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov0(%nanos6_task_args__Z3foov0* %task_args_src, %nanos6_task_args__Z3foov0** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:  %0 = load %nanos6_task_args__Z3foov0*, %nanos6_task_args__Z3foov0** %task_args_dst, align 8
; CHECK:  %gep_src_vla = getelementptr %nanos6_task_args__Z3foov0, %nanos6_task_args__Z3foov0* %task_args_src, i32 0, i32 0
; CHECK-NEXT:  %gep_dst_vla1 = getelementptr %nanos6_task_args__Z3foov0, %nanos6_task_args__Z3foov0* %0, i32 0, i32 0
; CHECK-NEXT:  %6 = load %struct.S*, %struct.S** %gep_src_vla, align 8
; CHECK-NEXT:  store %struct.S* %6, %struct.S** %gep_dst_vla1, align 8
; CHECK-NEXT:  %capt_gep_src_ = getelementptr %nanos6_task_args__Z3foov0, %nanos6_task_args__Z3foov0* %task_args_src, i32 0, i32 2
; CHECK-NEXT:  %capt_gep_dst_ = getelementptr %nanos6_task_args__Z3foov0, %nanos6_task_args__Z3foov0* %0, i32 0, i32 2
; CHECK-NEXT:  %7 = load i64, i64* %capt_gep_src_, align 8
; CHECK-NEXT:  store i64 %7, i64* %capt_gep_dst_, align 8

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov1(%nanos6_task_args__Z3foov1* %task_args_src, %nanos6_task_args__Z3foov1** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:  %0 = load %nanos6_task_args__Z3foov1*, %nanos6_task_args__Z3foov1** %task_args_dst, align 8
; CHECK-NEXT:  %1 = bitcast %nanos6_task_args__Z3foov1* %0 to i8*
; CHECK-NEXT:  %args_end = getelementptr i8, i8* %1, i64 32
; CHECK-NEXT:  %gep_dst_vla = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %0, i32 0, i32 0
; CHECK-NEXT:  %2 = bitcast %struct.S** %gep_dst_vla to i8**
; CHECK-NEXT:  store i8* %args_end, i8** %2, align 8
; CHECK:  %gep_src_ = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %task_args_src, i32 0, i32 2
; CHECK-NEXT:  %6 = load i64, i64* %gep_src_, align 8
; CHECK-NEXT:  %7 = mul nuw i64 1, %6
; CHECK-NEXT:  %gep_vla = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %0, i32 0, i32 0
; CHECK-NEXT:  %8 = load %struct.S*, %struct.S** %gep_vla, align 8
; CHECK-NEXT:  call void @oss_ctor_ZN1SC1Ev(%struct.S* %8, i64 %7)
; CHECK-NEXT:  %capt_gep_src_ = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %task_args_src, i32 0, i32 2
; CHECK-NEXT:  %capt_gep_dst_ = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %0, i32 0, i32 2
; CHECK-NEXT:  %9 = load i64, i64* %capt_gep_src_, align 8
; CHECK-NEXT:  store i64 %9, i64* %capt_gep_dst_, align 8

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov2(%nanos6_task_args__Z3foov2* %task_args_src, %nanos6_task_args__Z3foov2** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:  %0 = load %nanos6_task_args__Z3foov2*, %nanos6_task_args__Z3foov2** %task_args_dst, align 8
; CHECK-NEXT:  %1 = bitcast %nanos6_task_args__Z3foov2* %0 to i8*
; CHECK-NEXT:  %args_end = getelementptr i8, i8* %1, i64 32
; CHECK-NEXT:  %gep_dst_vla = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %0, i32 0, i32 1
; CHECK-NEXT:  %2 = bitcast %struct.S** %gep_dst_vla to i8**
; CHECK-NEXT:  store i8* %args_end, i8** %2, align 8
; CHECK:  %gep_src_ = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %task_args_src, i32 0, i32 2
; CHECK-NEXT:  %6 = load i64, i64* %gep_src_, align 8
; CHECK-NEXT:  %7 = mul nuw i64 1, %6
; CHECK-NEXT:  %gep_src_vla = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %task_args_src, i32 0, i32 1
; CHECK-NEXT:  %gep_dst_vla1 = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %0, i32 0, i32 1
; CHECK-NEXT:  %8 = load %struct.S*, %struct.S** %gep_src_vla, align 8
; CHECK-NEXT:  %9 = load %struct.S*, %struct.S** %gep_dst_vla1, align 8
; CHECK-NEXT:  call void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %8, %struct.S* %9, i64 %7)
; CHECK-NEXT:  %capt_gep_src_ = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %task_args_src, i32 0, i32 2
; CHECK-NEXT:  %capt_gep_dst_ = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %0, i32 0, i32 2
; CHECK-NEXT:  %10 = load i64, i64* %capt_gep_src_, align 8
; CHECK-NEXT:  store i64 %10, i64* %capt_gep_dst_, align 8

attributes #0 = { noinline nounwind optnone mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nofree nosync nounwind willreturn }
attributes #2 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #3 = { nounwind }
attributes #4 = { "min-legal-vector-width"="0" }
attributes #5 = { noinline norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #6 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

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
!7 = !DIFile(filename: "loop_duplicate_args_vla.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 8, column: 9, scope: !6)
!10 = !DILocation(line: 8, column: 5, scope: !6)
!11 = !DILocation(line: 8, column: 7, scope: !6)
!12 = !DILocation(line: 10, column: 14, scope: !6)
!13 = !DILocation(line: 10, column: 10, scope: !6)
!14 = !DILocation(line: 10, column: 35, scope: !6)
!15 = !DILocation(line: 12, column: 14, scope: !6)
!16 = !DILocation(line: 12, column: 10, scope: !6)
!17 = !DILocation(line: 12, column: 35, scope: !6)
!18 = !DILocation(line: 14, column: 14, scope: !6)
!19 = !DILocation(line: 14, column: 10, scope: !6)
!20 = !DILocation(line: 14, column: 35, scope: !6)
!21 = !DILocation(line: 15, column: 1, scope: !6)
!22 = distinct !DISubprogram(linkageName: "compute_lb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!23 = !DILocation(line: 10, column: 18, scope: !24)
!24 = !DILexicalBlockFile(scope: !22, file: !7, discriminator: 0)
!25 = distinct !DISubprogram(linkageName: "compute_ub", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 10, column: 25, scope: !27)
!27 = !DILexicalBlockFile(scope: !25, file: !7, discriminator: 0)
!28 = distinct !DISubprogram(linkageName: "compute_step", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!29 = !DILocation(line: 10, column: 29, scope: !30)
!30 = !DILexicalBlockFile(scope: !28, file: !7, discriminator: 0)
!31 = distinct !DISubprogram(linkageName: "oss_ctor_ZN1SC1Ev", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!32 = !DILocation(line: 0, scope: !31)
!33 = !DILocation(line: 11, column: 34, scope: !34)
!34 = !DILexicalBlockFile(scope: !31, file: !7, discriminator: 0)
!35 = distinct !DISubprogram(linkageName: "oss_dtor_ZN1SD1Ev", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!36 = !DILocation(line: 0, scope: !35)
!37 = distinct !DISubprogram(linkageName: "compute_lb.1", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!38 = !DILocation(line: 12, column: 18, scope: !39)
!39 = !DILexicalBlockFile(scope: !37, file: !7, discriminator: 0)
!40 = distinct !DISubprogram(linkageName: "compute_ub.2", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!41 = !DILocation(line: 12, column: 25, scope: !42)
!42 = !DILexicalBlockFile(scope: !40, file: !7, discriminator: 0)
!43 = distinct !DISubprogram(linkageName: "compute_step.3", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!44 = !DILocation(line: 12, column: 29, scope: !45)
!45 = !DILexicalBlockFile(scope: !43, file: !7, discriminator: 0)
!46 = distinct !DISubprogram(linkageName: "oss_copy_ctor_ZN1SC1ERKS_", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!47 = !DILocation(line: 0, scope: !46)
!48 = !DILocation(line: 13, column: 39, scope: !49)
!49 = !DILexicalBlockFile(scope: !46, file: !7, discriminator: 0)
!50 = distinct !DISubprogram(linkageName: "compute_lb.4", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!51 = !DILocation(line: 14, column: 18, scope: !52)
!52 = !DILexicalBlockFile(scope: !50, file: !7, discriminator: 0)
!53 = distinct !DISubprogram(linkageName: "compute_ub.5", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!54 = !DILocation(line: 14, column: 25, scope: !55)
!55 = !DILexicalBlockFile(scope: !53, file: !7, discriminator: 0)
!56 = distinct !DISubprogram(linkageName: "compute_step.6", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!57 = !DILocation(line: 14, column: 29, scope: !58)
!58 = !DILexicalBlockFile(scope: !56, file: !7, discriminator: 0)


