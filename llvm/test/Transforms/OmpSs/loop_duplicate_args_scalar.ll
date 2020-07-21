; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop_duplicate_args.ll'
source_filename = "loop_duplicate_args.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i8 }

; This test checks nonpod shared/private/firstprivate

; struct S {
;     S();
;     S(const S&);
;     ~S();
; };
;
; void foo() {
;     S s;
;     #pragma oss task for shared(s)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss task for private(s)
;     for (int i = 0; i < 10; ++i) {}
;     #pragma oss task for firstprivate(s)
;     for (int i = 0; i < 10; ++i) {}
; }

; Function Attrs: noinline nounwind optnone
define void @_Z3foov() #0 !dbg !6 {
entry:
  %s = alloca %struct.S, align 1
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  call void @_ZN1SC1Ev(%struct.S* %s), !dbg !9
  store i32 0, i32* %i, align 4, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(%struct.S* %s), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !10
  call void @llvm.directive.region.exit(token %0), !dbg !10
  store i32 0, i32* %i1, align 4, !dbg !11
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(%struct.S* %s), "QUAL.OSS.INIT"(%struct.S* %s, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !11
  call void @llvm.directive.region.exit(token %1), !dbg !11
  store i32 0, i32* %i2, align 4, !dbg !12
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %s), "QUAL.OSS.COPY"(%struct.S* %s, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_), "QUAL.OSS.DEINIT"(%struct.S* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !12
  call void @llvm.directive.region.exit(token %2), !dbg !12
  call void @_ZN1SD1Ev(%struct.S* %s) #2, !dbg !13
  ret void, !dbg !13
}

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov0(%nanos6_task_args__Z3foov0* %task_args_src, %nanos6_task_args__Z3foov0** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:  %0 = load %nanos6_task_args__Z3foov0*, %nanos6_task_args__Z3foov0** %task_args_dst, align 8
; CHECK:  %gep_src_s = getelementptr %nanos6_task_args__Z3foov0, %nanos6_task_args__Z3foov0* %task_args_src, i32 0, i32 0
; CHECK-NEXT:  %gep_dst_s = getelementptr %nanos6_task_args__Z3foov0, %nanos6_task_args__Z3foov0* %0, i32 0, i32 0
; CHECK-NEXT:  %2 = load %struct.S*, %struct.S** %gep_src_s, align 8
; CHECK-NEXT:  store %struct.S* %2, %struct.S** %gep_dst_s, align 8

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov1(%nanos6_task_args__Z3foov1* %task_args_src, %nanos6_task_args__Z3foov1** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:   %0 = load %nanos6_task_args__Z3foov1*, %nanos6_task_args__Z3foov1** %task_args_dst, align 8
; CHECK:  %gep_s = getelementptr %nanos6_task_args__Z3foov1, %nanos6_task_args__Z3foov1* %0, i32 0, i32 0
; CHECK-NEXT:  call void @oss_ctor_ZN1SC1Ev(%struct.S* %gep_s, i64 1)

; CHECK: define internal void @nanos6_ol_duplicate__Z3foov2(%nanos6_task_args__Z3foov2* %task_args_src, %nanos6_task_args__Z3foov2** %task_args_dst) {
; CHECK: entry:
; CHECK-NEXT:   %0 = load %nanos6_task_args__Z3foov2*, %nanos6_task_args__Z3foov2** %task_args_dst, align 8
; CHECK:  %gep_src_s = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %task_args_src, i32 0, i32 1
; CHECK-NEXT:  %gep_dst_s = getelementptr %nanos6_task_args__Z3foov2, %nanos6_task_args__Z3foov2* %0, i32 0, i32 1
; CHECK-NEXT:  call void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %gep_src_s, %struct.S* %gep_dst_s, i64 1)

declare void @_ZN1SC1Ev(%struct.S*) unnamed_addr #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

; Function Attrs: noinline norecurse nounwind
define internal void @oss_ctor_ZN1SC1Ev(%struct.S* %0, i64 %1) #3 !dbg !14 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store i64 %1, i64* %.addr1, align 8
  %2 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !15
  %3 = load i64, i64* %.addr1, align 8, !dbg !15
  %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3, !dbg !15
  br label %arrayctor.loop, !dbg !15

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ], !dbg !15
  call void @_ZN1SC1Ev(%struct.S* %arrayctor.dst.cur), !dbg !16
  %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1, !dbg !15
  %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end, !dbg !15
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !15

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !16
}

; Function Attrs: nounwind
declare void @_ZN1SD1Ev(%struct.S*) unnamed_addr #4

; Function Attrs: noinline norecurse nounwind
define internal void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1) #3 !dbg !18 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store i64 %1, i64* %.addr1, align 8
  %2 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !19
  %3 = load i64, i64* %.addr1, align 8, !dbg !19
  %arraydtor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3, !dbg !19
  br label %arraydtor.loop, !dbg !19

arraydtor.loop:                                   ; preds = %arraydtor.loop, %entry
  %arraydtor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arraydtor.dst.next, %arraydtor.loop ], !dbg !19
  call void @_ZN1SD1Ev(%struct.S* %arraydtor.dst.cur) #2, !dbg !19
  %arraydtor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arraydtor.dst.cur, i64 1, !dbg !19
  %arraydtor.done = icmp eq %struct.S* %arraydtor.dst.next, %arraydtor.dst.end, !dbg !19
  br i1 %arraydtor.done, label %arraydtor.cont, label %arraydtor.loop, !dbg !19

arraydtor.cont:                                   ; preds = %arraydtor.loop
  ret void, !dbg !19
}

declare void @_ZN1SC1ERKS_(%struct.S*, %struct.S* dereferenceable(1)) unnamed_addr #1

; Function Attrs: noinline norecurse nounwind
define internal void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %0, %struct.S* %1, i64 %2) #3 !dbg !20 {
entry:
  %.addr = alloca %struct.S*, align 8
  %.addr1 = alloca %struct.S*, align 8
  %.addr2 = alloca i64, align 8
  store %struct.S* %0, %struct.S** %.addr, align 8
  store %struct.S* %1, %struct.S** %.addr1, align 8
  store i64 %2, i64* %.addr2, align 8
  %3 = load %struct.S*, %struct.S** %.addr, align 8, !dbg !21
  %4 = load %struct.S*, %struct.S** %.addr1, align 8, !dbg !21
  %5 = load i64, i64* %.addr2, align 8, !dbg !21
  %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %4, i64 %5, !dbg !21
  br label %arrayctor.loop, !dbg !21

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi %struct.S* [ %4, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ], !dbg !21
  %arrayctor.src.cur = phi %struct.S* [ %3, %entry ], [ %arrayctor.src.next, %arrayctor.loop ], !dbg !21
  call void @_ZN1SC1ERKS_(%struct.S* %arrayctor.dst.cur, %struct.S* dereferenceable(1) %arrayctor.src.cur), !dbg !22
  %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1, !dbg !21
  %arrayctor.src.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.src.cur, i64 1, !dbg !21
  %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end, !dbg !21
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop, !dbg !21

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !22
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { noinline norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "loop_duplicate_args.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 8, scope: !6)
!10 = !DILocation(line: 10, scope: !6)
!11 = !DILocation(line: 12, scope: !6)
!12 = !DILocation(line: 14, scope: !6)
!13 = !DILocation(line: 15, scope: !6)
!14 = distinct !DISubprogram(linkageName: "oss_ctor_ZN1SC1Ev", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 0, scope: !14)
!16 = !DILocation(line: 11, scope: !17)
!17 = !DILexicalBlockFile(scope: !14, file: !7, discriminator: 0)
!18 = distinct !DISubprogram(linkageName: "oss_dtor_ZN1SD1Ev", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!19 = !DILocation(line: 0, scope: !18)
!20 = distinct !DISubprogram(linkageName: "oss_copy_ctor_ZN1SC1ERKS_", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 0, scope: !20)
!22 = !DILocation(line: 13, scope: !23)
!23 = !DILexicalBlockFile(scope: !20, file: !7, discriminator: 0)
