; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_args_func_param.ll'
source_filename = "task_args_func_param.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }

; SROA may optimize and make us use the func params directly.

; void foo(int x, int &y) {
;     #pragma oss task cost(x) priority(x) reduction(+: y)
;     {}
; }

; Function Attrs: nounwind mustprogress
define dso_local void @_Z3fooiRi(i32 %x, i32* nonnull align 4 dereferenceable(4) %y) #0 !dbg !6 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4, !tbaa !9
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %y), "QUAL.OSS.FIRSTPRIVATE"(i32* %x.addr), "QUAL.OSS.COST"(i32 (i32*)* @compute_cost, i32* %x.addr), "QUAL.OSS.PRIORITY"(i32 (i32*)* @compute_priority, i32* %x.addr), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %y, [2 x i8] c"y\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %y), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %y, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %y, void (i32*, i32*, i64)* @red_comb) ], !dbg !13
  call void @llvm.directive.region.exit(token %0), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal i32 @compute_cost(i32* %x) !dbg !16 {
entry:
  %0 = load i32, i32* %x, align 4, !dbg !17, !tbaa !9
  ret i32 %0, !dbg !17
}

define internal i32 @compute_priority(i32* %x) !dbg !19 {
entry:
  %0 = load i32, i32* %x, align 4, !dbg !20, !tbaa !9
  ret i32 %0, !dbg !20
}

; Function Attrs: norecurse nounwind
define internal void @red_init(i32* %0, i32* %1, i64 %2) #2 !dbg !22 {
entry:
  %3 = udiv exact i64 %2, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %0, i64 %3
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %0, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %1, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  store i32 0, i32* %arrayctor.dst.cur, align 4, !tbaa !9
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !23
}

; Function Attrs: norecurse nounwind
define internal void @red_comb(i32* %0, i32* %1, i64 %2) #2 !dbg !25 {
entry:
  %3 = udiv exact i64 %2, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %0, i64 %3
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %0, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %1, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  %4 = load i32, i32* %arrayctor.dst.cur, align 4, !dbg !26, !tbaa !9
  %5 = load i32, i32* %arrayctor.src.cur, align 4, !dbg !26, !tbaa !9
  %add = add nsw i32 %4, %5, !dbg !28
  store i32 %add, i32* %arrayctor.dst.cur, align 4, !dbg !28, !tbaa !9
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !26
}

define internal %struct._depend_unpack_t @compute_dep(i32* %y) !dbg !29 {
entry:
  %.fca.0.insert = insertvalue %struct._depend_unpack_t undef, i32* %y, 0
  %.fca.1.insert = insertvalue %struct._depend_unpack_t %.fca.0.insert, i64 4, 1
  %.fca.2.insert = insertvalue %struct._depend_unpack_t %.fca.1.insert, i64 0, 2
  %.fca.3.insert = insertvalue %struct._depend_unpack_t %.fca.2.insert, i64 4, 3
  ret %struct._depend_unpack_t %.fca.3.insert
}

; CHECK: define internal void @nanos6_ol_task_region__Z3fooiRi0(%nanos6_task_args__Z3fooiRi0* %task_args, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK: entry:
; CHECK:   %2 = call %struct._depend_unpack_t @compute_dep(i32* %load_gep_y)

; CHECK: define internal void @nanos6_unpacked_constraints__Z3fooiRi0(i32* %y, i32* %x.addr, %nanos6_task_constraints_t* %constraints) {
; CHECK: entry:
; CHECK:   %0 = call i32 @compute_cost(i32* %x.addr)
; CHECK:   %1 = zext i32 %0 to i64

; CHECK: define internal void @nanos6_unpacked_priority__Z3fooiRi0(i32* %y, i32* %x.addr, i64* %priority) {
; CHECK: entry:
; CHECK:   %0 = call i32 @compute_priority(i32* %x.addr)
; CHECK:   %1 = sext i32 %0 to i64

attributes #0 = { nounwind mustprogress "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind }
attributes #2 = { norecurse nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 13.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_args_func_param.cpp", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C++ TBAA"}
!13 = !DILocation(line: 2, column: 13, scope: !6)
!14 = !DILocation(line: 3, column: 6, scope: !6)
!15 = !DILocation(line: 4, column: 1, scope: !6)
!16 = distinct !DISubprogram(linkageName: "compute_cost", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!17 = !DILocation(line: 2, column: 27, scope: !18)
!18 = !DILexicalBlockFile(scope: !16, file: !7, discriminator: 0)
!19 = distinct !DISubprogram(linkageName: "compute_priority", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 2, column: 39, scope: !21)
!21 = !DILexicalBlockFile(scope: !19, file: !7, discriminator: 0)
!22 = distinct !DISubprogram(linkageName: "red_init", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!23 = !DILocation(line: 2, column: 55, scope: !24)
!24 = !DILexicalBlockFile(scope: !22, file: !7, discriminator: 0)
!25 = distinct !DISubprogram(linkageName: "red_comb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 2, column: 55, scope: !27)
!27 = !DILexicalBlockFile(scope: !25, file: !7, discriminator: 0)
!28 = !DILocation(line: 2, column: 52, scope: !27)
!29 = distinct !DISubprogram(linkageName: "compute_dep", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
