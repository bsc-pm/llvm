; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'cost.ll'
source_filename = "cost.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._depend_unpack_t = type { i32*, i64, i64, i64 }

; SROA may optimize and make us use the func params directly.

; void foo(int x, int &y) {
;     #pragma oss task cost(x) priority(x) reduction(+: y)
;     {}
; }

; Function Attrs: noinline nounwind
define void @_Z3fooiRi(i32 %x, i32* dereferenceable(4) %y) #0 !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %y), "QUAL.OSS.COST"(i32 %x), "QUAL.OSS.PRIORITY"(i32 %x), "QUAL.OSS.CAPTURED"(i32 %x, i32 %x), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %y, [2 x i8] c"y\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %y), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %y, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %y, void (i32*, i32*, i64)* @red_comb) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !10
  ret void, !dbg !11
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

; Function Attrs: noinline norecurse nounwind
define internal void @red_init(i32* %0, i32* %1, i64 %2) #2 !dbg !12 {
entry:
  %3 = udiv exact i64 %2, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %0, i64 %3
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %0, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %1, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  store i32 0, i32* %0, align 4
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !13
}

; Function Attrs: noinline norecurse nounwind
define internal void @red_comb(i32* %0, i32* %1, i64 %2) #2 !dbg !15 {
entry:
  %3 = udiv exact i64 %2, 4
  %arrayctor.dst.end = getelementptr inbounds i32, i32* %0, i64 %3
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.dst.cur = phi i32* [ %0, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
  %arrayctor.src.cur = phi i32* [ %1, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
  %4 = load i32, i32* %arrayctor.dst.cur, align 4, !dbg !16
  %5 = load i32, i32* %arrayctor.src.cur, align 4, !dbg !16
  %add = add nsw i32 %4, %5, !dbg !16
  store i32 %add, i32* %arrayctor.dst.cur, align 4, !dbg !16
  %arrayctor.dst.next = getelementptr inbounds i32, i32* %arrayctor.dst.cur, i64 1
  %arrayctor.src.next = getelementptr inbounds i32, i32* %arrayctor.src.cur, i64 1
  %arrayctor.done = icmp eq i32* %arrayctor.dst.next, %arrayctor.dst.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void, !dbg !16
}

define internal %struct._depend_unpack_t @compute_dep(i32* %0) {
entry:
  %.fca.0.insert = insertvalue %struct._depend_unpack_t undef, i32* %0, 0
  %.fca.1.insert = insertvalue %struct._depend_unpack_t %.fca.0.insert, i64 4, 1
  %.fca.2.insert = insertvalue %struct._depend_unpack_t %.fca.1.insert, i64 0, 2
  %.fca.3.insert = insertvalue %struct._depend_unpack_t %.fca.2.insert, i64 4, 3
  ret %struct._depend_unpack_t %.fca.3.insert
}

; CHECK: define internal void @nanos6_ol_task_region__Z3fooiRi0(%nanos6_task_args__Z3fooiRi0* %task_args, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK: entry:
; CHECK:   %2 = call %struct._depend_unpack_t @compute_dep(i32* %load_gep_y)
; CHECK: }

; CHECK: define internal void @nanos6_unpacked_constraints__Z3fooiRi0(i32* %y, i32 %x, %nanos6_task_constraints_t* %constraints) {
; CHECK: entry:
; CHECK:   %0 = zext i32 %x to i64
; CHECK: }

; CHECK: define internal void @nanos6_unpacked_priority__Z3fooiRi0(i32* %y, i32 %x, i64* %priority) {
; CHECK: entry:
; CHECK:   %0 = sext i32 %x to i64
; CHECK: }

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { noinline norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "cost.cpp", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, scope: !6)
!10 = !DILocation(line: 3, scope: !6)
!11 = !DILocation(line: 4, scope: !6)
!12 = distinct !DISubprogram(linkageName: "red_init", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 2, scope: !14)
!14 = !DILexicalBlockFile(scope: !12, file: !7, discriminator: 0)
!15 = distinct !DISubprogram(linkageName: "red_comb", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 2, scope: !17)
!17 = !DILexicalBlockFile(scope: !15, file: !7, discriminator: 0)
