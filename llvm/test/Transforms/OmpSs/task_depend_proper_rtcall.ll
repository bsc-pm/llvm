; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_depend_proper_rtcall.ll'
source_filename = "task_depend_proper_rtcall.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32 }

@array = common dso_local global [10 x [20 x i32]] zeroinitializer, align 16
@s = common dso_local global %struct.S zeroinitializer, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() !dbg !6 {
entry:
  %n = alloca i32, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, i64 80, i64 0, i64 80, i64 10, i64 0, i64 10), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, i64 80, i64 0, i64 80, i64 10, i64 0, i64 10), "QUAL.OSS.DEP.IN"([10 x [20 x i32]]* @array, i64 80, i64 0, i64 80, i64 10, i64 0, i64 10) ], !dbg !8
  call void @llvm.directive.region.exit(token %0), !dbg !9
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* @s), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 0), i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 0), i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 0), i64 4, i64 0, i64 4) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %n), "QUAL.OSS.DEP.IN"(i32* %n, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* %n, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* %n, i64 4, i64 0, i64 4) ], !dbg !12
  call void @llvm.directive.region.exit(token %2), !dbg !13
  ret i32 0, !dbg !14
}

; Check we access properly a local variable, a global variable, a constant expression

; CHECK: define internal void @nanos6_unpacked_deps_main0([10 x [20 x i32]]* %array, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast [10 x [20 x i32]]* %array to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %0, i64 80, i64 0, i64 80, i64 10, i64 0, i64 10)
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %0, i64 80, i64 0, i64 80, i64 10, i64 0, i64 10)
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo2(i8* %handler, i32 0, i8* null, i8* %0, i64 80, i64 0, i64 80, i64 10, i64 0, i64 10)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main1(%struct.S* %s, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
; CHECK-NEXT:   %1 = bitcast i32* %0 to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %1, i64 4, i64 0, i64 4)
; CHECK-NEXT:   %2 = bitcast i32* %0 to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %2, i64 4, i64 0, i64 4)
; CHECK-NEXT:   %3 = bitcast i32* %0 to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %3, i64 4, i64 0, i64 4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_main2(i32* %n, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i32* %n to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %0, i64 4, i64 0, i64 4)
; CHECK-NEXT:   %1 = bitcast i32* %n to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %1, i64 4, i64 0, i64 4)
; CHECK-NEXT:   %2 = bitcast i32* %n to i8*
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo1(i8* %handler, i32 0, i8* null, i8* %2, i64 4, i64 0, i64 4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }



declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_depend_proper_rtcall.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !7, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 8, column: 13, scope: !6)
!9 = !DILocation(line: 9, column: 6, scope: !6)
!10 = !DILocation(line: 10, column: 13, scope: !6)
!11 = !DILocation(line: 11, column: 6, scope: !6)
!12 = !DILocation(line: 12, column: 13, scope: !6)
!13 = !DILocation(line: 13, column: 6, scope: !6)
!14 = !DILocation(line: 14, column: 1, scope: !6)


