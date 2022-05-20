; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_label.c'
source_filename = "task_label.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo() {
;     const char text1[] = "T2";
;     #pragma oss task label("T1", text1)
;     {}
; }

@__const.foo.text1 = private unnamed_addr constant [3 x i8] c"T2\00", align 1
@.str = private unnamed_addr constant [3 x i8] c"T1\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !7 {
entry:
  %text1 = alloca [3 x i8], align 1
  %0 = bitcast [3 x i8]* %text1 to i8*, !dbg !10
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 getelementptr inbounds ([3 x i8], [3 x i8]* @__const.foo.text1, i32 0, i32 0), i64 3, i1 false), !dbg !10
  %arraydecay = getelementptr inbounds [3 x i8], [3 x i8]* %text1, i64 0, i64 0, !dbg !11
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.LABEL"(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %arraydecay) ], !dbg !12
  call void @llvm.directive.region.exit(token %1), !dbg !13
  ret void, !dbg !14
}

; CHECK: call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_foo0, %nanos6_task_invocation_info_t* @task_invocation_info_foo0, i8* %arraydecay, i64 0, i8** %3, i8** %2, i64 0, i64 %4)

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nounwind willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "label2.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 2, column: 16, scope: !7)
!11 = !DILocation(line: 3, column: 34, scope: !7)
!12 = !DILocation(line: 3, column: 13, scope: !7)
!13 = !DILocation(line: 4, column: 6, scope: !7)
!14 = !DILocation(line: 5, column: 1, scope: !7)
