; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_no_const_label.c'
source_filename = "task_no_const_label.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.foo.text = private unnamed_addr constant [3 x i8] c"T1\00", align 1

; void foo() {
;     const char text[] = "T1";
;     #pragma oss task label(text)
;     {}
; }

; Function Attrs: noinline nounwind optnone
define void @foo() #0 !dbg !6 {
entry:
  %text = alloca [3 x i8], align 1
  %0 = bitcast [3 x i8]* %text to i8*, !dbg !9
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 getelementptr inbounds ([3 x i8], [3 x i8]* @__const.foo.text, i32 0, i32 0), i64 3, i1 false), !dbg !9
  %arraydecay = getelementptr inbounds [3 x i8], [3 x i8]* %text, i64 0, i64 0, !dbg !10
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.LABEL"(i8* %arraydecay) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !11
  ret void, !dbg !12
}

; store the label ptr into nanos6_task_implementation_info_t
; CHECK: store i8* %arraydecay, i8** getelementptr inbounds ([1 x %nanos6_task_implementation_info_t], [1 x %nanos6_task_implementation_info_t]* @implementations_var_foo0, i32 0, i32 0, i32 3), align 8

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_no_const_label.c", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, scope: !6)
!10 = !DILocation(line: 3, scope: !6)
!11 = !DILocation(line: 4, scope: !6)
!12 = !DILocation(line: 5, scope: !6)
