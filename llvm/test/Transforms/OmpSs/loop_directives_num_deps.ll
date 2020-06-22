; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'loop_directives_num_deps.ll'
source_filename = "loop_directives_num_deps.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; taskloop directives have -1 num dependencies

; void foo(int lb, int ub, int step) {
;     #pragma oss task for
;     for (int i = 0; i < 10; i += 1) {}
;     #pragma oss taskloop
;     for (int i = 0; i < 10; i += 1) {}
;     #pragma oss taskloop for
;     for (int i = 0; i < 10; i += 1) {}
; }

; Function Attrs: noinline nounwind optnone
define void @foo(i32 %lb, i32 %ub, i32 %step) #0 !dbg !6 {
entry:
  %lb.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  store i32 %lb, i32* %lb.addr, align 4
  store i32 %ub, i32* %ub.addr, align 4
  store i32 %step, i32* %step.addr, align 4
  store i32 0, i32* %i, align 4, !dbg !9
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !9
  store i32 0, i32* %i1, align 4, !dbg !10
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !10
  call void @llvm.directive.region.exit(token %1), !dbg !10
  store i32 0, i32* %i2, align 4, !dbg !11
  %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1) ], !dbg !11
  call void @llvm.directive.region.exit(token %2), !dbg !11
  ret void, !dbg !12
}
; CHECK: call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_foo0, %nanos6_task_invocation_info_t* @task_invocation_info_foo0, i64 16, i8** %1, i8** %2, i64 8, i64 0)
; CHECK: call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_foo1, %nanos6_task_invocation_info_t* @task_invocation_info_foo1, i64 16, i8** %7, i8** %8, i64 4, i64 -1)
; CHECK: call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_foo2, %nanos6_task_invocation_info_t* @task_invocation_info_foo2, i64 16, i8** %13, i8** %14, i64 12, i64 -1)

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "loop_directives_num_deps.ll", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, scope: !6)
!10 = !DILocation(line: 5, scope: !6)
!11 = !DILocation(line: 7, scope: !6)
!12 = !DILocation(line: 8, scope: !6)
