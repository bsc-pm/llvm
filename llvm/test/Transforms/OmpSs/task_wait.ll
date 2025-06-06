; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --include-generated-funcs
; RUN: opt %s -passes=ompss-2 -S | FileCheck %s
; ModuleID = 'task_wait.c'
source_filename = "task_wait.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo() {
;     #pragma oss task wait
;     {}
; }

; wait clause adds 1 << 4 -> 16 to flags

; Function Attrs: noinline nounwind optnone
define void @foo() #0 !dbg !6 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.WAIT"(i1 true) ], !dbg !9
  call void @llvm.directive.region.exit(token %0), !dbg !10
  ret void, !dbg !11
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 11.0.0 "}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "task_wait.c", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, scope: !6)
!10 = !DILocation(line: 3, scope: !6)
!11 = !DILocation(line: 4, scope: !6)
; CHECK-LABEL: define {{[^@]+}}@foo
; CHECK-SAME: () #[[ATTR0:[0-9]+]] !dbg [[DBG6:![0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = alloca ptr, align 8, !dbg [[DBG9:![0-9]+]]
; CHECK-NEXT:    [[TMP1:%.*]] = alloca ptr, align 8, !dbg [[DBG9]]
; CHECK-NEXT:    [[NUM_DEPS:%.*]] = alloca i64, align 8, !dbg [[DBG9]]
; CHECK-NEXT:    br label [[FINAL_COND:%.*]], !dbg [[DBG9]]
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[TMP0]]), !dbg [[DBG9]]
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[TMP1]]), !dbg [[DBG9]]
; CHECK-NEXT:    store i64 0, ptr [[NUM_DEPS]], align 8, !dbg [[DBG9]]
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr [[NUM_DEPS]], align 8, !dbg [[DBG9]]
; CHECK-NEXT:    call void @nanos6_create_task(ptr @task_info_var_foo, ptr @task_invocation_info_foo, ptr null, i64 0, ptr [[TMP0]], ptr [[TMP1]], i64 16, i64 [[TMP2]]), !dbg [[DBG9]]
; CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP0]], align 8, !dbg [[DBG9]]
; CHECK-NEXT:    [[ARGS_END:%.*]] = getelementptr i8, ptr [[TMP3]], i64 0, !dbg [[DBG9]]
; CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP1]], align 8, !dbg [[DBG9]]
; CHECK-NEXT:    call void @nanos6_submit_task(ptr [[TMP4]]), !dbg [[DBG9]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[TMP0]]), !dbg [[DBG9]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[TMP1]]), !dbg [[DBG9]]
; CHECK-NEXT:    br label [[FINAL_END:%.*]], !dbg [[DBG9]]
; CHECK:       final.end:
; CHECK-NEXT:    ret void, !dbg [[DBG10:![0-9]+]]
; CHECK:       final.then:
; CHECK-NEXT:    br label [[FINAL_END]], !dbg [[DBG10]]
; CHECK:       final.cond:
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @nanos6_in_final(), !dbg [[DBG9]]
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i32 [[TMP5]], 0, !dbg [[DBG9]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[FINAL_THEN:%.*]], label [[CODEREPL:%.*]], !dbg [[DBG9]]
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_constructor_check_version() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @nanos6_check_version(i64 1, ptr @nanos6_versions, ptr @[[GLOB0:[0-9]+]])
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_unpacked_task_region_foo
; CHECK-SAME: (ptr [[DEVICE_ENV:%.*]], ptr [[ADDRESS_TRANSLATION_TABLE:%.*]]) !dbg [[DBG11:![0-9]+]] {
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label [[TMP0:%.*]], !dbg [[DBG12:![0-9]+]]
; CHECK:       0:
; CHECK-NEXT:    br label [[DOTEXITSTUB:%.*]], !dbg [[DBG13:![0-9]+]]
; CHECK:       .exitStub:
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_ol_task_region_foo
; CHECK-SAME: (ptr [[TASK_ARGS:%.*]], ptr [[DEVICE_ENV:%.*]], ptr [[ADDRESS_TRANSLATION_TABLE:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ne ptr [[ADDRESS_TRANSLATION_TABLE]], null
; CHECK-NEXT:    br i1 [[TMP0]], label [[TLATE_IF:%.*]], label [[TLATE_END:%.*]]
; CHECK:       end:
; CHECK-NEXT:    call void @nanos6_unpacked_task_region_foo(ptr [[DEVICE_ENV]], ptr [[ADDRESS_TRANSLATION_TABLE]])
; CHECK-NEXT:    ret void
; CHECK:       tlate.if:
; CHECK-NEXT:    br label [[TLATE_END]]
; CHECK:       tlate.end:
; CHECK-NEXT:    br label [[END:%.*]]
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_constructor_register_task_info() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @nanos6_register_task_info(ptr @task_info_var_foo)
; CHECK-NEXT:    ret void
;
