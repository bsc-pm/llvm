; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --include-generated-funcs
; RUN: opt %s -passes=ompss-2 -S | FileCheck %s
; ModuleID = 'task_microtask.ll'
source_filename = "task_microtask.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int main() {
;   #pragma oss task microtask(1)
;   {}
; }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !7 {
entry:
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.MICROTASK"(i1 true) ], !dbg !10
  call void @llvm.directive.region.exit(token %0), !dbg !11
  ret i32 0, !dbg !12
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "task_microtask.ll", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 2, column: 3, scope: !7)
!11 = !DILocation(line: 3, column: 4, scope: !7)
!12 = !DILocation(line: 4, column: 1, scope: !7)
; CHECK-LABEL: define {{[^@]+}}@main
; CHECK-SAME: () #[[ATTR0:[0-9]+]] !dbg [[DBG7:![0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = alloca ptr, align 8, !dbg [[DBG10:![0-9]+]]
; CHECK-NEXT:    [[TMP1:%.*]] = alloca ptr, align 8, !dbg [[DBG10]]
; CHECK-NEXT:    [[NUM_DEPS:%.*]] = alloca i64, align 8, !dbg [[DBG10]]
; CHECK-NEXT:    br label [[FINAL_COND:%.*]], !dbg [[DBG10]]
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[TMP0]]), !dbg [[DBG10]]
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[TMP1]]), !dbg [[DBG10]]
; CHECK-NEXT:    store i64 0, ptr [[NUM_DEPS]], align 8, !dbg [[DBG10]]
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr [[NUM_DEPS]], align 8, !dbg [[DBG10]]
; CHECK-NEXT:    call void @nanos6_create_task(ptr @task_info_var_main, ptr @task_invocation_info_main, ptr null, i64 0, ptr [[TMP0]], ptr [[TMP1]], i64 1024, i64 [[TMP2]]), !dbg [[DBG10]]
; CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP0]], align 8, !dbg [[DBG10]]
; CHECK-NEXT:    [[ARGS_END:%.*]] = getelementptr i8, ptr [[TMP3]], i64 0, !dbg [[DBG10]]
; CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP1]], align 8, !dbg [[DBG10]]
; CHECK-NEXT:    call void @nanos6_submit_task(ptr [[TMP4]]), !dbg [[DBG10]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[TMP0]]), !dbg [[DBG10]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[TMP1]]), !dbg [[DBG10]]
; CHECK-NEXT:    br label [[FINAL_END:%.*]], !dbg [[DBG10]]
; CHECK:       final.end:
; CHECK-NEXT:    ret i32 0, !dbg [[DBG11:![0-9]+]]
; CHECK:       final.then:
; CHECK-NEXT:    br label [[FINAL_END]], !dbg [[DBG11]]
; CHECK:       final.cond:
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @nanos6_in_final(), !dbg [[DBG10]]
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i32 [[TMP5]], 0, !dbg [[DBG10]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[FINAL_THEN:%.*]], label [[CODEREPL:%.*]], !dbg [[DBG10]]
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_constructor_check_version() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @nanos6_check_version(i64 1, ptr @nanos6_versions, ptr @[[GLOB0:[0-9]+]])
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_unpacked_task_region_main
; CHECK-SAME: (ptr [[DEVICE_ENV:%.*]], ptr [[ADDRESS_TRANSLATION_TABLE:%.*]]) !dbg [[DBG12:![0-9]+]] {
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label [[TMP0:%.*]], !dbg [[DBG13:![0-9]+]]
; CHECK:       0:
; CHECK-NEXT:    br label [[DOTEXITSTUB:%.*]], !dbg [[DBG14:![0-9]+]]
; CHECK:       .exitStub:
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_ol_task_region_main
; CHECK-SAME: (ptr [[TASK_ARGS:%.*]], ptr [[DEVICE_ENV:%.*]], ptr [[ADDRESS_TRANSLATION_TABLE:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ne ptr [[ADDRESS_TRANSLATION_TABLE]], null
; CHECK-NEXT:    br i1 [[TMP0]], label [[TLATE_IF:%.*]], label [[TLATE_END:%.*]]
; CHECK:       end:
; CHECK-NEXT:    call void @nanos6_unpacked_task_region_main(ptr [[DEVICE_ENV]], ptr [[ADDRESS_TRANSLATION_TABLE]])
; CHECK-NEXT:    ret void
; CHECK:       tlate.if:
; CHECK-NEXT:    br label [[TLATE_END]]
; CHECK:       tlate.end:
; CHECK-NEXT:    br label [[END:%.*]]
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_constructor_register_task_info() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @nanos6_register_task_info(ptr @task_info_var_main)
; CHECK-NEXT:    ret void
;
