; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --include-generated-funcs
; RUN: opt %s -passes=ompss-2 -S | FileCheck %s
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

; Function Attrs: noinline nounwind optnone
define dso_local void @foo() #0 !dbg !5 {
entry:
  %text1 = alloca [3 x i8], align 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %text1, ptr align 1 @__const.foo.text1, i64 3, i1 false), !dbg !9
  %arraydecay = getelementptr inbounds [3 x i8], ptr %text1, i64 0, i64 0, !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.LABEL"(ptr @.str, ptr %arraydecay) ], !dbg !11
  call void @llvm.directive.region.exit(token %0), !dbg !12
  ret void, !dbg !13
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { argmemonly nofree nounwind willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!""}
!5 = distinct !DISubprogram(name: "foo", scope: !6, file: !6, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DIFile(filename: "task_label.ll", directory: "")
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = !DILocation(line: 2, column: 16, scope: !5)
!10 = !DILocation(line: 3, column: 34, scope: !5)
!11 = !DILocation(line: 3, column: 13, scope: !5)
!12 = !DILocation(line: 4, column: 6, scope: !5)
!13 = !DILocation(line: 5, column: 1, scope: !5)
; CHECK-LABEL: define {{[^@]+}}@foo
; CHECK-SAME: () #[[ATTR0:[0-9]+]] !dbg [[DBG5:![0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TEXT1:%.*]] = alloca [3 x i8], align 1
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 1 [[TEXT1]], ptr align 1 @__const.foo.text1, i64 3, i1 false), !dbg [[DBG9:![0-9]+]]
; CHECK-NEXT:    [[ARRAYDECAY:%.*]] = getelementptr inbounds [3 x i8], ptr [[TEXT1]], i64 0, i64 0, !dbg [[DBG10:![0-9]+]]
; CHECK-NEXT:    [[TMP0:%.*]] = alloca ptr, align 8, !dbg [[DBG11:![0-9]+]]
; CHECK-NEXT:    [[TMP1:%.*]] = alloca ptr, align 8, !dbg [[DBG11]]
; CHECK-NEXT:    [[NUM_DEPS:%.*]] = alloca i64, align 8, !dbg [[DBG11]]
; CHECK-NEXT:    br label [[FINAL_COND:%.*]], !dbg [[DBG11]]
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[TMP0]]), !dbg [[DBG11]]
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr [[TMP1]]), !dbg [[DBG11]]
; CHECK-NEXT:    store i64 0, ptr [[NUM_DEPS]], align 8, !dbg [[DBG11]]
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr [[NUM_DEPS]], align 8, !dbg [[DBG11]]
; CHECK-NEXT:    call void @nanos6_create_task(ptr @task_info_var_foo, ptr @task_invocation_info_foo, ptr [[ARRAYDECAY]], i64 0, ptr [[TMP0]], ptr [[TMP1]], i64 0, i64 [[TMP2]]), !dbg [[DBG11]]
; CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP0]], align 8, !dbg [[DBG11]]
; CHECK-NEXT:    [[ARGS_END:%.*]] = getelementptr i8, ptr [[TMP3]], i64 0, !dbg [[DBG11]]
; CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP1]], align 8, !dbg [[DBG11]]
; CHECK-NEXT:    call void @nanos6_submit_task(ptr [[TMP4]]), !dbg [[DBG11]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[TMP0]]), !dbg [[DBG11]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr [[TMP1]]), !dbg [[DBG11]]
; CHECK-NEXT:    br label [[FINAL_END:%.*]], !dbg [[DBG11]]
; CHECK:       final.end:
; CHECK-NEXT:    ret void, !dbg [[DBG12:![0-9]+]]
; CHECK:       final.then:
; CHECK-NEXT:    br label [[FINAL_END]], !dbg [[DBG12]]
; CHECK:       final.cond:
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @nanos6_in_final(), !dbg [[DBG11]]
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ne i32 [[TMP5]], 0, !dbg [[DBG11]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[FINAL_THEN:%.*]], label [[CODEREPL:%.*]], !dbg [[DBG11]]
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_constructor_check_version() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @nanos6_check_version(i64 1, ptr @nanos6_versions, ptr @[[GLOB0:[0-9]+]])
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define {{[^@]+}}@nanos6_unpacked_task_region_foo
; CHECK-SAME: (ptr [[DEVICE_ENV:%.*]], ptr [[ADDRESS_TRANSLATION_TABLE:%.*]]) !dbg [[DBG13:![0-9]+]] {
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label [[TMP0:%.*]], !dbg [[DBG14:![0-9]+]]
; CHECK:       0:
; CHECK-NEXT:    br label [[DOTEXITSTUB:%.*]], !dbg [[DBG15:![0-9]+]]
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
