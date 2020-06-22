; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_final_exception.ll'
source_filename = "task_final_exception.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; struct S {
;     int x;
;     S() {}
;     S(const S &s) {}
;     ~S() {}
; };
;
; void foo(S const & s){}
;
; int main() {
;     S s;
;     #pragma oss task
;     foo(s);
; }

%struct.S = type { i32 }

$_ZN1SC2Ev = comdat any

$_ZN1SC2ERKS_ = comdat any

$_ZN1SD2Ev = comdat any

$__clang_call_terminate = comdat any

; Function Attrs: noinline nounwind optnone uwtable
declare void @_Z3fooRK1S(%struct.S* dereferenceable(4) %s)

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !9 {
entry:
  %s = alloca %struct.S, align 4
  call void @_ZN1SC2Ev(%struct.S* %s), !dbg !10
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %s), "QUAL.OSS.COPY"(%struct.S* %s, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERKS_), "QUAL.OSS.DEINIT"(%struct.S* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ], !dbg !11
  invoke void @_Z3fooRK1S(%struct.S* dereferenceable(4) %s)
          to label %invoke.cont unwind label %terminate.lpad, !dbg !12

invoke.cont:                                      ; preds = %entry
  call void @llvm.directive.region.exit(token %0), !dbg !12
  call void @_ZN1SD2Ev(%struct.S* %s) #2, !dbg !13
  ret i32 0, !dbg !13

terminate.lpad:                                   ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null, !dbg !12
  %2 = extractvalue { i8*, i32 } %1, 0, !dbg !12
  call void @__clang_call_terminate(i8* %2) #5, !dbg !12
  unreachable, !dbg !12
}

; CHECK: entry:
; CHECK-NEXT:   %s = alloca %struct.S, align 4
; CHECK-NEXT:   call void @_ZN1SC2Ev(%struct.S* %s), !dbg !8
; CHECK-NEXT:   br label %final.cond, !dbg !9
; CHECK: codeRepl:                                         ; preds = %final.cond
; CHECK-NEXT:   %0 = alloca %nanos6_task_args_main0*, align 8, !dbg !9
; CHECK-NEXT:   %1 = bitcast %nanos6_task_args_main0** %0 to i8**, !dbg !9
; CHECK-NEXT:   %2 = alloca i8*, align 8, !dbg !9
; CHECK-NEXT:   call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_main0, %nanos6_task_invocation_info_t* @task_invocation_info_main0, i64 16, i8** %1, i8** %2, i64 0, i64 0), !dbg !9
; CHECK-NEXT:   %3 = load %nanos6_task_args_main0*, %nanos6_task_args_main0** %0, align 8, !dbg !9
; CHECK-NEXT:   %4 = bitcast %nanos6_task_args_main0* %3 to i8*, !dbg !9
; CHECK-NEXT:   %args_end = getelementptr i8, i8* %4, i64 16, !dbg !9
; CHECK-NEXT:   %gep_s = getelementptr %nanos6_task_args_main0, %nanos6_task_args_main0* %3, i32 0, i32 0, !dbg !9
; CHECK-NEXT:   call void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %s, %struct.S* %gep_s, i64 1), !dbg !9
; CHECK-NEXT:   %5 = load i8*, i8** %2, align 8, !dbg !9
; CHECK-NEXT:   call void @nanos6_submit_task(i8* %5), !dbg !9
; CHECK-NEXT:   br label %final.end, !dbg !9
; CHECK: final.end:                                        ; preds = %codeRepl, %invoke.cont.clone
; CHECK-NEXT:   call void @_ZN1SD2Ev(%struct.S* %s) #1, !dbg !10
; CHECK-NEXT:   ret i32 0, !dbg !10
; CHECK: final.then:                                       ; preds = %final.cond
; CHECK-NEXT:   invoke void @_Z3fooRK1S(%struct.S* dereferenceable(4) %s)
; CHECK-NEXT:           to label %invoke.cont.clone unwind label %terminate.lpad.clone, !dbg !11
; CHECK: invoke.cont.clone:                                ; preds = %final.then
; CHECK-NEXT:   br label %final.end, !dbg !10
; CHECK: terminate.lpad.clone:                             ; preds = %final.then
; CHECK-NEXT:   %6 = landingpad { i8*, i32 }
; CHECK-NEXT:           catch i8* null, !dbg !11
; CHECK-NEXT:   %7 = extractvalue { i8*, i32 } %6, 0, !dbg !11
; CHECK-NEXT:   call void @__clang_call_terminate(i8* %7) #2, !dbg !11
; CHECK-NEXT:   unreachable, !dbg !11
; CHECK: final.cond:                                       ; preds = %entry
; CHECK-NEXT:   %8 = call i32 @nanos6_in_final(), !dbg !9
; CHECK-NEXT:   %9 = icmp ne i32 %8, 0, !dbg !9
; CHECK-NEXT:   br i1 %9, label %final.then, label %codeRepl, !dbg !9

; Function Attrs: noinline nounwind optnone uwtable
declare void @_ZN1SC2Ev(%struct.S* %this)

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #2

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #2

; Function Attrs: noinline nounwind optnone uwtable
declare void @_ZN1SC2ERKS_(%struct.S* %this, %struct.S* dereferenceable(4) %s)

; Function Attrs: noinline norecurse uwtable
declare void @oss_copy_ctor_ZN1SC1ERKS_(%struct.S* %0, %struct.S* %1, i64 %2)

; Function Attrs: noinline nounwind optnone uwtable
declare void @_ZN1SD2Ev(%struct.S* %this)

; Function Attrs: noinline norecurse uwtable
declare void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1)

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind
declare void @__clang_call_terminate(i8* %0)

; CHECK: define internal void @nanos6_unpacked_destroy_main0(%struct.S* %s) {
; CHECK: entry:
; CHECK-NEXT:   call void @oss_dtor_ZN1SD1Ev(%struct.S* %s, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_task_region_main0(%struct.S* %s, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !12 {
; CHECK: newFuncRoot:
; CHECK-NEXT:  br label %0, !dbg !13
; CHECK: .exitStub:                                        ; preds = %invoke.cont
; CHECK-NEXT:  ret void
; CHECK: 0:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   invoke void @_Z3fooRK1S(%struct.S* dereferenceable(4) %s)
; CHECK-NEXT:           to label %invoke.cont unwind label %terminate.lpad, !dbg !13
; CHECK: invoke.cont:                                      ; preds = %0
; CHECK-NEXT:   br label %.exitStub, !dbg !14
; CHECK: terminate.lpad:                                   ; preds = %0
; CHECK-NEXT:   %1 = landingpad { i8*, i32 }
; CHECK-NEXT:           catch i8* null
; CHECK-NEXT:   %2 = extractvalue { i8*, i32 } %1, 0
; CHECK-NEXT:   call void @__clang_call_terminate(i8* %2)
; CHECK-NEXT:   unreachable
; CHECK-NEXT: }

declare dso_local i8* @__cxa_begin_catch(i8*)

declare dso_local void @_ZSt9terminatev()

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline noreturn nounwind }
attributes #5 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_final_exception.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 9, type: !7, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 9, column: 23, scope: !6)
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !7, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DILocation(line: 13, column: 7, scope: !9)
!11 = !DILocation(line: 14, column: 13, scope: !9)
!12 = !DILocation(line: 15, column: 5, scope: !9)
!13 = !DILocation(line: 16, column: 1, scope: !9)
