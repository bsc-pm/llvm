; RUN: opt -passes='print<ompss-2-regions>'  -disable-checks -print-verbosity=non_pod_dsa_missing < %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.S = type <{ ptr, i32, [4 x i8] }>

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S, align 8
  store i32 0, ptr %retval, align 4
  call void @_ZN1SC1Ev(ptr %s)
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.COPY"(ptr %s, ptr @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(ptr %s, ptr @oss_dtor_ZN1SD1Ev) ]
  call void @llvm.directive.region.exit(token %0)
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.INIT"(ptr %s, ptr @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(ptr %s, ptr @oss_dtor_ZN1SD1Ev) ]
  call void @llvm.directive.region.exit(token %1)
  store i32 0, ptr %retval, align 4
  call void @_ZN1SD1Ev(ptr %s) #2
  %2 = load i32, ptr %retval, align 4
  ret i32 %2
}

; CHECK: [0] TASK %0
; CHECK-NEXT:   [Copy] %s
; CHECK-NEXT:   [Deinit] %s
; CHECK-NEXT: [0] TASK %1
; CHECK-NEXT:   [Init] %s
; CHECK-NEXT:   [Deinit] %s

declare void @_ZN1SC1Ev(ptr)
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)

declare void @_ZN1SC1ERS_i(ptr, ptr dereferenceable(16), i32)
declare void @oss_copy_ctor_ZN1SC1ERS_i(ptr %0, ptr %1, i64 %2)
declare void @_ZN1SD1Ev(ptr) unnamed_addr #4

declare void @oss_dtor_ZN1SD1Ev(ptr %0, i64 %1)
declare void @oss_ctor_ZN1SC1Ev(ptr %0, i64 %1)
