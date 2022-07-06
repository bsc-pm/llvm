; RUN: opt -passes='print<ompss-2-regions>' -disable-checks -print-verbosity=dsa_vla_dims_missing < %s 2>&1 | FileCheck %s

; void foo() {
;     int n;
;     int vla[n];
;     #pragma oss task shared(vla)
;     {}
;     #pragma oss task shared(vla)
;     {}
; }

define dso_local void @foo() {
entry:
  %n = alloca i32, align 4
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  %0 = load i32, ptr %n, align 4
  %1 = zext i32 %0 to i64
  %2 = call ptr @llvm.stacksave()
  store ptr %2, ptr %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 16
  store i64 %1, i64* %__vla_expr0, align 8
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.VLA.DIMS"(ptr %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ]
  call void @llvm.directive.region.exit(token %3)
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr %vla, i32 undef), "QUAL.OSS.CAPTURED"(i64 %1) ]
  call void @llvm.directive.region.exit(token %4)
  %5 = load ptr, ptr %saved_stack, align 8
  call void @llvm.stackrestore(ptr %5)
  ret void
}

; CHECK: [0] TASK %3
; CHECK-NEXT:   %vla
; CHECK-NEXT: [0] TASK %4
; CHECK-NEXT:   %vla

declare ptr @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(ptr)
