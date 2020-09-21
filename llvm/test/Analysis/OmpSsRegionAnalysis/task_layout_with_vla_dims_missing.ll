; RUN: opt -ompss-2-regions -analyze -disable-checks -print-verbosity=dsa_vla_dims_missing < %s 2>&1 | FileCheck %s

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
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %0 = load i32, i32* %n, align 4
  %1 = zext i32 %0 to i64
  %2 = call i8* @llvm.stacksave()
  store i8* %2, i8** %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 16
  store i64 %1, i64* %__vla_expr0, align 8
  %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ]
  call void @llvm.directive.region.exit(token %3)
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.CAPTURED"(i64 %1) ]
  call void @llvm.directive.region.exit(token %4)
  %5 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %5)
  ret void
}

; CHECK: [0] TASK %3
; CHECK-NEXT:   %vla
; CHECK-NEXT: [0] TASK %4
; CHECK-NEXT:   %vla

declare i8* @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(i8*)
