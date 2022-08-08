; RUN: opt -passes='print<ompss-2-regions>' -disable-checks -print-verbosity=vla_dims_capture_missing < %s 2>&1 | FileCheck %s

define dso_local void @vla_section_dep(i32 %n) {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  %__vla_expr1 = alloca i64, align 8
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %add = add nsw i32 %0, 1
  %1 = zext i32 %add to i64
  %2 = load i32, ptr %n.addr, align 4
  %add1 = add nsw i32 %2, 2
  %3 = zext i32 %add1 to i64
  %4 = call ptr @llvm.stacksave()
  store ptr %4, ptr %saved_stack, align 8
  %5 = mul nuw i64 %1, %3
  %vla = alloca i32, i64 %5, align 16
  store i64 %1, ptr %__vla_expr0, align 8
  store i64 %3, ptr %__vla_expr1, align 8
  %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr %vla, i32 undef), "QUAL.OSS.VLA.DIMS"(ptr %vla, i64 %1, i64 %3) ]
  call void @llvm.directive.region.exit(token %6)
  %7 = load ptr, ptr %saved_stack, align 8
  call void @llvm.stackrestore(ptr %7)
  ret void
}

; CHECK: [0] TASK %6
; CHECK-NEXT:   %1
; CHECK-NEXT:   %3

declare ptr @llvm.stacksave()
declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare void @llvm.stackrestore(ptr)
