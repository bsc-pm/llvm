; RUN: opt -ompss-2-regions -analyze -disable-checks -enable-new-pm=0 < %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<ompss-2-regions>' -disable-checks < %s 2>&1 | FileCheck %s
define i32 @main() {
entry:
  %i2 = alloca i32, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
  %i = alloca i32, align 4
  store i32 0, i32* %i, align 4
  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64
0, i64 1, i64 1, i64 1, i64 1) ]
  %2 = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00") ]
  call void @llvm.directive.region.exit(token %1)
  %3 = call i1 @llvm.directive.marker() [ "DIR.OSS"([9 x i8] c"TASKWAIT\00") ]
  call void @llvm.directive.region.exit(token %0)
  store i32 0, i32* %i2, align 4
  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i2), "QUAL.OSS.LOOP.IND.VAR"(i32* %i2), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ]
  call void @llvm.directive.region.exit(token %4)
  ret i32 0
}

define internal i32 @compute_lb() {
entry:
  ret i32 0
}

define internal i32 @compute_ub() {
entry:
  ret i32 10
}

define internal i32 @compute_step() {
entry:
  ret i32 1
}

define internal i32 @compute_lb.1() {
entry:
  ret i32 0
}

define internal i32 @compute_ub.2() {
entry:
  ret i32 10
}

define internal i32 @compute_step.3() {
entry:
  ret i32 1
}

; CHECK: [0] TASK %0
; CHECK-NEXT:   [1] TASK.FOR %1
; CHECK-NEXT:     [2] RELEASE %2
; CHECK-NEXT:   [1] TASKWAIT %3
; CHECK-NEXT: [0] TASKLOOP %4

declare token @llvm.directive.region.entry()
declare void @llvm.directive.region.exit(token)
declare i1 @llvm.directive.marker()

