// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// LT → signed(<)    → 0
// LE → signed(<=)   → 1
// GT → signed(>)    → 2
// GE → signed(>=)   → 3
int sum = 0;
void signed_loop(int lb, int ub, int step) {
  #pragma oss task for
  for (int i = lb; i < ub; i += step)
      sum += i;
  #pragma oss task for
  for (int i = lb; i <= ub; i += step)
      sum += i;
  #pragma oss task for
  for (int i = ub; i > lb; i -= step)
      sum += i;
  #pragma oss task for
  for (int i = ub; i >= lb; i -= step)
      sum += i;
}

void unsigned_loop(unsigned lb, unsigned ub, unsigned step) {
  #pragma oss task for
  for (unsigned i = lb; i < ub; i += step)
      sum += i;
  #pragma oss task for
  for (unsigned i = lb; i <= ub; i += step)
      sum += i;
  #pragma oss task for
  for (unsigned i = ub; i > lb; i -= step)
      sum += i;
  #pragma oss task for
  for (unsigned i = ub; i >= lb; i -= step)
      sum += i;
}

// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb, i32* %lb.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub, i32* %ub.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.1, i32* %lb.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.2, i32* %ub.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.3, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1) ]
// CHECK: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.4, i32* %ub.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.5, i32* %lb.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.6, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 1, i64 1, i64 1, i64 1) ]
// CHECK: %13 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i5), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i5), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.7, i32* %ub.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.8, i32* %lb.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.9, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 1, i64 1, i64 1, i64 1) ]

// CHECK: define internal i32 @compute_lb(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.1(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.2(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.3(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.4(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.5(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.6(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   %sub = sub nsw i32 0, %0
// CHECK-NEXT:   ret i32 %sub
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.7(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.8(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.9(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   %sub = sub nsw i32 0, %0
// CHECK-NEXT:   ret i32 %sub
// CHECK-NEXT: }

// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.10, i32* %lb.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.11, i32* %ub.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.12, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 0, i64 0, i64 0, i64 0) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.13, i32* %lb.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.14, i32* %ub.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.15, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 0, i64 0, i64 0, i64 0) ]
// CHECK: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.16, i32* %ub.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.17, i32* %lb.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.18, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 0, i64 0, i64 0, i64 0) ]
// CHECK: %13 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i5), "QUAL.OSS.FIRSTPRIVATE"(i32* %ub.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %lb.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %step.addr), "QUAL.OSS.LOOP.IND.VAR"(i32* %i5), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 (i32*)* @compute_lb.19, i32* %ub.addr), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 (i32*)* @compute_ub.20, i32* %lb.addr), "QUAL.OSS.LOOP.STEP"(i32 (i32*)* @compute_step.21, i32* %step.addr), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 0, i64 0, i64 0, i64 0) ]

// CHECK: define internal i32 @compute_lb.10(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.11(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.12(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.13(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.14(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.15(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.16(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.17(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.18(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   %sub = sub i32 0, %0
// CHECK-NEXT:   ret i32 %sub
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.19(i32* %ub)
// CHECK: entry:
// CHECK-NEXT:   %ub.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %ub, i32** %ub.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %ub, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.20(i32* %lb)
// CHECK: entry:
// CHECK-NEXT:   %lb.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %lb, i32** %lb.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %lb, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.21(i32* %step)
// CHECK: entry:
// CHECK-NEXT:   %step.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %step, i32** %step.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %step, align 4
// CHECK-NEXT:   %sub = sub i32 0, %0
// CHECK-NEXT:   ret i32 %sub
// CHECK-NEXT: }

