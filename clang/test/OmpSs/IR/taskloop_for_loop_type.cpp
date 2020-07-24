// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// LT → signed(<)    → 0
// LE → signed(<=)   → 1
// GT → signed(>)    → 2
// GE → signed(>=)   → 3
int sum = 0;
void signed_loop(int lb, int ub, int step) {
  #pragma oss taskloop for
  for (int i = lb; i < ub; i += step)
      sum += i;
  #pragma oss taskloop for
  for (int i = lb; i <= ub; i += step)
      sum += i;
  #pragma oss taskloop for
  for (int i = ub; i > lb; i -= step)
      sum += i;
  #pragma oss taskloop for
  for (int i = ub; i >= lb; i -= step)
      sum += i;
}

void unsigned_loop(unsigned lb, unsigned ub, unsigned step) {
  #pragma oss taskloop for
  for (unsigned i = lb; i < ub; i += step)
      sum += i;
  #pragma oss taskloop for
  for (unsigned i = lb; i <= ub; i += step)
      sum += i;
  #pragma oss taskloop for
  for (unsigned i = ub; i > lb; i -= step)
      sum += i;
  #pragma oss taskloop for
  for (unsigned i = ub; i >= lb; i -= step)
      sum += i;
}


// CHECK: %1 = load i32, i32* %lb.addr, align 4
// CHECK-NEXT: %2 = load i32, i32* %ub.addr, align 4
// CHECK-NEXT: %3 = load i32, i32* %step.addr, align 4
// CHECK-NEXT: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ]

// CHECK: %8 = load i32, i32* %lb.addr, align 4
// CHECK-NEXT: %9 = load i32, i32* %ub.addr, align 4
// CHECK-NEXT: %10 = load i32, i32* %step.addr, align 4
// CHECK-NEXT: %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %8), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %9), "QUAL.OSS.LOOP.STEP"(i32 %10), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %8, i32 %9, i32 %10) ]

// CHECK: %15 = load i32, i32* %ub.addr, align 4, !dbg !13
// CHECK-NEXT: %16 = load i32, i32* %lb.addr, align 4, !dbg !13
// CHECK-NEXT: %17 = load i32, i32* %step.addr, align 4, !dbg !13
// CHECK-NEXT: %sub = sub nsw i32 0, %17, !dbg !13
// CHECK-NEXT: %18 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %15), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %16), "QUAL.OSS.LOOP.STEP"(i32 %sub), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %15, i32 %16, i32 %sub) ]

// CHECK: %22 = load i32, i32* %ub.addr, align 4, !dbg !15
// CHECK-NEXT: %23 = load i32, i32* %lb.addr, align 4, !dbg !15
// CHECK-NEXT: %24 = load i32, i32* %step.addr, align 4, !dbg !15
// CHECK-NEXT: %sub6 = sub nsw i32 0, %24, !dbg !15
// CHECK-NEXT: %25 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i5), "QUAL.OSS.LOOP.IND.VAR"(i32* %i5), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %22), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %23), "QUAL.OSS.LOOP.STEP"(i32 %sub6), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.CAPTURED"(i32 %22, i32 %23, i32 %sub6) ]

// CHECK: %1 = load i32, i32* %lb.addr, align 4, !dbg !19
// CHECK-NEXT: %2 = load i32, i32* %ub.addr, align 4, !dbg !19
// CHECK-NEXT: %3 = load i32, i32* %step.addr, align 4, !dbg !19
// CHECK-NEXT: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %2), "QUAL.OSS.LOOP.STEP"(i32 %3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %1, i32 %2, i32 %3) ], !dbg !19

// CHECK: %8 = load i32, i32* %lb.addr, align 4, !dbg !21
// CHECK-NEXT: %9 = load i32, i32* %ub.addr, align 4, !dbg !21
// CHECK-NEXT: %10 = load i32, i32* %step.addr, align 4, !dbg !21
// CHECK-NEXT: %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %8), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %9), "QUAL.OSS.LOOP.STEP"(i32 %10), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %8, i32 %9, i32 %10) ], !dbg !21

// CHECK: %15 = load i32, i32* %ub.addr, align 4, !dbg !23
// CHECK-NEXT: %16 = load i32, i32* %lb.addr, align 4, !dbg !23
// CHECK-NEXT: %17 = load i32, i32* %step.addr, align 4, !dbg !23
// CHECK-NEXT: %sub = sub i32 0, %17, !dbg !23
// CHECK-NEXT: %18 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %15), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %16), "QUAL.OSS.LOOP.STEP"(i32 %sub), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %15, i32 %16, i32 %sub) ], !dbg !23

// CHECK: %22 = load i32, i32* %ub.addr, align 4, !dbg !25
// CHECK-NEXT: %23 = load i32, i32* %lb.addr, align 4, !dbg !25
// CHECK-NEXT: %24 = load i32, i32* %step.addr, align 4, !dbg !25
// CHECK-NEXT: %sub6 = sub i32 0, %24, !dbg !25
// CHECK-NEXT: %25 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i5), "QUAL.OSS.LOOP.IND.VAR"(i32* %i5), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 %22), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 %23), "QUAL.OSS.LOOP.STEP"(i32 %sub6), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 0, i64 0, i64 0, i64 0), "QUAL.OSS.CAPTURED"(i32 %22, i32 %23, i32 %sub6) ], !dbg !25

