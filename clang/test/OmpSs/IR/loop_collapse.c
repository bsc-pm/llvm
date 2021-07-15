// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
int main() {
  #pragma oss taskloop collapse(2)
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j)
    {
    }
  }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32* %j), "QUAL.OSS.LOOP.IND.VAR"(i32* %i, i32* %j), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb, i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub, i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step, i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1, i64 0, i64 1, i64 1, i64 1, i64 1) ]

// CHECK: define internal i32 @compute_lb()
// CHECK: entry:
// CHECK-NEXT:   ret i32 0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub()
// CHECK: entry:
// CHECK-NEXT:   ret i32 10
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step()
// CHECK: entry:
// CHECK-NEXT:   ret i32 1
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_lb.1()
// CHECK: entry:
// CHECK-NEXT:   ret i32 0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_ub.2()
// CHECK: entry:
// CHECK-NEXT:   ret i32 10
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_step.3()
// CHECK: entry:
// CHECK-NEXT:   ret i32 1
// CHECK-NEXT: }

