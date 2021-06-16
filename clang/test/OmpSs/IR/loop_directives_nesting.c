// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int sum;
void task_taskfor() {
  #pragma oss task
  {
    int sum_local = 0;
    #pragma oss task for
    for (int i = 0; i < 10; ++i) {
      sum_local += i;
      sum += i;
    }
  }
}

void taskfor_task() {
  #pragma oss task for
  for (int i = 0; i < 10; ++i) {
    int sum_local = 0;
    #pragma oss task
    {
      sum_local += i;
      sum += i;
    }
  }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @sum) ]
// CHECK-NEXT: %sum_local = alloca i32, align 4
// CHECK-NEXT: %i = alloca i32, align 4
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32* %sum_local), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ]

// CHECK: %i = alloca i32, align 4
// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 ()* @compute_lb.1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 ()* @compute_ub.2), "QUAL.OSS.LOOP.STEP"(i32 ()* @compute_step.3), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ]
// CHECK-NEXT: %sum_local = alloca i32, align 4
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @sum), "QUAL.OSS.FIRSTPRIVATE"(i32* %sum_local), "QUAL.OSS.FIRSTPRIVATE"(i32* %i) ]

