// RUN: %clang_cc1 -emit-llvm -o - -fopenmp -fopenmp-nosv %s | FileCheck %s
// expected-no-diagnostics

// Task types are registered before user global constructors, which may use
// OpenMP

// CHECK: @llvm.global_ctors = {{.*}}{ i32 100, ptr @__nosv_ctor_register_task_info, ptr null }{{.*}}{ i32 65535, ptr @_GLOBAL__sub_I_

int foo() {
  int s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < 10; i++)
    s += i;
  return s;
}

static int x = foo();
