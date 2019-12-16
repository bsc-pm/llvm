// Check that when no -march is used, riscv64-linux defines
// the relevant RV64-MFDAC (GC) features.
// 
// RUN: %clang -target riscv64-linux -x c -E -dM %s \
// RUN: -o - | FileCheck %s

// CHECK: __riscv_atomic 1
// CHECK: __riscv_compressed 1
// CHECK: __riscv_div 1
// CHECK: __riscv_fdiv 1
// CHECK: __riscv_flen 64
// CHECK: __riscv_fsqrt 1
// CHECK: __riscv_mul 1
// CHECK: __riscv_muldiv 1
