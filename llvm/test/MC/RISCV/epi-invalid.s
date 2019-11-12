# RUN: not llvm-mc < %s -arch=riscv64 -mattr=+v 2>&1 | FileCheck %s

# CHECK: :17: error: expected '.t' suffix
vlb.v v0, (a0), x0

# CHECK: :16: error: unknown operand
vlb.v v0, (a0),

# CHECK: :21: error: expected '.t' suffix
vadd.vv v1, v2, v3, v0

vadd.vv v4, v5, v6, v0.t
