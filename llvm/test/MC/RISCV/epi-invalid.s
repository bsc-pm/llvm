# Reenable once we can have this working again
# RUN: not llvm-mc < %s -arch=riscv64 -mattr=+epi 2>&1 | FileCheck %s

# CHECK: :17: error: vector mask operand must be of the form 'v0.t'
vlb.v v0, (a0), x0

# CHECK: :16: error: unknown operand
vlb.v v0, (a0),
