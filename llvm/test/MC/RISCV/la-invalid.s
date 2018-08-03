# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

la a0, %pcrel_hi(y) # CHECK: error: operand must be a bare symbol name
la.tls.ie	a0, %pcrel_hi(y) # CHECK: error: operand must be a bare symbol name
la.tls.gd a0, %pcrel_hi(y) # CHECK: error: operand must be a bare symbol name
