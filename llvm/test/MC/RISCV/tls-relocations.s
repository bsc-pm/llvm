# RUN: llvm-mc -filetype obj -triple riscv64 < %s \
# RUN:    | llvm-objdump -d -r - | FileCheck %s

# CHECK: 	lui	a0, 0
# CHECK:  R_RISCV_TPREL_HI20	y
# CHECK: 	add	a0, tp, a0
# CHECK:  R_RISCV_TPREL_ADD	y
# CHECK: 	lw	a1, 0(a0)
# CHECK:  R_RISCV_TPREL_LO12_I	y
# CHECK: 	sw	a1, 0(a0)
# CHECK:  R_RISCV_TPREL_LO12_S	y

lui	a0, %tprel_hi(y)
add	a0, tp, a0, %tprel_add(y)
lw	a1, %tprel_lo(y)(a0)
sw	a1, %tprel_lo(y)(a0)
