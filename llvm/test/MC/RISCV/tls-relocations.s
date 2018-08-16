# RUN: llvm-mc -filetype obj -triple riscv64 < %s \
# RUN:    | llvm-objdump -d -r - | FileCheck %s
# RUN: llvm-mc -filetype obj -triple riscv64 < %s \
# RUN:    | llvm-readobj -r -t - | FileCheck --check-prefix=CHECK-ELF %s

# CHECK: 	lui	a0, 0
# CHECK:  R_RISCV_TPREL_HI20	y
# CHECK: 	add	a0, tp, a0
# CHECK:  R_RISCV_TPREL_ADD	y
# CHECK: 	lw	a1, 0(a0)
# CHECK:  R_RISCV_TPREL_LO12_I	y
# CHECK: 	sw	a1, 0(a0)
# CHECK:  R_RISCV_TPREL_LO12_S	y

# CHECK-ELF: Symbol {
# CHECK-ELF: Name: y (1)
# CHECK-ELF-NEXT: Value: 0x0
# CHECK-ELF-NEXT: Size: 0
# CHECK-ELF-NEXT: Binding: Global (0x1)
# CHECK-ELF-NEXT: Type: TLS (0x6)

lui	a0, %tprel_hi(y)
add	a0, tp, a0, %tprel_add(y)
lw	a1, %tprel_lo(y)(a0)
sw	a1, %tprel_lo(y)(a0)
