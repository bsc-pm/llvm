# RUN: llvm-mc -filetype obj -triple riscv64 < %s \
# RUN:    | llvm-objdump -d -r - | FileCheck %s
# RUN: llvm-mc -filetype obj -triple riscv64 < %s \
# RUN:    | llvm-readobj -r -t - | FileCheck --check-prefix=CHECK-ELF %s

# CHECK: lui	a0, 0
# CHECK: R_RISCV_TPREL_HI20	a
# CHECK: add	a0, a0, tp
# CHECK: R_RISCV_TPREL_ADD	a
# CHECK: lw	a1, 0(a0)
# CHECK: R_RISCV_TPREL_LO12_I	a
# CHECK: sw	a1, 0(a0)
# CHECK: R_RISCV_TPREL_LO12_S	a

# CHECK-ELF: Symbols [

# CHECK-ELF-LABEL: Name: a (
# CHECK-ELF: Binding: Global (0x1)
# CHECK-ELF: Type: TLS (0x6)

lui	a0, %tprel_hi(a)
add	a0, a0, tp, %tprel_add(a)
lw	a1, %tprel_lo(a)(a0)
sw	a1, %tprel_lo(a)(a0)

# CHECK: auipc a0, 0
# CHECK: R_RISCV_TLS_GD_HI20  b
# CHECK: mv  a0, a0
# CHECK: R_RISCV_PCREL_LO12_I .Ltls_gd_hi0

# CHECK-ELF-LABEL: Name: b (
# CHECK-ELF: Binding: Global (0x1)
# CHECK-ELF: Type: TLS (0x6)

la.tls.gd a0, b

# CHECK: auipc a5, 0
# CHECK: R_RISCV_TLS_GOT_HI20 c
# CHECK: ld  a5, 0(a5)
# CHECK: R_RISCV_PCREL_LO12_I .Ltls_got_hi0

# CHECK-ELF-LABEL: Name: c (
# CHECK-ELF: Binding: Global (0x1)
# CHECK-ELF: Type: TLS (0x6)

la.tls.ie a5, c

# CHECK-ELF: ]
