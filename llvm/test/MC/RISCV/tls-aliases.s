# RUN: llvm-mc -triple riscv32 -filetype obj < %s | \
# RUN:   llvm-objdump -d -r - | \
# RUN:   FileCheck --check-prefix=RV32 --check-prefix=CHECK %s
# RUN: llvm-mc -triple riscv64 -filetype obj < %s | \
# RUN:   llvm-objdump -d -r - | \
# RUN:   FileCheck --check-prefix=RV64 --check-prefix=CHECK %s

# CHECK: .Ltls_got_hi0:
# CHECK:   auipc a0, 0
# CHECK: R_RISCV_TLS_GOT_HI20 y
# RV32: lw  a0, 0(a0)
# RV64: ld  a0, 0(a0)
# CHECK:  R_RISCV_PCREL_LO12_I .Ltls_got_hi0
la.tls.ie	a0, y

# CHECK: .Ltls_gd_hi0:
# CHECK: auipc	a0, 0
# CHECK: R_RISCV_TLS_GD_HI20	y
# CHECK: mv	a0, a0
# CHECK: R_RISCV_PCREL_LO12_I	.Ltls_gd_hi0
la.tls.gd a0, y
