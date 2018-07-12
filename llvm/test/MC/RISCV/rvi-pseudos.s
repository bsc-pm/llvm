# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ32,CHECK-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ64,CHECK-OBJ %s
.data
a_symbol: .word 42
another_symbol: .word 56
zero: .word 0
a0: .word 0

.text

# CHECK-ASM-label: foo
# CHECK-OBJ32-LABEL: foo
# CHECK-OBJ64-LABEL: foo
foo:
# This is here just to avoid the pseudos using the same address as foo
nop

## lla
# CHECK-ASM: lla a0, a_symbol

# CHECK-OBJ: .Lpcrel_hi0:
# CHECK-OBJ-NEXT: auipc a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_HI20 a_symbol
# CHECK-OBJ-NEXT: mv  a0, a0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi0
lla a0, a_symbol


## la (under default of nopic)
# CHECK-ASM: lla a0, a_symbol

# CHECK-OBJ: .Lpcrel_hi1:
# CHECK-OBJ-NEXT: auipc a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_HI20 a_symbol
# CHECK-OBJ-NEXT: mv  a0, a0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi1
la a0, a_symbol


## la (under pic)
# CHECK-ASM: la a0, a_symbol

# CHECK-OBJ32: .Lgot_hi0:
# CHECK-OBJ32-NEXT: auipc a0, 0
# CHECK-OBJ32-NEXT: R_RISCV_GOT_HI20 a_symbol
# CHECK-OBJ32-NEXT: lw  a0, 0(a0)
# CHECK-OBJ32-NEXT: R_RISCV_PCREL_LO12_I .Lgot_hi0

# CHECK-OBJ64: .Lgot_hi0:
# CHECK-OBJ64-NEXT: auipc a0, 0
# CHECK-OBJ64-NEXT: R_RISCV_GOT_HI20 a_symbol
# CHECK-OBJ64-NEXT: ld  a0, 0(a0)
# CHECK-OBJ64-NEXT: R_RISCV_PCREL_LO12_I .Lgot_hi0
.option pic
la a0, a_symbol


## la (under nopic)
# CHECK-ASM: lla a0, a_symbol

# CHECK-OBJ: .Lpcrel_hi2:
# CHECK-OBJ: auipc a0, 0
# CHECK-OBJ: R_RISCV_PCREL_HI20 a_symbol
# CHECK-OBJ: mv  a0, a0
# CHECK-OBJ: R_RISCV_PCREL_LO12_I .Lpcrel_hi2
.option nopic
la a0, a_symbol

## Check that lla and la don't choke with a symbol named like a register
# CHECK-ASM: la a0, zero
lla a0, zero
# CHECK-ASM: lla a0, zero
lla a0, zero
# CHECK-ASM: la a0, a0
lla a0, a0
# CHECK-ASM: lla a0, a0
lla a0, a0
