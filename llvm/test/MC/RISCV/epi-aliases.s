# RUN: llvm-mc --triple=riscv64 -mattr +v < %s --show-encoding | FileCheck %s

# CHECK:	vsetvli	s0, t3, e32, m1         # encoding: [0x57,0x74,0x8e,0x00]
vsetvli x8, x28, e32
# CHECK:	vsetvli	s0, t3, e32, m1         # encoding: [0x57,0x74,0x8e,0x00]
vsetvli x8, x28, e32, m1
