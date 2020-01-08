// RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+v < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=riscv64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+v < %s \
// RUN:        | llvm-objdump -d -mattr=+v - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+v < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
vmv.v.v v1, v2
// CHECK-INST: vmv.v.v v1, v2
// CHECK-ENCODING: [0xd7,0x00,0x01,0x5e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: d7 00 01 5e <unknown>

vmv.v.x v1, a0
// CHECK-INST: vmv.v.x v1, a0
// CHECK-ENCODING: [0xd7,0x40,0x05,0x5e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: d7 40 05 5e <unknown>

vmv.v.i v1, 15
// CHECK-INST: vmv.v.i v1, 15
// CHECK-ENCODING: [0xd7,0xb0,0x07,0x5e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: d7 b0 07 5e <unknown>

vmv.x.s a2, v3
// CHECK-INST: vmv.x.s a2, v3
// CHECK-ENCODING: [0x57,0x26,0x30,0x42]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: 57 26 30 42 <unknown>

vmv.s.x v1, a0
// CHECK-INST: vmv.s.x v1, a0
// CHECK-ENCODING: [0xd7,0x60,0x05,0x42]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: d7 60 05 42 <unknown>

vfmv.v.f v1, fa0
// CHECK-INST: vfmv.v.f v1, fa0
// CHECK-ENCODING: [0xd7,0x50,0x05,0x5e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: d7 50 05 5e <unknown>

vfmv.f.s fa0, v3
// CHECK-INST: vfmv.f.s fa0, v3
// CHECK-ENCODING: [0x57,0x15,0x30,0x42]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: 57 15 30 42 <unknown>

vfmv.s.f v1, fa0
// CHECK-INST: vfmv.s.f v1, fa0
// CHECK-ENCODING: [0xd7,0x50,0x05,0x42]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: d7 50 05 42 <unknown>

vmv1r.v v0, v1
// CHECK-INST: vmv1r.v v0, v1
// CHECK-ENCODING: [0x57,0x30,0x10,0x9e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: 57 30 10 9e <unknown>

vmv2r.v v0, v2
// CHECK-INST: vmv2r.v v0, v2
// CHECK-ENCODING: [0x57,0xb0,0x20,0x9e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: 57 b0 20 9e <unknown>

vmv4r.v v0, v4
// CHECK-INST: vmv4r.v v0, v4
// CHECK-ENCODING: [0x57,0xb0,0x41,0x9e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: 57 b0 41 9e <unknown>

vmv8r.v v0, v8
// CHECK-INST: vmv8r.v v0, v8
// CHECK-ENCODING: [0x57,0xb0,0x83,0x9e]
// CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
// CHECK-UNKNOWN: 57 b0 83 9e <unknown>
