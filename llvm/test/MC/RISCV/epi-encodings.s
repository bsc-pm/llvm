# Generated witu utils/EPI/process.py
# RUN: llvm-mc < %s -arch=riscv64 -mattr=+m,+f,+d,+a,+v -show-encoding | FileCheck %s

# Encoding: |000000|1|00000|00001|000|00010|1010111|
# CHECK: vadd.vv v2, v0, v1
# CHECK-SAME: [0x57,0x81,0x00,0x02]
vadd.vv v2, v0, v1
# Encoding: |000000|0|00011|00100|000|00101|1010111|
# CHECK: vadd.vv v5, v3, v4, v0.t
# CHECK-SAME: [0xd7,0x02,0x32,0x00]
vadd.vv v5, v3, v4, v0.t

# Encoding: |000000|1|00110|00001|100|00111|1010111|
# CHECK: vadd.vx v7, v6, ra
# CHECK-SAME: [0xd7,0xc3,0x60,0x02]
vadd.vx v7, v6, ra
# Encoding: |000000|0|01000|00011|100|01001|1010111|
# CHECK: vadd.vx v9, v8, gp, v0.t
# CHECK-SAME: [0xd7,0xc4,0x81,0x00]
vadd.vx v9, v8, gp, v0.t

# Encoding: |000000|1|01010|00000|011|01011|1010111|
# CHECK: vadd.vi v11, v10, 0
# CHECK-SAME: [0xd7,0x35,0xa0,0x02]
vadd.vi v11, v10, 0
# Encoding: |000000|0|01100|00001|011|01101|1010111|
# CHECK: vadd.vi v13, v12, 1, v0.t
# CHECK-SAME: [0xd7,0xb6,0xc0,0x00]
vadd.vi v13, v12, 1, v0.t

# Encoding: |000010|1|01110|01111|000|10000|1010111|
# CHECK: vsub.vv v16, v14, v15
# CHECK-SAME: [0x57,0x88,0xe7,0x0a]
vsub.vv v16, v14, v15
# Encoding: |000010|0|10001|10010|000|10011|1010111|
# CHECK: vsub.vv v19, v17, v18, v0.t
# CHECK-SAME: [0xd7,0x09,0x19,0x09]
vsub.vv v19, v17, v18, v0.t

# Encoding: |000010|1|10100|00101|100|10101|1010111|
# CHECK: vsub.vx v21, v20, t0
# CHECK-SAME: [0xd7,0xca,0x42,0x0b]
vsub.vx v21, v20, t0
# Encoding: |000010|0|10110|00111|100|10111|1010111|
# CHECK: vsub.vx v23, v22, t2, v0.t
# CHECK-SAME: [0xd7,0xcb,0x63,0x09]
vsub.vx v23, v22, t2, v0.t

# Encoding: |000011|1|11000|01001|100|11001|1010111|
# CHECK: vrsub.vx v25, v24, s1
# CHECK-SAME: [0xd7,0xcc,0x84,0x0f]
vrsub.vx v25, v24, s1
# Encoding: |000011|0|11010|01011|100|11011|1010111|
# CHECK: vrsub.vx v27, v26, a1, v0.t
# CHECK-SAME: [0xd7,0xcd,0xa5,0x0d]
vrsub.vx v27, v26, a1, v0.t

# Encoding: |000011|1|11100|00010|011|11101|1010111|
# CHECK: vrsub.vi v29, v28, 2
# CHECK-SAME: [0xd7,0x3e,0xc1,0x0f]
vrsub.vi v29, v28, 2
# Encoding: |000011|0|11110|00011|011|11111|1010111|
# CHECK: vrsub.vi v31, v30, 3, v0.t
# CHECK-SAME: [0xd7,0xbf,0xe1,0x0d]
vrsub.vi v31, v30, 3, v0.t

# Encoding: |000100|1|00000|00001|000|00010|1010111|
# CHECK: vminu.vv v2, v0, v1
# CHECK-SAME: [0x57,0x81,0x00,0x12]
vminu.vv v2, v0, v1
# Encoding: |000100|0|00011|00100|000|00101|1010111|
# CHECK: vminu.vv v5, v3, v4, v0.t
# CHECK-SAME: [0xd7,0x02,0x32,0x10]
vminu.vv v5, v3, v4, v0.t

# Encoding: |000100|1|00110|01101|100|00111|1010111|
# CHECK: vminu.vx v7, v6, a3
# CHECK-SAME: [0xd7,0xc3,0x66,0x12]
vminu.vx v7, v6, a3
# Encoding: |000100|0|01000|01111|100|01001|1010111|
# CHECK: vminu.vx v9, v8, a5, v0.t
# CHECK-SAME: [0xd7,0xc4,0x87,0x10]
vminu.vx v9, v8, a5, v0.t

# Encoding: |000101|1|01010|01011|000|01100|1010111|
# CHECK: vmin.vv v12, v10, v11
# CHECK-SAME: [0x57,0x86,0xa5,0x16]
vmin.vv v12, v10, v11
# Encoding: |000101|0|01101|01110|000|01111|1010111|
# CHECK: vmin.vv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x07,0xd7,0x14]
vmin.vv v15, v13, v14, v0.t

# Encoding: |000101|1|10000|10001|100|10001|1010111|
# CHECK: vmin.vx v17, v16, a7
# CHECK-SAME: [0xd7,0xc8,0x08,0x17]
vmin.vx v17, v16, a7
# Encoding: |000101|0|10010|10011|100|10011|1010111|
# CHECK: vmin.vx v19, v18, s3, v0.t
# CHECK-SAME: [0xd7,0xc9,0x29,0x15]
vmin.vx v19, v18, s3, v0.t

# Encoding: |000110|1|10100|10101|000|10110|1010111|
# CHECK: vmaxu.vv v22, v20, v21
# CHECK-SAME: [0x57,0x8b,0x4a,0x1b]
vmaxu.vv v22, v20, v21
# Encoding: |000110|0|10111|11000|000|11001|1010111|
# CHECK: vmaxu.vv v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x0c,0x7c,0x19]
vmaxu.vv v25, v23, v24, v0.t

# Encoding: |000110|1|11010|10101|100|11011|1010111|
# CHECK: vmaxu.vx v27, v26, s5
# CHECK-SAME: [0xd7,0xcd,0xaa,0x1b]
vmaxu.vx v27, v26, s5
# Encoding: |000110|0|11100|10111|100|11101|1010111|
# CHECK: vmaxu.vx v29, v28, s7, v0.t
# CHECK-SAME: [0xd7,0xce,0xcb,0x19]
vmaxu.vx v29, v28, s7, v0.t

# Encoding: |000111|1|11110|11111|000|00000|1010111|
# CHECK: vmax.vv v0, v30, v31
# CHECK-SAME: [0x57,0x80,0xef,0x1f]
vmax.vv v0, v30, v31
# Encoding: |000111|0|00001|00010|000|00011|1010111|
# CHECK: vmax.vv v3, v1, v2, v0.t
# CHECK-SAME: [0xd7,0x01,0x11,0x1c]
vmax.vv v3, v1, v2, v0.t

# Encoding: |000111|1|00100|11001|100|00101|1010111|
# CHECK: vmax.vx v5, v4, s9
# CHECK-SAME: [0xd7,0xc2,0x4c,0x1e]
vmax.vx v5, v4, s9
# Encoding: |000111|0|00110|11011|100|00111|1010111|
# CHECK: vmax.vx v7, v6, s11, v0.t
# CHECK-SAME: [0xd7,0xc3,0x6d,0x1c]
vmax.vx v7, v6, s11, v0.t

# Encoding: |001001|1|01000|01001|000|01010|1010111|
# CHECK: vand.vv v10, v8, v9
# CHECK-SAME: [0x57,0x85,0x84,0x26]
vand.vv v10, v8, v9
# Encoding: |001001|0|01011|01100|000|01101|1010111|
# CHECK: vand.vv v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x06,0xb6,0x24]
vand.vv v13, v11, v12, v0.t

# Encoding: |001001|1|01110|11101|100|01111|1010111|
# CHECK: vand.vx v15, v14, t4
# CHECK-SAME: [0xd7,0xc7,0xee,0x26]
vand.vx v15, v14, t4
# Encoding: |001001|0|10000|11111|100|10001|1010111|
# CHECK: vand.vx v17, v16, t6, v0.t
# CHECK-SAME: [0xd7,0xc8,0x0f,0x25]
vand.vx v17, v16, t6, v0.t

# Encoding: |001001|1|10010|00100|011|10011|1010111|
# CHECK: vand.vi v19, v18, 4
# CHECK-SAME: [0xd7,0x39,0x22,0x27]
vand.vi v19, v18, 4
# Encoding: |001001|0|10100|00101|011|10101|1010111|
# CHECK: vand.vi v21, v20, 5, v0.t
# CHECK-SAME: [0xd7,0xba,0x42,0x25]
vand.vi v21, v20, 5, v0.t

# Encoding: |001010|1|10110|10111|000|11000|1010111|
# CHECK: vor.vv v24, v22, v23
# CHECK-SAME: [0x57,0x8c,0x6b,0x2b]
vor.vv v24, v22, v23
# Encoding: |001010|0|11001|11010|000|11011|1010111|
# CHECK: vor.vv v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x0d,0x9d,0x29]
vor.vv v27, v25, v26, v0.t

# Encoding: |001010|1|11100|00010|100|11101|1010111|
# CHECK: vor.vx v29, v28, sp
# CHECK-SAME: [0xd7,0x4e,0xc1,0x2b]
vor.vx v29, v28, sp
# Encoding: |001010|0|11110|00100|100|11111|1010111|
# CHECK: vor.vx v31, v30, tp, v0.t
# CHECK-SAME: [0xd7,0x4f,0xe2,0x29]
vor.vx v31, v30, tp, v0.t

# Encoding: |001010|1|00000|00110|011|00001|1010111|
# CHECK: vor.vi v1, v0, 6
# CHECK-SAME: [0xd7,0x30,0x03,0x2a]
vor.vi v1, v0, 6
# Encoding: |001010|0|00010|00111|011|00011|1010111|
# CHECK: vor.vi v3, v2, 7, v0.t
# CHECK-SAME: [0xd7,0xb1,0x23,0x28]
vor.vi v3, v2, 7, v0.t

# Encoding: |001011|1|00100|00101|000|00110|1010111|
# CHECK: vxor.vv v6, v4, v5
# CHECK-SAME: [0x57,0x83,0x42,0x2e]
vxor.vv v6, v4, v5
# Encoding: |001011|0|00111|01000|000|01001|1010111|
# CHECK: vxor.vv v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x04,0x74,0x2c]
vxor.vv v9, v7, v8, v0.t

# Encoding: |001011|1|01010|00110|100|01011|1010111|
# CHECK: vxor.vx v11, v10, t1
# CHECK-SAME: [0xd7,0x45,0xa3,0x2e]
vxor.vx v11, v10, t1
# Encoding: |001011|0|01100|01000|100|01101|1010111|
# CHECK: vxor.vx v13, v12, s0, v0.t
# CHECK-SAME: [0xd7,0x46,0xc4,0x2c]
vxor.vx v13, v12, s0, v0.t

# Encoding: |001011|1|01110|01000|011|01111|1010111|
# CHECK: vxor.vi v15, v14, 8
# CHECK-SAME: [0xd7,0x37,0xe4,0x2e]
vxor.vi v15, v14, 8
# Encoding: |001011|0|10000|01001|011|10001|1010111|
# CHECK: vxor.vi v17, v16, 9, v0.t
# CHECK-SAME: [0xd7,0xb8,0x04,0x2d]
vxor.vi v17, v16, 9, v0.t

# Encoding: |001100|1|10010|10011|000|10100|1010111|
# CHECK: vrgather.vv v20, v18, v19
# CHECK-SAME: [0x57,0x8a,0x29,0x33]
vrgather.vv v20, v18, v19
# Encoding: |001100|0|10101|10110|000|10111|1010111|
# CHECK: vrgather.vv v23, v21, v22, v0.t
# CHECK-SAME: [0xd7,0x0b,0x5b,0x31]
vrgather.vv v23, v21, v22, v0.t

# Encoding: |001100|1|11000|01010|100|11001|1010111|
# CHECK: vrgather.vx v25, v24, a0
# CHECK-SAME: [0xd7,0x4c,0x85,0x33]
vrgather.vx v25, v24, a0
# Encoding: |001100|0|11010|01100|100|11011|1010111|
# CHECK: vrgather.vx v27, v26, a2, v0.t
# CHECK-SAME: [0xd7,0x4d,0xa6,0x31]
vrgather.vx v27, v26, a2, v0.t

# Encoding: |001100|1|11100|01010|011|11101|1010111|
# CHECK: vrgather.vi v29, v28, 10
# CHECK-SAME: [0xd7,0x3e,0xc5,0x33]
vrgather.vi v29, v28, 10
# Encoding: |001100|0|11110|01011|011|11111|1010111|
# CHECK: vrgather.vi v31, v30, 11, v0.t
# CHECK-SAME: [0xd7,0xbf,0xe5,0x31]
vrgather.vi v31, v30, 11, v0.t

# Encoding: |001110|1|00000|01110|100|00001|1010111|
# CHECK: vslideup.vx v1, v0, a4
# CHECK-SAME: [0xd7,0x40,0x07,0x3a]
vslideup.vx v1, v0, a4
# Encoding: |001110|0|00010|10000|100|00011|1010111|
# CHECK: vslideup.vx v3, v2, a6, v0.t
# CHECK-SAME: [0xd7,0x41,0x28,0x38]
vslideup.vx v3, v2, a6, v0.t

# Encoding: |001110|1|00100|01100|011|00101|1010111|
# CHECK: vslideup.vi v5, v4, 12
# CHECK-SAME: [0xd7,0x32,0x46,0x3a]
vslideup.vi v5, v4, 12
# Encoding: |001110|0|00110|01101|011|00111|1010111|
# CHECK: vslideup.vi v7, v6, 13, v0.t
# CHECK-SAME: [0xd7,0xb3,0x66,0x38]
vslideup.vi v7, v6, 13, v0.t

# Encoding: |001111|1|01000|10010|100|01001|1010111|
# CHECK: vslidedown.vx v9, v8, s2
# CHECK-SAME: [0xd7,0x44,0x89,0x3e]
vslidedown.vx v9, v8, s2
# Encoding: |001111|0|01010|10100|100|01011|1010111|
# CHECK: vslidedown.vx v11, v10, s4, v0.t
# CHECK-SAME: [0xd7,0x45,0xaa,0x3c]
vslidedown.vx v11, v10, s4, v0.t

# Encoding: |001111|1|01100|01110|011|01101|1010111|
# CHECK: vslidedown.vi v13, v12, 14
# CHECK-SAME: [0xd7,0x36,0xc7,0x3e]
vslidedown.vi v13, v12, 14
# Encoding: |001111|0|01110|01111|011|01111|1010111|
# CHECK: vslidedown.vi v15, v14, 15, v0.t
# CHECK-SAME: [0xd7,0xb7,0xe7,0x3c]
vslidedown.vi v15, v14, 15, v0.t

# Encoding: |010000|0|10000|10001|000|10010|1010111|
# CHECK: vadc.vvm v18, v16, v17, v0
# CHECK-SAME: [0x57,0x89,0x08,0x41]
vadc.vvm v18, v16, v17, v0

# Encoding: |010000|0|10011|10110|100|10100|1010111|
# CHECK: vadc.vxm v20, v19, s6, v0
# CHECK-SAME: [0x57,0x4a,0x3b,0x41]
vadc.vxm v20, v19, s6, v0

# Encoding: |010000|0|10101|10000|011|10110|1010111|
# CHECK: vadc.vim v22, v21, -16, v0
# CHECK-SAME: [0x57,0x3b,0x58,0x41]
vadc.vim v22, v21, -16, v0

# Encoding: |010010|0|10111|11000|000|11001|1010111|
# CHECK: vsbc.vvm v25, v23, v24, v0
# CHECK-SAME: [0xd7,0x0c,0x7c,0x49]
vsbc.vvm v25, v23, v24, v0

# Encoding: |010010|0|11010|11000|100|11011|1010111|
# CHECK: vsbc.vxm v27, v26, s8, v0
# CHECK-SAME: [0xd7,0x4d,0xac,0x49]
vsbc.vxm v27, v26, s8, v0

# Encoding: |010111|1|00000|11100|000|11101|1010111|
# CHECK: vmv.v.v v29, v28
# CHECK-SAME: [0xd7,0x0e,0x0e,0x5e]
vmv.v.v v29, v28

# Encoding: |010111|1|00000|11010|100|11110|1010111|
# CHECK: vmv.v.x v30, s10
# CHECK-SAME: [0x57,0x4f,0x0d,0x5e]
vmv.v.x v30, s10

# Encoding: |010111|1|00000|10001|011|11111|1010111|
# CHECK: vmv.v.i v31, -15
# CHECK-SAME: [0xd7,0xbf,0x08,0x5e]
vmv.v.i v31, -15

# Encoding: |010111|0|00000|00001|000|00010|1010111|
# CHECK: vmerge.vvm v2, v0, v1, v0
# CHECK-SAME: [0x57,0x81,0x00,0x5c]
vmerge.vvm v2, v0, v1, v0

# Encoding: |010111|0|00011|11100|100|00100|1010111|
# CHECK: vmerge.vxm v4, v3, t3, v0
# CHECK-SAME: [0x57,0x42,0x3e,0x5c]
vmerge.vxm v4, v3, t3, v0

# Encoding: |010111|0|00101|10010|011|00110|1010111|
# CHECK: vmerge.vim v6, v5, -14, v0
# CHECK-SAME: [0x57,0x33,0x59,0x5c]
vmerge.vim v6, v5, -14, v0

# Encoding: |011000|1|00111|01000|000|01001|1010111|
# CHECK: vmseq.vv v9, v7, v8
# CHECK-SAME: [0xd7,0x04,0x74,0x62]
vmseq.vv v9, v7, v8
# Encoding: |011000|0|01010|01011|000|01100|1010111|
# CHECK: vmseq.vv v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0x86,0xa5,0x60]
vmseq.vv v12, v10, v11, v0.t

# Encoding: |011000|1|01101|11110|100|01110|1010111|
# CHECK: vmseq.vx v14, v13, t5
# CHECK-SAME: [0x57,0x47,0xdf,0x62]
vmseq.vx v14, v13, t5
# Encoding: |011000|0|01111|00001|100|10000|1010111|
# CHECK: vmseq.vx v16, v15, ra, v0.t
# CHECK-SAME: [0x57,0xc8,0xf0,0x60]
vmseq.vx v16, v15, ra, v0.t

# Encoding: |011000|1|10001|10011|011|10010|1010111|
# CHECK: vmseq.vi v18, v17, -13
# CHECK-SAME: [0x57,0xb9,0x19,0x63]
vmseq.vi v18, v17, -13
# Encoding: |011000|0|10011|10100|011|10100|1010111|
# CHECK: vmseq.vi v20, v19, -12, v0.t
# CHECK-SAME: [0x57,0x3a,0x3a,0x61]
vmseq.vi v20, v19, -12, v0.t

# Encoding: |011001|1|10101|10110|000|10111|1010111|
# CHECK: vmsne.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x0b,0x5b,0x67]
vmsne.vv v23, v21, v22
# Encoding: |011001|0|11000|11001|000|11010|1010111|
# CHECK: vmsne.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0x8d,0x8c,0x65]
vmsne.vv v26, v24, v25, v0.t

# Encoding: |011001|1|11011|00011|100|11100|1010111|
# CHECK: vmsne.vx v28, v27, gp
# CHECK-SAME: [0x57,0xce,0xb1,0x67]
vmsne.vx v28, v27, gp
# Encoding: |011001|0|11101|00101|100|11110|1010111|
# CHECK: vmsne.vx v30, v29, t0, v0.t
# CHECK-SAME: [0x57,0xcf,0xd2,0x65]
vmsne.vx v30, v29, t0, v0.t

# Encoding: |011001|1|11111|10101|011|00000|1010111|
# CHECK: vmsne.vi v0, v31, -11
# CHECK-SAME: [0x57,0xb0,0xfa,0x67]
vmsne.vi v0, v31, -11
# Encoding: |011001|0|00001|10110|011|00010|1010111|
# CHECK: vmsne.vi v2, v1, -10, v0.t
# CHECK-SAME: [0x57,0x31,0x1b,0x64]
vmsne.vi v2, v1, -10, v0.t

# Encoding: |011010|1|00011|00100|000|00101|1010111|
# CHECK: vmsltu.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x02,0x32,0x6a]
vmsltu.vv v5, v3, v4
# Encoding: |011010|0|00110|00111|000|01000|1010111|
# CHECK: vmsltu.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x84,0x63,0x68]
vmsltu.vv v8, v6, v7, v0.t

# Encoding: |011010|1|01001|00111|100|01010|1010111|
# CHECK: vmsltu.vx v10, v9, t2
# CHECK-SAME: [0x57,0xc5,0x93,0x6a]
vmsltu.vx v10, v9, t2
# Encoding: |011010|0|01011|01001|100|01100|1010111|
# CHECK: vmsltu.vx v12, v11, s1, v0.t
# CHECK-SAME: [0x57,0xc6,0xb4,0x68]
vmsltu.vx v12, v11, s1, v0.t

# Encoding: |011011|1|01101|01110|000|01111|1010111|
# CHECK: vmslt.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x07,0xd7,0x6e]
vmslt.vv v15, v13, v14
# Encoding: |011011|0|10000|10001|000|10010|1010111|
# CHECK: vmslt.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x89,0x08,0x6d]
vmslt.vv v18, v16, v17, v0.t

# Encoding: |011011|1|10011|01011|100|10100|1010111|
# CHECK: vmslt.vx v20, v19, a1
# CHECK-SAME: [0x57,0xca,0x35,0x6f]
vmslt.vx v20, v19, a1
# Encoding: |011011|0|10101|01101|100|10110|1010111|
# CHECK: vmslt.vx v22, v21, a3, v0.t
# CHECK-SAME: [0x57,0xcb,0x56,0x6d]
vmslt.vx v22, v21, a3, v0.t

# Encoding: |011100|1|10111|11000|000|11001|1010111|
# CHECK: vmsleu.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x0c,0x7c,0x73]
vmsleu.vv v25, v23, v24
# Encoding: |011100|0|11010|11011|000|11100|1010111|
# CHECK: vmsleu.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x8e,0xad,0x71]
vmsleu.vv v28, v26, v27, v0.t

# Encoding: |011100|1|11101|01111|100|11110|1010111|
# CHECK: vmsleu.vx v30, v29, a5
# CHECK-SAME: [0x57,0xcf,0xd7,0x73]
vmsleu.vx v30, v29, a5
# Encoding: |011100|0|11111|10001|100|00000|1010111|
# CHECK: vmsleu.vx v0, v31, a7, v0.t
# CHECK-SAME: [0x57,0xc0,0xf8,0x71]
vmsleu.vx v0, v31, a7, v0.t

# Encoding: |011100|1|00001|10111|011|00010|1010111|
# CHECK: vmsleu.vi v2, v1, -9
# CHECK-SAME: [0x57,0xb1,0x1b,0x72]
vmsleu.vi v2, v1, -9
# Encoding: |011100|0|00011|11000|011|00100|1010111|
# CHECK: vmsleu.vi v4, v3, -8, v0.t
# CHECK-SAME: [0x57,0x32,0x3c,0x70]
vmsleu.vi v4, v3, -8, v0.t

# Encoding: |011101|1|00101|00110|000|00111|1010111|
# CHECK: vmsle.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x03,0x53,0x76]
vmsle.vv v7, v5, v6
# Encoding: |011101|0|01000|01001|000|01010|1010111|
# CHECK: vmsle.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x85,0x84,0x74]
vmsle.vv v10, v8, v9, v0.t

# Encoding: |011101|1|01011|10011|100|01100|1010111|
# CHECK: vmsle.vx v12, v11, s3
# CHECK-SAME: [0x57,0xc6,0xb9,0x76]
vmsle.vx v12, v11, s3
# Encoding: |011101|0|01101|10101|100|01110|1010111|
# CHECK: vmsle.vx v14, v13, s5, v0.t
# CHECK-SAME: [0x57,0xc7,0xda,0x74]
vmsle.vx v14, v13, s5, v0.t

# Encoding: |011101|1|01111|11001|011|10000|1010111|
# CHECK: vmsle.vi v16, v15, -7
# CHECK-SAME: [0x57,0xb8,0xfc,0x76]
vmsle.vi v16, v15, -7
# Encoding: |011101|0|10001|11010|011|10010|1010111|
# CHECK: vmsle.vi v18, v17, -6, v0.t
# CHECK-SAME: [0x57,0x39,0x1d,0x75]
vmsle.vi v18, v17, -6, v0.t

# Encoding: |011110|1|10011|10111|100|10100|1010111|
# CHECK: vmsgtu.vx v20, v19, s7
# CHECK-SAME: [0x57,0xca,0x3b,0x7b]
vmsgtu.vx v20, v19, s7
# Encoding: |011110|0|10101|11001|100|10110|1010111|
# CHECK: vmsgtu.vx v22, v21, s9, v0.t
# CHECK-SAME: [0x57,0xcb,0x5c,0x79]
vmsgtu.vx v22, v21, s9, v0.t

# Encoding: |011110|1|10111|11011|011|11000|1010111|
# CHECK: vmsgtu.vi v24, v23, -5
# CHECK-SAME: [0x57,0xbc,0x7d,0x7b]
vmsgtu.vi v24, v23, -5
# Encoding: |011110|0|11001|11100|011|11010|1010111|
# CHECK: vmsgtu.vi v26, v25, -4, v0.t
# CHECK-SAME: [0x57,0x3d,0x9e,0x79]
vmsgtu.vi v26, v25, -4, v0.t

# Encoding: |011111|1|11011|11011|100|11100|1010111|
# CHECK: vmsgt.vx v28, v27, s11
# CHECK-SAME: [0x57,0xce,0xbd,0x7f]
vmsgt.vx v28, v27, s11
# Encoding: |011111|0|11101|11101|100|11110|1010111|
# CHECK: vmsgt.vx v30, v29, t4, v0.t
# CHECK-SAME: [0x57,0xcf,0xde,0x7d]
vmsgt.vx v30, v29, t4, v0.t

# Encoding: |011111|1|11111|11101|011|00000|1010111|
# CHECK: vmsgt.vi v0, v31, -3
# CHECK-SAME: [0x57,0xb0,0xfe,0x7f]
vmsgt.vi v0, v31, -3
# Encoding: |011111|0|00001|11110|011|00010|1010111|
# CHECK: vmsgt.vi v2, v1, -2, v0.t
# CHECK-SAME: [0x57,0x31,0x1f,0x7c]
vmsgt.vi v2, v1, -2, v0.t

# Encoding: |100000|1|00011|00100|000|00101|1010111|
# CHECK: vsaddu.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x02,0x32,0x82]
vsaddu.vv v5, v3, v4
# Encoding: |100000|0|00110|00111|000|01000|1010111|
# CHECK: vsaddu.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x84,0x63,0x80]
vsaddu.vv v8, v6, v7, v0.t

# Encoding: |100000|1|01001|11111|100|01010|1010111|
# CHECK: vsaddu.vx v10, v9, t6
# CHECK-SAME: [0x57,0xc5,0x9f,0x82]
vsaddu.vx v10, v9, t6
# Encoding: |100000|0|01011|00010|100|01100|1010111|
# CHECK: vsaddu.vx v12, v11, sp, v0.t
# CHECK-SAME: [0x57,0x46,0xb1,0x80]
vsaddu.vx v12, v11, sp, v0.t

# Encoding: |100000|1|01101|11111|011|01110|1010111|
# CHECK: vsaddu.vi v14, v13, -1
# CHECK-SAME: [0x57,0xb7,0xdf,0x82]
vsaddu.vi v14, v13, -1
# Encoding: |100000|0|01111|00000|011|10000|1010111|
# CHECK: vsaddu.vi v16, v15, 0, v0.t
# CHECK-SAME: [0x57,0x38,0xf0,0x80]
vsaddu.vi v16, v15, 0, v0.t

# Encoding: |100001|1|10001|10010|000|10011|1010111|
# CHECK: vsadd.vv v19, v17, v18
# CHECK-SAME: [0xd7,0x09,0x19,0x87]
vsadd.vv v19, v17, v18
# Encoding: |100001|0|10100|10101|000|10110|1010111|
# CHECK: vsadd.vv v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0x8b,0x4a,0x85]
vsadd.vv v22, v20, v21, v0.t

# Encoding: |100001|1|10111|00100|100|11000|1010111|
# CHECK: vsadd.vx v24, v23, tp
# CHECK-SAME: [0x57,0x4c,0x72,0x87]
vsadd.vx v24, v23, tp
# Encoding: |100001|0|11001|00110|100|11010|1010111|
# CHECK: vsadd.vx v26, v25, t1, v0.t
# CHECK-SAME: [0x57,0x4d,0x93,0x85]
vsadd.vx v26, v25, t1, v0.t

# Encoding: |100001|1|11011|00001|011|11100|1010111|
# CHECK: vsadd.vi v28, v27, 1
# CHECK-SAME: [0x57,0xbe,0xb0,0x87]
vsadd.vi v28, v27, 1
# Encoding: |100001|0|11101|00010|011|11110|1010111|
# CHECK: vsadd.vi v30, v29, 2, v0.t
# CHECK-SAME: [0x57,0x3f,0xd1,0x85]
vsadd.vi v30, v29, 2, v0.t

# Encoding: |100010|1|11111|00000|000|00001|1010111|
# CHECK: vssubu.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x00,0xf0,0x8b]
vssubu.vv v1, v31, v0
# Encoding: |100010|0|00010|00011|000|00100|1010111|
# CHECK: vssubu.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x82,0x21,0x88]
vssubu.vv v4, v2, v3, v0.t

# Encoding: |100010|1|00101|01000|100|00110|1010111|
# CHECK: vssubu.vx v6, v5, s0
# CHECK-SAME: [0x57,0x43,0x54,0x8a]
vssubu.vx v6, v5, s0
# Encoding: |100010|0|00111|01010|100|01000|1010111|
# CHECK: vssubu.vx v8, v7, a0, v0.t
# CHECK-SAME: [0x57,0x44,0x75,0x88]
vssubu.vx v8, v7, a0, v0.t

# Encoding: |100011|1|01001|01010|000|01011|1010111|
# CHECK: vssub.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x05,0x95,0x8e]
vssub.vv v11, v9, v10
# Encoding: |100011|0|01100|01101|000|01110|1010111|
# CHECK: vssub.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x87,0xc6,0x8c]
vssub.vv v14, v12, v13, v0.t

# Encoding: |100011|1|01111|01100|100|10000|1010111|
# CHECK: vssub.vx v16, v15, a2
# CHECK-SAME: [0x57,0x48,0xf6,0x8e]
vssub.vx v16, v15, a2
# Encoding: |100011|0|10001|01110|100|10010|1010111|
# CHECK: vssub.vx v18, v17, a4, v0.t
# CHECK-SAME: [0x57,0x49,0x17,0x8d]
vssub.vx v18, v17, a4, v0.t

# Encoding: |001001|1|10011|10100|010|10101|1010111|
# CHECK: vaadd.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x2a,0x3a,0x27]
vaadd.vv v21, v19, v20
# Encoding: |001001|0|10110|10111|010|11000|1010111|
# CHECK: vaadd.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0xac,0x6b,0x25]
vaadd.vv v24, v22, v23, v0.t

# Encoding: |001001|1|11001|10000|110|11010|1010111|
# CHECK: vaadd.vx v26, v25, a6
# CHECK-SAME: [0x57,0x6d,0x98,0x27]
vaadd.vx v26, v25, a6
# Encoding: |001001|0|11011|10010|110|11100|1010111|
# CHECK: vaadd.vx v28, v27, s2, v0.t
# CHECK-SAME: [0x57,0x6e,0xb9,0x25]
vaadd.vx v28, v27, s2, v0.t

# Encoding: |100101|1|11101|11110|000|11111|1010111|
# CHECK: vsll.vv v31, v29, v30
# CHECK-SAME: [0xd7,0x0f,0xdf,0x97]
vsll.vv v31, v29, v30
# Encoding: |100101|0|00000|00001|000|00010|1010111|
# CHECK: vsll.vv v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x81,0x00,0x94]
vsll.vv v2, v0, v1, v0.t

# Encoding: |100101|1|00011|10100|100|00100|1010111|
# CHECK: vsll.vx v4, v3, s4
# CHECK-SAME: [0x57,0x42,0x3a,0x96]
vsll.vx v4, v3, s4
# Encoding: |100101|0|00101|10110|100|00110|1010111|
# CHECK: vsll.vx v6, v5, s6, v0.t
# CHECK-SAME: [0x57,0x43,0x5b,0x94]
vsll.vx v6, v5, s6, v0.t

# Encoding: |100101|1|00111|00011|011|01000|1010111|
# CHECK: vsll.vi v8, v7, 3
# CHECK-SAME: [0x57,0xb4,0x71,0x96]
vsll.vi v8, v7, 3
# Encoding: |100101|0|01001|00100|011|01010|1010111|
# CHECK: vsll.vi v10, v9, 4, v0.t
# CHECK-SAME: [0x57,0x35,0x92,0x94]
vsll.vi v10, v9, 4, v0.t

# Encoding: |001011|1|01011|01100|010|01101|1010111|
# CHECK: vasub.vv v13, v11, v12
# CHECK-SAME: [0xd7,0x26,0xb6,0x2e]
vasub.vv v13, v11, v12
# Encoding: |001011|0|01110|01111|010|10000|1010111|
# CHECK: vasub.vv v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0xa8,0xe7,0x2c]
vasub.vv v16, v14, v15, v0.t

# Encoding: |001011|1|10001|11000|110|10010|1010111|
# CHECK: vasub.vx v18, v17, s8
# CHECK-SAME: [0x57,0x69,0x1c,0x2f]
vasub.vx v18, v17, s8
# Encoding: |001011|0|10011|11010|110|10100|1010111|
# CHECK: vasub.vx v20, v19, s10, v0.t
# CHECK-SAME: [0x57,0x6a,0x3d,0x2d]
vasub.vx v20, v19, s10, v0.t

# Encoding: |100111|1|10101|10110|000|10111|1010111|
# CHECK: vsmul.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x0b,0x5b,0x9f]
vsmul.vv v23, v21, v22
# Encoding: |100111|0|11000|11001|000|11010|1010111|
# CHECK: vsmul.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0x8d,0x8c,0x9d]
vsmul.vv v26, v24, v25, v0.t

# Encoding: |100111|1|11011|11100|100|11100|1010111|
# CHECK: vsmul.vx v28, v27, t3
# CHECK-SAME: [0x57,0x4e,0xbe,0x9f]
vsmul.vx v28, v27, t3
# Encoding: |100111|0|11101|11110|100|11110|1010111|
# CHECK: vsmul.vx v30, v29, t5, v0.t
# CHECK-SAME: [0x57,0x4f,0xdf,0x9d]
vsmul.vx v30, v29, t5, v0.t

# Encoding: |101000|1|11111|00000|000|00001|1010111|
# CHECK: vsrl.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x00,0xf0,0xa3]
vsrl.vv v1, v31, v0
# Encoding: |101000|0|00010|00011|000|00100|1010111|
# CHECK: vsrl.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x82,0x21,0xa0]
vsrl.vv v4, v2, v3, v0.t

# Encoding: |101000|1|00101|00001|100|00110|1010111|
# CHECK: vsrl.vx v6, v5, ra
# CHECK-SAME: [0x57,0xc3,0x50,0xa2]
vsrl.vx v6, v5, ra
# Encoding: |101000|0|00111|00011|100|01000|1010111|
# CHECK: vsrl.vx v8, v7, gp, v0.t
# CHECK-SAME: [0x57,0xc4,0x71,0xa0]
vsrl.vx v8, v7, gp, v0.t

# Encoding: |101000|1|01001|00101|011|01010|1010111|
# CHECK: vsrl.vi v10, v9, 5
# CHECK-SAME: [0x57,0xb5,0x92,0xa2]
vsrl.vi v10, v9, 5
# Encoding: |101000|0|01011|00110|011|01100|1010111|
# CHECK: vsrl.vi v12, v11, 6, v0.t
# CHECK-SAME: [0x57,0x36,0xb3,0xa0]
vsrl.vi v12, v11, 6, v0.t

# Encoding: |101001|1|01101|01110|000|01111|1010111|
# CHECK: vsra.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x07,0xd7,0xa6]
vsra.vv v15, v13, v14
# Encoding: |101001|0|10000|10001|000|10010|1010111|
# CHECK: vsra.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x89,0x08,0xa5]
vsra.vv v18, v16, v17, v0.t

# Encoding: |101001|1|10011|00101|100|10100|1010111|
# CHECK: vsra.vx v20, v19, t0
# CHECK-SAME: [0x57,0xca,0x32,0xa7]
vsra.vx v20, v19, t0
# Encoding: |101001|0|10101|00111|100|10110|1010111|
# CHECK: vsra.vx v22, v21, t2, v0.t
# CHECK-SAME: [0x57,0xcb,0x53,0xa5]
vsra.vx v22, v21, t2, v0.t

# Encoding: |101001|1|10111|00111|011|11000|1010111|
# CHECK: vsra.vi v24, v23, 7
# CHECK-SAME: [0x57,0xbc,0x73,0xa7]
vsra.vi v24, v23, 7
# Encoding: |101001|0|11001|01000|011|11010|1010111|
# CHECK: vsra.vi v26, v25, 8, v0.t
# CHECK-SAME: [0x57,0x3d,0x94,0xa5]
vsra.vi v26, v25, 8, v0.t

# Encoding: |101010|1|11011|11100|000|11101|1010111|
# CHECK: vssrl.vv v29, v27, v28
# CHECK-SAME: [0xd7,0x0e,0xbe,0xab]
vssrl.vv v29, v27, v28
# Encoding: |101010|0|11110|11111|000|00000|1010111|
# CHECK: vssrl.vv v0, v30, v31, v0.t
# CHECK-SAME: [0x57,0x80,0xef,0xa9]
vssrl.vv v0, v30, v31, v0.t

# Encoding: |101010|1|00001|01001|100|00010|1010111|
# CHECK: vssrl.vx v2, v1, s1
# CHECK-SAME: [0x57,0xc1,0x14,0xaa]
vssrl.vx v2, v1, s1
# Encoding: |101010|0|00011|01011|100|00100|1010111|
# CHECK: vssrl.vx v4, v3, a1, v0.t
# CHECK-SAME: [0x57,0xc2,0x35,0xa8]
vssrl.vx v4, v3, a1, v0.t

# Encoding: |101010|1|00101|01001|011|00110|1010111|
# CHECK: vssrl.vi v6, v5, 9
# CHECK-SAME: [0x57,0xb3,0x54,0xaa]
vssrl.vi v6, v5, 9
# Encoding: |101010|0|00111|01010|011|01000|1010111|
# CHECK: vssrl.vi v8, v7, 10, v0.t
# CHECK-SAME: [0x57,0x34,0x75,0xa8]
vssrl.vi v8, v7, 10, v0.t

# Encoding: |101011|1|01001|01010|000|01011|1010111|
# CHECK: vssra.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x05,0x95,0xae]
vssra.vv v11, v9, v10
# Encoding: |101011|0|01100|01101|000|01110|1010111|
# CHECK: vssra.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x87,0xc6,0xac]
vssra.vv v14, v12, v13, v0.t

# Encoding: |101011|1|01111|01101|100|10000|1010111|
# CHECK: vssra.vx v16, v15, a3
# CHECK-SAME: [0x57,0xc8,0xf6,0xae]
vssra.vx v16, v15, a3
# Encoding: |101011|0|10001|01111|100|10010|1010111|
# CHECK: vssra.vx v18, v17, a5, v0.t
# CHECK-SAME: [0x57,0xc9,0x17,0xad]
vssra.vx v18, v17, a5, v0.t

# Encoding: |101011|1|10011|01011|011|10100|1010111|
# CHECK: vssra.vi v20, v19, 11
# CHECK-SAME: [0x57,0xba,0x35,0xaf]
vssra.vi v20, v19, 11
# Encoding: |101011|0|10101|01100|011|10110|1010111|
# CHECK: vssra.vi v22, v21, 12, v0.t
# CHECK-SAME: [0x57,0x3b,0x56,0xad]
vssra.vi v22, v21, 12, v0.t

# Encoding: |101100|1|10111|11000|000|11001|1010111|
# CHECK: vnsrl.wv v25, v23, v24
# CHECK-SAME: [0xd7,0x0c,0x7c,0xb3]
vnsrl.wv v25, v23, v24
# Encoding: |101100|0|11010|11011|000|11100|1010111|
# CHECK: vnsrl.wv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x8e,0xad,0xb1]
vnsrl.wv v28, v26, v27, v0.t

# Encoding: |101100|1|11101|10001|100|11110|1010111|
# CHECK: vnsrl.wx v30, v29, a7
# CHECK-SAME: [0x57,0xcf,0xd8,0xb3]
vnsrl.wx v30, v29, a7
# Encoding: |101100|0|11111|10011|100|00000|1010111|
# CHECK: vnsrl.wx v0, v31, s3, v0.t
# CHECK-SAME: [0x57,0xc0,0xf9,0xb1]
vnsrl.wx v0, v31, s3, v0.t

# Encoding: |101100|1|00001|01101|011|00010|1010111|
# CHECK: vnsrl.wi v2, v1, 13
# CHECK-SAME: [0x57,0xb1,0x16,0xb2]
vnsrl.wi v2, v1, 13
# Encoding: |101100|0|00011|01110|011|00100|1010111|
# CHECK: vnsrl.wi v4, v3, 14, v0.t
# CHECK-SAME: [0x57,0x32,0x37,0xb0]
vnsrl.wi v4, v3, 14, v0.t

# Encoding: |101101|1|00101|00110|000|00111|1010111|
# CHECK: vnsra.wv v7, v5, v6
# CHECK-SAME: [0xd7,0x03,0x53,0xb6]
vnsra.wv v7, v5, v6
# Encoding: |101101|0|01000|01001|000|01010|1010111|
# CHECK: vnsra.wv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x85,0x84,0xb4]
vnsra.wv v10, v8, v9, v0.t

# Encoding: |101101|1|01011|10101|100|01100|1010111|
# CHECK: vnsra.wx v12, v11, s5
# CHECK-SAME: [0x57,0xc6,0xba,0xb6]
vnsra.wx v12, v11, s5
# Encoding: |101101|0|01101|10111|100|01110|1010111|
# CHECK: vnsra.wx v14, v13, s7, v0.t
# CHECK-SAME: [0x57,0xc7,0xdb,0xb4]
vnsra.wx v14, v13, s7, v0.t

# Encoding: |101101|1|01111|01111|011|10000|1010111|
# CHECK: vnsra.wi v16, v15, 15
# CHECK-SAME: [0x57,0xb8,0xf7,0xb6]
vnsra.wi v16, v15, 15
# Encoding: |101101|0|10001|10000|011|10010|1010111|
# CHECK: vnsra.wi v18, v17, 16, v0.t
# CHECK-SAME: [0x57,0x39,0x18,0xb5]
vnsra.wi v18, v17, 16, v0.t

# Encoding: |101110|1|10011|10100|000|10101|1010111|
# CHECK: vnclipu.wv v21, v19, v20
# CHECK-SAME: [0xd7,0x0a,0x3a,0xbb]
vnclipu.wv v21, v19, v20
# Encoding: |101110|0|10110|10111|000|11000|1010111|
# CHECK: vnclipu.wv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x8c,0x6b,0xb9]
vnclipu.wv v24, v22, v23, v0.t

# Encoding: |101110|1|11001|11001|100|11010|1010111|
# CHECK: vnclipu.wx v26, v25, s9
# CHECK-SAME: [0x57,0xcd,0x9c,0xbb]
vnclipu.wx v26, v25, s9
# Encoding: |101110|0|11011|11011|100|11100|1010111|
# CHECK: vnclipu.wx v28, v27, s11, v0.t
# CHECK-SAME: [0x57,0xce,0xbd,0xb9]
vnclipu.wx v28, v27, s11, v0.t

# Encoding: |101110|1|11101|10001|011|11110|1010111|
# CHECK: vnclipu.wi v30, v29, 17
# CHECK-SAME: [0x57,0xbf,0xd8,0xbb]
vnclipu.wi v30, v29, 17
# Encoding: |101110|0|11111|10010|011|00000|1010111|
# CHECK: vnclipu.wi v0, v31, 18, v0.t
# CHECK-SAME: [0x57,0x30,0xf9,0xb9]
vnclipu.wi v0, v31, 18, v0.t

# Encoding: |101111|1|00001|00010|000|00011|1010111|
# CHECK: vnclip.wv v3, v1, v2
# CHECK-SAME: [0xd7,0x01,0x11,0xbe]
vnclip.wv v3, v1, v2
# Encoding: |101111|0|00100|00101|000|00110|1010111|
# CHECK: vnclip.wv v6, v4, v5, v0.t
# CHECK-SAME: [0x57,0x83,0x42,0xbc]
vnclip.wv v6, v4, v5, v0.t

# Encoding: |101111|1|00111|11101|100|01000|1010111|
# CHECK: vnclip.wx v8, v7, t4
# CHECK-SAME: [0x57,0xc4,0x7e,0xbe]
vnclip.wx v8, v7, t4
# Encoding: |101111|0|01001|11111|100|01010|1010111|
# CHECK: vnclip.wx v10, v9, t6, v0.t
# CHECK-SAME: [0x57,0xc5,0x9f,0xbc]
vnclip.wx v10, v9, t6, v0.t

# Encoding: |101111|1|01011|10011|011|01100|1010111|
# CHECK: vnclip.wi v12, v11, 19
# CHECK-SAME: [0x57,0xb6,0xb9,0xbe]
vnclip.wi v12, v11, 19
# Encoding: |101111|0|01101|10100|011|01110|1010111|
# CHECK: vnclip.wi v14, v13, 20, v0.t
# CHECK-SAME: [0x57,0x37,0xda,0xbc]
vnclip.wi v14, v13, 20, v0.t

# Encoding: |110000|1|01111|10000|000|10010|1010111|
# CHECK: vwredsumu.vs v18, v15, v16
# CHECK-SAME: [0x57,0x09,0xf8,0xc2]
vwredsumu.vs v18, v15, v16
# Encoding: |110000|0|10011|10100|000|10110|1010111|
# CHECK: vwredsumu.vs v22, v19, v20, v0.t
# CHECK-SAME: [0x57,0x0b,0x3a,0xc1]
vwredsumu.vs v22, v19, v20, v0.t

# Encoding: |110001|1|10111|11000|000|11010|1010111|
# CHECK: vwredsum.vs v26, v23, v24
# CHECK-SAME: [0x57,0x0d,0x7c,0xc7]
vwredsum.vs v26, v23, v24
# Encoding: |110001|0|11011|11100|000|11110|1010111|
# CHECK: vwredsum.vs v30, v27, v28, v0.t
# CHECK-SAME: [0x57,0x0f,0xbe,0xc5]
vwredsum.vs v30, v27, v28, v0.t

# Encoding: |000000|1|11111|00000|010|00001|1010111|
# CHECK: vredsum.vs v1, v31, v0
# CHECK-SAME: [0xd7,0x20,0xf0,0x03]
vredsum.vs v1, v31, v0
# Encoding: |000000|0|00010|00011|010|00100|1010111|
# CHECK: vredsum.vs v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0xa2,0x21,0x00]
vredsum.vs v4, v2, v3, v0.t

# Encoding: |000001|1|00101|00110|010|00111|1010111|
# CHECK: vredand.vs v7, v5, v6
# CHECK-SAME: [0xd7,0x23,0x53,0x06]
vredand.vs v7, v5, v6
# Encoding: |000001|0|01000|01001|010|01010|1010111|
# CHECK: vredand.vs v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0xa5,0x84,0x04]
vredand.vs v10, v8, v9, v0.t

# Encoding: |000010|1|01011|01100|010|01101|1010111|
# CHECK: vredor.vs v13, v11, v12
# CHECK-SAME: [0xd7,0x26,0xb6,0x0a]
vredor.vs v13, v11, v12
# Encoding: |000010|0|01110|01111|010|10000|1010111|
# CHECK: vredor.vs v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0xa8,0xe7,0x08]
vredor.vs v16, v14, v15, v0.t

# Encoding: |000011|1|10001|10010|010|10011|1010111|
# CHECK: vredxor.vs v19, v17, v18
# CHECK-SAME: [0xd7,0x29,0x19,0x0f]
vredxor.vs v19, v17, v18
# Encoding: |000011|0|10100|10101|010|10110|1010111|
# CHECK: vredxor.vs v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0xab,0x4a,0x0d]
vredxor.vs v22, v20, v21, v0.t

# Encoding: |000100|1|10111|11000|010|11001|1010111|
# CHECK: vredminu.vs v25, v23, v24
# CHECK-SAME: [0xd7,0x2c,0x7c,0x13]
vredminu.vs v25, v23, v24
# Encoding: |000100|0|11010|11011|010|11100|1010111|
# CHECK: vredminu.vs v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0xae,0xad,0x11]
vredminu.vs v28, v26, v27, v0.t

# Encoding: |000101|1|11101|11110|010|11111|1010111|
# CHECK: vredmin.vs v31, v29, v30
# CHECK-SAME: [0xd7,0x2f,0xdf,0x17]
vredmin.vs v31, v29, v30
# Encoding: |000101|0|00000|00001|010|00010|1010111|
# CHECK: vredmin.vs v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0xa1,0x00,0x14]
vredmin.vs v2, v0, v1, v0.t

# Encoding: |000110|1|00011|00100|010|00101|1010111|
# CHECK: vredmaxu.vs v5, v3, v4
# CHECK-SAME: [0xd7,0x22,0x32,0x1a]
vredmaxu.vs v5, v3, v4
# Encoding: |000110|0|00110|00111|010|01000|1010111|
# CHECK: vredmaxu.vs v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0xa4,0x63,0x18]
vredmaxu.vs v8, v6, v7, v0.t

# Encoding: |000111|1|01001|01010|010|01011|1010111|
# CHECK: vredmax.vs v11, v9, v10
# CHECK-SAME: [0xd7,0x25,0x95,0x1e]
vredmax.vs v11, v9, v10
# Encoding: |000111|0|01100|01101|010|01110|1010111|
# CHECK: vredmax.vs v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0xa7,0xc6,0x1c]
vredmax.vs v14, v12, v13, v0.t

# Encoding: |010000|1|00000|00010|110|01111|1010111|
# CHECK: vmv.s.x v15, sp
# CHECK-SAME: [0xd7,0x67,0x01,0x42]
vmv.s.x v15, sp

# Encoding: |001110|1|10000|00100|110|10001|1010111|
# CHECK: vslide1up.vx v17, v16, tp
# CHECK-SAME: [0xd7,0x68,0x02,0x3b]
vslide1up.vx v17, v16, tp
# Encoding: |001110|0|10010|00110|110|10011|1010111|
# CHECK: vslide1up.vx v19, v18, t1, v0.t
# CHECK-SAME: [0xd7,0x69,0x23,0x39]
vslide1up.vx v19, v18, t1, v0.t

# Encoding: |001111|1|10100|01000|110|10101|1010111|
# CHECK: vslide1down.vx v21, v20, s0
# CHECK-SAME: [0xd7,0x6a,0x44,0x3f]
vslide1down.vx v21, v20, s0
# Encoding: |001111|0|10110|01010|110|10111|1010111|
# CHECK: vslide1down.vx v23, v22, a0, v0.t
# CHECK-SAME: [0xd7,0x6b,0x65,0x3d]
vslide1down.vx v23, v22, a0, v0.t

# Encoding: |010000|1|11000|00000|010|01100|1010111|
# CHECK: vmv.x.s a2, v24
# CHECK-SAME: [0x57,0x26,0x80,0x43]
vmv.x.s a2, v24

# Encoding: |010000|1|11001|10000|010|01110|1010111|
# CHECK: vpopc.m a4, v25
# CHECK-SAME: [0x57,0x27,0x98,0x43]
vpopc.m a4, v25
# Encoding: |010000|0|11010|10000|010|10000|1010111|
# CHECK: vpopc.m a6, v26, v0.t
# CHECK-SAME: [0x57,0x28,0xa8,0x41]
vpopc.m a6, v26, v0.t

# Encoding: |010000|1|11011|10001|010|10010|1010111|
# CHECK: vfirst.m s2, v27
# CHECK-SAME: [0x57,0xa9,0xb8,0x43]
vfirst.m s2, v27
# Encoding: |010000|0|11100|10001|010|10100|1010111|
# CHECK: vfirst.m s4, v28, v0.t
# CHECK-SAME: [0x57,0xaa,0xc8,0x41]
vfirst.m s4, v28, v0.t

# Encoding: |010111|1|11101|11110|010|11111|1010111|
# CHECK: vcompress.vm v31, v29, v30
# CHECK-SAME: [0xd7,0x2f,0xdf,0x5f]
vcompress.vm v31, v29, v30

# Encoding: |011000|1|00000|00001|010|00010|1010111|
# CHECK: vmandnot.mm v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0x62]
vmandnot.mm v2, v0, v1

# Encoding: |011001|1|00011|00100|010|00101|1010111|
# CHECK: vmand.mm v5, v3, v4
# CHECK-SAME: [0xd7,0x22,0x32,0x66]
vmand.mm v5, v3, v4

# Encoding: |011010|1|00110|00111|010|01000|1010111|
# CHECK: vmor.mm v8, v6, v7
# CHECK-SAME: [0x57,0xa4,0x63,0x6a]
vmor.mm v8, v6, v7

# Encoding: |011011|1|01001|01010|010|01011|1010111|
# CHECK: vmxor.mm v11, v9, v10
# CHECK-SAME: [0xd7,0x25,0x95,0x6e]
vmxor.mm v11, v9, v10

# Encoding: |011100|1|01100|01101|010|01110|1010111|
# CHECK: vmornot.mm v14, v12, v13
# CHECK-SAME: [0x57,0xa7,0xc6,0x72]
vmornot.mm v14, v12, v13

# Encoding: |011101|1|01111|10000|010|10001|1010111|
# CHECK: vmnand.mm v17, v15, v16
# CHECK-SAME: [0xd7,0x28,0xf8,0x76]
vmnand.mm v17, v15, v16

# Encoding: |011110|1|10010|10011|010|10100|1010111|
# CHECK: vmnor.mm v20, v18, v19
# CHECK-SAME: [0x57,0xaa,0x29,0x7b]
vmnor.mm v20, v18, v19

# Encoding: |011111|1|10101|10110|010|10111|1010111|
# CHECK: vmxnor.mm v23, v21, v22
# CHECK-SAME: [0xd7,0x2b,0x5b,0x7f]
vmxnor.mm v23, v21, v22

# Encoding: |100000|1|11000|11001|010|11010|1010111|
# CHECK: vdivu.vv v26, v24, v25
# CHECK-SAME: [0x57,0xad,0x8c,0x83]
vdivu.vv v26, v24, v25
# Encoding: |100000|0|11011|11100|010|11101|1010111|
# CHECK: vdivu.vv v29, v27, v28, v0.t
# CHECK-SAME: [0xd7,0x2e,0xbe,0x81]
vdivu.vv v29, v27, v28, v0.t

# Encoding: |100000|1|11110|10110|110|11111|1010111|
# CHECK: vdivu.vx v31, v30, s6
# CHECK-SAME: [0xd7,0x6f,0xeb,0x83]
vdivu.vx v31, v30, s6
# Encoding: |100000|0|00000|11000|110|00001|1010111|
# CHECK: vdivu.vx v1, v0, s8, v0.t
# CHECK-SAME: [0xd7,0x60,0x0c,0x80]
vdivu.vx v1, v0, s8, v0.t

# Encoding: |100001|1|00010|00011|010|00100|1010111|
# CHECK: vdiv.vv v4, v2, v3
# CHECK-SAME: [0x57,0xa2,0x21,0x86]
vdiv.vv v4, v2, v3
# Encoding: |100001|0|00101|00110|010|00111|1010111|
# CHECK: vdiv.vv v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x23,0x53,0x84]
vdiv.vv v7, v5, v6, v0.t

# Encoding: |100001|1|01000|11010|110|01001|1010111|
# CHECK: vdiv.vx v9, v8, s10
# CHECK-SAME: [0xd7,0x64,0x8d,0x86]
vdiv.vx v9, v8, s10
# Encoding: |100001|0|01010|11100|110|01011|1010111|
# CHECK: vdiv.vx v11, v10, t3, v0.t
# CHECK-SAME: [0xd7,0x65,0xae,0x84]
vdiv.vx v11, v10, t3, v0.t

# Encoding: |100010|1|01100|01101|010|01110|1010111|
# CHECK: vremu.vv v14, v12, v13
# CHECK-SAME: [0x57,0xa7,0xc6,0x8a]
vremu.vv v14, v12, v13
# Encoding: |100010|0|01111|10000|010|10001|1010111|
# CHECK: vremu.vv v17, v15, v16, v0.t
# CHECK-SAME: [0xd7,0x28,0xf8,0x88]
vremu.vv v17, v15, v16, v0.t

# Encoding: |100010|1|10010|11110|110|10011|1010111|
# CHECK: vremu.vx v19, v18, t5
# CHECK-SAME: [0xd7,0x69,0x2f,0x8b]
vremu.vx v19, v18, t5
# Encoding: |100010|0|10100|00001|110|10101|1010111|
# CHECK: vremu.vx v21, v20, ra, v0.t
# CHECK-SAME: [0xd7,0xea,0x40,0x89]
vremu.vx v21, v20, ra, v0.t

# Encoding: |100011|1|10110|10111|010|11000|1010111|
# CHECK: vrem.vv v24, v22, v23
# CHECK-SAME: [0x57,0xac,0x6b,0x8f]
vrem.vv v24, v22, v23
# Encoding: |100011|0|11001|11010|010|11011|1010111|
# CHECK: vrem.vv v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x2d,0x9d,0x8d]
vrem.vv v27, v25, v26, v0.t

# Encoding: |100011|1|11100|00011|110|11101|1010111|
# CHECK: vrem.vx v29, v28, gp
# CHECK-SAME: [0xd7,0xee,0xc1,0x8f]
vrem.vx v29, v28, gp
# Encoding: |100011|0|11110|00101|110|11111|1010111|
# CHECK: vrem.vx v31, v30, t0, v0.t
# CHECK-SAME: [0xd7,0xef,0xe2,0x8d]
vrem.vx v31, v30, t0, v0.t

# Encoding: |100100|1|00000|00001|010|00010|1010111|
# CHECK: vmulhu.vv v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0x92]
vmulhu.vv v2, v0, v1
# Encoding: |100100|0|00011|00100|010|00101|1010111|
# CHECK: vmulhu.vv v5, v3, v4, v0.t
# CHECK-SAME: [0xd7,0x22,0x32,0x90]
vmulhu.vv v5, v3, v4, v0.t

# Encoding: |100100|1|00110|00111|110|00111|1010111|
# CHECK: vmulhu.vx v7, v6, t2
# CHECK-SAME: [0xd7,0xe3,0x63,0x92]
vmulhu.vx v7, v6, t2
# Encoding: |100100|0|01000|01001|110|01001|1010111|
# CHECK: vmulhu.vx v9, v8, s1, v0.t
# CHECK-SAME: [0xd7,0xe4,0x84,0x90]
vmulhu.vx v9, v8, s1, v0.t

# Encoding: |100101|1|01010|01011|010|01100|1010111|
# CHECK: vmul.vv v12, v10, v11
# CHECK-SAME: [0x57,0xa6,0xa5,0x96]
vmul.vv v12, v10, v11
# Encoding: |100101|0|01101|01110|010|01111|1010111|
# CHECK: vmul.vv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x27,0xd7,0x94]
vmul.vv v15, v13, v14, v0.t

# Encoding: |100101|1|10000|01011|110|10001|1010111|
# CHECK: vmul.vx v17, v16, a1
# CHECK-SAME: [0xd7,0xe8,0x05,0x97]
vmul.vx v17, v16, a1
# Encoding: |100101|0|10010|01101|110|10011|1010111|
# CHECK: vmul.vx v19, v18, a3, v0.t
# CHECK-SAME: [0xd7,0xe9,0x26,0x95]
vmul.vx v19, v18, a3, v0.t

# Encoding: |100110|1|10100|10101|010|10110|1010111|
# CHECK: vmulhsu.vv v22, v20, v21
# CHECK-SAME: [0x57,0xab,0x4a,0x9b]
vmulhsu.vv v22, v20, v21
# Encoding: |100110|0|10111|11000|010|11001|1010111|
# CHECK: vmulhsu.vv v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x2c,0x7c,0x99]
vmulhsu.vv v25, v23, v24, v0.t

# Encoding: |100110|1|11010|01111|110|11011|1010111|
# CHECK: vmulhsu.vx v27, v26, a5
# CHECK-SAME: [0xd7,0xed,0xa7,0x9b]
vmulhsu.vx v27, v26, a5
# Encoding: |100110|0|11100|10001|110|11101|1010111|
# CHECK: vmulhsu.vx v29, v28, a7, v0.t
# CHECK-SAME: [0xd7,0xee,0xc8,0x99]
vmulhsu.vx v29, v28, a7, v0.t

# Encoding: |100111|1|11110|11111|010|00000|1010111|
# CHECK: vmulh.vv v0, v30, v31
# CHECK-SAME: [0x57,0xa0,0xef,0x9f]
vmulh.vv v0, v30, v31
# Encoding: |100111|0|00001|00010|010|00011|1010111|
# CHECK: vmulh.vv v3, v1, v2, v0.t
# CHECK-SAME: [0xd7,0x21,0x11,0x9c]
vmulh.vv v3, v1, v2, v0.t

# Encoding: |100111|1|00100|10011|110|00101|1010111|
# CHECK: vmulh.vx v5, v4, s3
# CHECK-SAME: [0xd7,0xe2,0x49,0x9e]
vmulh.vx v5, v4, s3
# Encoding: |100111|0|00110|10101|110|00111|1010111|
# CHECK: vmulh.vx v7, v6, s5, v0.t
# CHECK-SAME: [0xd7,0xe3,0x6a,0x9c]
vmulh.vx v7, v6, s5, v0.t

# Encoding: |101001|1|01000|01001|010|01010|1010111|
# CHECK: vmadd.vv v10, v9, v8
# CHECK-SAME: [0x57,0xa5,0x84,0xa6]
vmadd.vv v10, v9, v8
# Encoding: |101001|0|01011|01100|010|01101|1010111|
# CHECK: vmadd.vv v13, v12, v11, v0.t
# CHECK-SAME: [0xd7,0x26,0xb6,0xa4]
vmadd.vv v13, v12, v11, v0.t

# Encoding: |101001|1|01110|10111|110|01111|1010111|
# CHECK: vmadd.vx v15, s7, v14
# CHECK-SAME: [0xd7,0xe7,0xeb,0xa6]
vmadd.vx v15, s7, v14
# Encoding: |101001|0|10000|11001|110|10001|1010111|
# CHECK: vmadd.vx v17, s9, v16, v0.t
# CHECK-SAME: [0xd7,0xe8,0x0c,0xa5]
vmadd.vx v17, s9, v16, v0.t

# Encoding: |101011|1|10010|10011|010|10100|1010111|
# CHECK: vnmsub.vv v20, v19, v18
# CHECK-SAME: [0x57,0xaa,0x29,0xaf]
vnmsub.vv v20, v19, v18
# Encoding: |101011|0|10101|10110|010|10111|1010111|
# CHECK: vnmsub.vv v23, v22, v21, v0.t
# CHECK-SAME: [0xd7,0x2b,0x5b,0xad]
vnmsub.vv v23, v22, v21, v0.t

# Encoding: |101011|1|11000|11011|110|11001|1010111|
# CHECK: vnmsub.vx v25, s11, v24
# CHECK-SAME: [0xd7,0xec,0x8d,0xaf]
vnmsub.vx v25, s11, v24
# Encoding: |101011|0|11010|11101|110|11011|1010111|
# CHECK: vnmsub.vx v27, t4, v26, v0.t
# CHECK-SAME: [0xd7,0xed,0xae,0xad]
vnmsub.vx v27, t4, v26, v0.t

# Encoding: |101101|1|11100|11101|010|11110|1010111|
# CHECK: vmacc.vv v30, v29, v28
# CHECK-SAME: [0x57,0xaf,0xce,0xb7]
vmacc.vv v30, v29, v28
# Encoding: |101101|0|11111|00000|010|00001|1010111|
# CHECK: vmacc.vv v1, v0, v31, v0.t
# CHECK-SAME: [0xd7,0x20,0xf0,0xb5]
vmacc.vv v1, v0, v31, v0.t

# Encoding: |101101|1|00010|11111|110|00011|1010111|
# CHECK: vmacc.vx v3, t6, v2
# CHECK-SAME: [0xd7,0xe1,0x2f,0xb6]
vmacc.vx v3, t6, v2
# Encoding: |101101|0|00100|00010|110|00101|1010111|
# CHECK: vmacc.vx v5, sp, v4, v0.t
# CHECK-SAME: [0xd7,0x62,0x41,0xb4]
vmacc.vx v5, sp, v4, v0.t

# Encoding: |101111|1|00110|00111|010|01000|1010111|
# CHECK: vnmsac.vv v8, v7, v6
# CHECK-SAME: [0x57,0xa4,0x63,0xbe]
vnmsac.vv v8, v7, v6
# Encoding: |101111|0|01001|01010|010|01011|1010111|
# CHECK: vnmsac.vv v11, v10, v9, v0.t
# CHECK-SAME: [0xd7,0x25,0x95,0xbc]
vnmsac.vv v11, v10, v9, v0.t

# Encoding: |101111|1|01100|00100|110|01101|1010111|
# CHECK: vnmsac.vx v13, tp, v12
# CHECK-SAME: [0xd7,0x66,0xc2,0xbe]
vnmsac.vx v13, tp, v12
# Encoding: |101111|0|01110|00110|110|01111|1010111|
# CHECK: vnmsac.vx v15, t1, v14, v0.t
# CHECK-SAME: [0xd7,0x67,0xe3,0xbc]
vnmsac.vx v15, t1, v14, v0.t

# Encoding: |110000|1|10000|10001|010|10010|1010111|
# CHECK: vwaddu.vv v18, v16, v17
# CHECK-SAME: [0x57,0xa9,0x08,0xc3]
vwaddu.vv v18, v16, v17
# Encoding: |110000|0|10011|10100|010|10110|1010111|
# CHECK: vwaddu.vv v22, v19, v20, v0.t
# CHECK-SAME: [0x57,0x2b,0x3a,0xc1]
vwaddu.vv v22, v19, v20, v0.t

# Encoding: |110000|1|10111|01000|110|11000|1010111|
# CHECK: vwaddu.vx v24, v23, s0
# CHECK-SAME: [0x57,0x6c,0x74,0xc3]
vwaddu.vx v24, v23, s0
# Encoding: |110000|0|11001|01010|110|11010|1010111|
# CHECK: vwaddu.vx v26, v25, a0, v0.t
# CHECK-SAME: [0x57,0x6d,0x95,0xc1]
vwaddu.vx v26, v25, a0, v0.t

# Encoding: |110001|1|11011|11100|010|11110|1010111|
# CHECK: vwadd.vv v30, v27, v28
# CHECK-SAME: [0x57,0x2f,0xbe,0xc7]
vwadd.vv v30, v27, v28
# Encoding: |110001|0|11111|00001|010|00010|1010111|
# CHECK: vwadd.vv v2, v31, v1, v0.t
# CHECK-SAME: [0x57,0xa1,0xf0,0xc5]
vwadd.vv v2, v31, v1, v0.t

# Encoding: |110001|1|00011|01100|110|00100|1010111|
# CHECK: vwadd.vx v4, v3, a2
# CHECK-SAME: [0x57,0x62,0x36,0xc6]
vwadd.vx v4, v3, a2
# Encoding: |110001|0|00101|01110|110|00110|1010111|
# CHECK: vwadd.vx v6, v5, a4, v0.t
# CHECK-SAME: [0x57,0x63,0x57,0xc4]
vwadd.vx v6, v5, a4, v0.t

# Encoding: |110010|1|00111|01000|010|01010|1010111|
# CHECK: vwsubu.vv v10, v7, v8
# CHECK-SAME: [0x57,0x25,0x74,0xca]
vwsubu.vv v10, v7, v8
# Encoding: |110010|0|01011|01100|010|01110|1010111|
# CHECK: vwsubu.vv v14, v11, v12, v0.t
# CHECK-SAME: [0x57,0x27,0xb6,0xc8]
vwsubu.vv v14, v11, v12, v0.t

# Encoding: |110010|1|01111|10000|110|10000|1010111|
# CHECK: vwsubu.vx v16, v15, a6
# CHECK-SAME: [0x57,0x68,0xf8,0xca]
vwsubu.vx v16, v15, a6
# Encoding: |110010|0|10001|10010|110|10010|1010111|
# CHECK: vwsubu.vx v18, v17, s2, v0.t
# CHECK-SAME: [0x57,0x69,0x19,0xc9]
vwsubu.vx v18, v17, s2, v0.t

# Encoding: |110011|1|10011|10100|010|10110|1010111|
# CHECK: vwsub.vv v22, v19, v20
# CHECK-SAME: [0x57,0x2b,0x3a,0xcf]
vwsub.vv v22, v19, v20
# Encoding: |110011|0|10111|11000|010|11010|1010111|
# CHECK: vwsub.vv v26, v23, v24, v0.t
# CHECK-SAME: [0x57,0x2d,0x7c,0xcd]
vwsub.vv v26, v23, v24, v0.t

# Encoding: |110011|1|11011|10100|110|11100|1010111|
# CHECK: vwsub.vx v28, v27, s4
# CHECK-SAME: [0x57,0x6e,0xba,0xcf]
vwsub.vx v28, v27, s4
# Encoding: |110011|0|11101|10110|110|11110|1010111|
# CHECK: vwsub.vx v30, v29, s6, v0.t
# CHECK-SAME: [0x57,0x6f,0xdb,0xcd]
vwsub.vx v30, v29, s6, v0.t

# Encoding: |110100|1|11111|00001|010|00010|1010111|
# CHECK: vwaddu.wv v2, v31, v1
# CHECK-SAME: [0x57,0xa1,0xf0,0xd3]
vwaddu.wv v2, v31, v1
# Encoding: |110100|0|00011|00100|010|00110|1010111|
# CHECK: vwaddu.wv v6, v3, v4, v0.t
# CHECK-SAME: [0x57,0x23,0x32,0xd0]
vwaddu.wv v6, v3, v4, v0.t

# Encoding: |110100|1|00111|11000|110|01000|1010111|
# CHECK: vwaddu.wx v8, v7, s8
# CHECK-SAME: [0x57,0x64,0x7c,0xd2]
vwaddu.wx v8, v7, s8
# Encoding: |110100|0|01001|11010|110|01010|1010111|
# CHECK: vwaddu.wx v10, v9, s10, v0.t
# CHECK-SAME: [0x57,0x65,0x9d,0xd0]
vwaddu.wx v10, v9, s10, v0.t

# Encoding: |110101|1|01011|01100|010|01110|1010111|
# CHECK: vwadd.wv v14, v11, v12
# CHECK-SAME: [0x57,0x27,0xb6,0xd6]
vwadd.wv v14, v11, v12
# Encoding: |110101|0|01111|10000|010|10010|1010111|
# CHECK: vwadd.wv v18, v15, v16, v0.t
# CHECK-SAME: [0x57,0x29,0xf8,0xd4]
vwadd.wv v18, v15, v16, v0.t

# Encoding: |110101|1|10011|11100|110|10100|1010111|
# CHECK: vwadd.wx v20, v19, t3
# CHECK-SAME: [0x57,0x6a,0x3e,0xd7]
vwadd.wx v20, v19, t3
# Encoding: |110101|0|10101|11110|110|10110|1010111|
# CHECK: vwadd.wx v22, v21, t5, v0.t
# CHECK-SAME: [0x57,0x6b,0x5f,0xd5]
vwadd.wx v22, v21, t5, v0.t

# Encoding: |110110|1|10111|11000|010|11010|1010111|
# CHECK: vwsubu.wv v26, v23, v24
# CHECK-SAME: [0x57,0x2d,0x7c,0xdb]
vwsubu.wv v26, v23, v24
# Encoding: |110110|0|11011|11100|010|11110|1010111|
# CHECK: vwsubu.wv v30, v27, v28, v0.t
# CHECK-SAME: [0x57,0x2f,0xbe,0xd9]
vwsubu.wv v30, v27, v28, v0.t

# Encoding: |110110|1|11111|00001|110|00010|1010111|
# CHECK: vwsubu.wx v2, v31, ra
# CHECK-SAME: [0x57,0xe1,0xf0,0xdb]
vwsubu.wx v2, v31, ra
# Encoding: |110110|0|00011|00011|110|00100|1010111|
# CHECK: vwsubu.wx v4, v3, gp, v0.t
# CHECK-SAME: [0x57,0xe2,0x31,0xd8]
vwsubu.wx v4, v3, gp, v0.t

# Encoding: |110111|1|00101|00110|010|01000|1010111|
# CHECK: vwsub.wv v8, v5, v6
# CHECK-SAME: [0x57,0x24,0x53,0xde]
vwsub.wv v8, v5, v6
# Encoding: |110111|0|01001|01010|010|01100|1010111|
# CHECK: vwsub.wv v12, v9, v10, v0.t
# CHECK-SAME: [0x57,0x26,0x95,0xdc]
vwsub.wv v12, v9, v10, v0.t

# Encoding: |110111|1|01101|00101|110|01110|1010111|
# CHECK: vwsub.wx v14, v13, t0
# CHECK-SAME: [0x57,0xe7,0xd2,0xde]
vwsub.wx v14, v13, t0
# Encoding: |110111|0|01111|00111|110|10000|1010111|
# CHECK: vwsub.wx v16, v15, t2, v0.t
# CHECK-SAME: [0x57,0xe8,0xf3,0xdc]
vwsub.wx v16, v15, t2, v0.t

# Encoding: |111000|1|10001|10010|010|10100|1010111|
# CHECK: vwmulu.vv v20, v17, v18
# CHECK-SAME: [0x57,0x2a,0x19,0xe3]
vwmulu.vv v20, v17, v18
# Encoding: |111000|0|10101|10110|010|11000|1010111|
# CHECK: vwmulu.vv v24, v21, v22, v0.t
# CHECK-SAME: [0x57,0x2c,0x5b,0xe1]
vwmulu.vv v24, v21, v22, v0.t

# Encoding: |111000|1|11001|01001|110|11010|1010111|
# CHECK: vwmulu.vx v26, v25, s1
# CHECK-SAME: [0x57,0xed,0x94,0xe3]
vwmulu.vx v26, v25, s1
# Encoding: |111000|0|11011|01011|110|11100|1010111|
# CHECK: vwmulu.vx v28, v27, a1, v0.t
# CHECK-SAME: [0x57,0xee,0xb5,0xe1]
vwmulu.vx v28, v27, a1, v0.t

# Encoding: |111010|1|11101|11110|010|00010|1010111|
# CHECK: vwmulsu.vv v2, v29, v30
# CHECK-SAME: [0x57,0x21,0xdf,0xeb]
vwmulsu.vv v2, v29, v30
# Encoding: |111010|0|00011|00100|010|00110|1010111|
# CHECK: vwmulsu.vv v6, v3, v4, v0.t
# CHECK-SAME: [0x57,0x23,0x32,0xe8]
vwmulsu.vv v6, v3, v4, v0.t

# Encoding: |111010|1|00111|01101|110|01000|1010111|
# CHECK: vwmulsu.vx v8, v7, a3
# CHECK-SAME: [0x57,0xe4,0x76,0xea]
vwmulsu.vx v8, v7, a3
# Encoding: |111010|0|01001|01111|110|01010|1010111|
# CHECK: vwmulsu.vx v10, v9, a5, v0.t
# CHECK-SAME: [0x57,0xe5,0x97,0xe8]
vwmulsu.vx v10, v9, a5, v0.t

# Encoding: |111011|1|01011|01100|010|01110|1010111|
# CHECK: vwmul.vv v14, v11, v12
# CHECK-SAME: [0x57,0x27,0xb6,0xee]
vwmul.vv v14, v11, v12
# Encoding: |111011|0|01111|10000|010|10010|1010111|
# CHECK: vwmul.vv v18, v15, v16, v0.t
# CHECK-SAME: [0x57,0x29,0xf8,0xec]
vwmul.vv v18, v15, v16, v0.t

# Encoding: |111011|1|10011|10001|110|10100|1010111|
# CHECK: vwmul.vx v20, v19, a7
# CHECK-SAME: [0x57,0xea,0x38,0xef]
vwmul.vx v20, v19, a7
# Encoding: |111011|0|10101|10011|110|10110|1010111|
# CHECK: vwmul.vx v22, v21, s3, v0.t
# CHECK-SAME: [0x57,0xeb,0x59,0xed]
vwmul.vx v22, v21, s3, v0.t

# Encoding: |111100|1|10111|11000|010|11010|1010111|
# CHECK: vwmaccu.vv v26, v24, v23
# CHECK-SAME: [0x57,0x2d,0x7c,0xf3]
vwmaccu.vv v26, v24, v23
# Encoding: |111100|0|11011|11100|010|11110|1010111|
# CHECK: vwmaccu.vv v30, v28, v27, v0.t
# CHECK-SAME: [0x57,0x2f,0xbe,0xf1]
vwmaccu.vv v30, v28, v27, v0.t

# Encoding: |111100|1|11111|10101|110|00010|1010111|
# CHECK: vwmaccu.vx v2, s5, v31
# CHECK-SAME: [0x57,0xe1,0xfa,0xf3]
vwmaccu.vx v2, s5, v31
# Encoding: |111100|0|00011|10111|110|00100|1010111|
# CHECK: vwmaccu.vx v4, s7, v3, v0.t
# CHECK-SAME: [0x57,0xe2,0x3b,0xf0]
vwmaccu.vx v4, s7, v3, v0.t

# Encoding: |111101|1|00101|00110|010|01000|1010111|
# CHECK: vwmacc.vv v8, v6, v5
# CHECK-SAME: [0x57,0x24,0x53,0xf6]
vwmacc.vv v8, v6, v5
# Encoding: |111101|0|01001|01010|010|01100|1010111|
# CHECK: vwmacc.vv v12, v10, v9, v0.t
# CHECK-SAME: [0x57,0x26,0x95,0xf4]
vwmacc.vv v12, v10, v9, v0.t

# Encoding: |111101|1|01101|11001|110|01110|1010111|
# CHECK: vwmacc.vx v14, s9, v13
# CHECK-SAME: [0x57,0xe7,0xdc,0xf6]
vwmacc.vx v14, s9, v13
# Encoding: |111101|0|01111|11011|110|10000|1010111|
# CHECK: vwmacc.vx v16, s11, v15, v0.t
# CHECK-SAME: [0x57,0xe8,0xfd,0xf4]
vwmacc.vx v16, s11, v15, v0.t

# Encoding: |111110|1|10001|11101|110|10010|1010111|
# CHECK: vwmaccus.vx v18, t4, v17
# CHECK-SAME: [0x57,0xe9,0x1e,0xfb]
vwmaccus.vx v18, t4, v17
# Encoding: |111110|0|10011|11111|110|10100|1010111|
# CHECK: vwmaccus.vx v20, t6, v19, v0.t
# CHECK-SAME: [0x57,0xea,0x3f,0xf9]
vwmaccus.vx v20, t6, v19, v0.t

# Encoding: |111111|1|10101|10110|010|11000|1010111|
# CHECK: vwmaccsu.vv v24, v22, v21
# CHECK-SAME: [0x57,0x2c,0x5b,0xff]
vwmaccsu.vv v24, v22, v21
# Encoding: |111111|0|11001|11010|010|11100|1010111|
# CHECK: vwmaccsu.vv v28, v26, v25, v0.t
# CHECK-SAME: [0x57,0x2e,0x9d,0xfd]
vwmaccsu.vv v28, v26, v25, v0.t

# Encoding: |111111|1|11101|00010|110|11110|1010111|
# CHECK: vwmaccsu.vx v30, sp, v29
# CHECK-SAME: [0x57,0x6f,0xd1,0xff]
vwmaccsu.vx v30, sp, v29
# Encoding: |111111|0|11111|00100|110|00010|1010111|
# CHECK: vwmaccsu.vx v2, tp, v31, v0.t
# CHECK-SAME: [0x57,0x61,0xf2,0xfd]
vwmaccsu.vx v2, tp, v31, v0.t

# Encoding: |000000|1|00011|00100|001|00101|1010111|
# CHECK: vfadd.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x12,0x32,0x02]
vfadd.vv v5, v3, v4
# Encoding: |000000|0|00110|00111|001|01000|1010111|
# CHECK: vfadd.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0x00]
vfadd.vv v8, v6, v7, v0.t

# Encoding: |000000|1|01001|00000|101|01010|1010111|
# CHECK: vfadd.vf v10, v9, ft0
# CHECK-SAME: [0x57,0x55,0x90,0x02]
vfadd.vf v10, v9, ft0
# Encoding: |000000|0|01011|00001|101|01100|1010111|
# CHECK: vfadd.vf v12, v11, ft1, v0.t
# CHECK-SAME: [0x57,0xd6,0xb0,0x00]
vfadd.vf v12, v11, ft1, v0.t

# Encoding: |000001|1|01101|01110|001|01111|1010111|
# CHECK: vfredsum.vs v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0x06]
vfredsum.vs v15, v13, v14
# Encoding: |000001|0|10000|10001|001|10010|1010111|
# CHECK: vfredsum.vs v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0x05]
vfredsum.vs v18, v16, v17, v0.t

# Encoding: |000010|1|10011|10100|001|10101|1010111|
# CHECK: vfsub.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x1a,0x3a,0x0b]
vfsub.vv v21, v19, v20
# Encoding: |000010|0|10110|10111|001|11000|1010111|
# CHECK: vfsub.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x9c,0x6b,0x09]
vfsub.vv v24, v22, v23, v0.t

# Encoding: |000010|1|11001|00010|101|11010|1010111|
# CHECK: vfsub.vf v26, v25, ft2
# CHECK-SAME: [0x57,0x5d,0x91,0x0b]
vfsub.vf v26, v25, ft2
# Encoding: |000010|0|11011|00011|101|11100|1010111|
# CHECK: vfsub.vf v28, v27, ft3, v0.t
# CHECK-SAME: [0x57,0xde,0xb1,0x09]
vfsub.vf v28, v27, ft3, v0.t

# Encoding: |000011|1|11101|11110|001|11111|1010111|
# CHECK: vfredosum.vs v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0x0f]
vfredosum.vs v31, v29, v30
# Encoding: |000011|0|00000|00001|001|00010|1010111|
# CHECK: vfredosum.vs v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0x0c]
vfredosum.vs v2, v0, v1, v0.t

# Encoding: |000100|1|00011|00100|001|00101|1010111|
# CHECK: vfmin.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x12,0x32,0x12]
vfmin.vv v5, v3, v4
# Encoding: |000100|0|00110|00111|001|01000|1010111|
# CHECK: vfmin.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0x10]
vfmin.vv v8, v6, v7, v0.t

# Encoding: |000100|1|01001|00100|101|01010|1010111|
# CHECK: vfmin.vf v10, v9, ft4
# CHECK-SAME: [0x57,0x55,0x92,0x12]
vfmin.vf v10, v9, ft4
# Encoding: |000100|0|01011|00101|101|01100|1010111|
# CHECK: vfmin.vf v12, v11, ft5, v0.t
# CHECK-SAME: [0x57,0xd6,0xb2,0x10]
vfmin.vf v12, v11, ft5, v0.t

# Encoding: |000101|1|01101|01110|001|01111|1010111|
# CHECK: vfredmin.vs v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0x16]
vfredmin.vs v15, v13, v14
# Encoding: |000101|0|10000|10001|001|10010|1010111|
# CHECK: vfredmin.vs v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0x15]
vfredmin.vs v18, v16, v17, v0.t

# Encoding: |000110|1|10011|10100|001|10101|1010111|
# CHECK: vfmax.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x1a,0x3a,0x1b]
vfmax.vv v21, v19, v20
# Encoding: |000110|0|10110|10111|001|11000|1010111|
# CHECK: vfmax.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x9c,0x6b,0x19]
vfmax.vv v24, v22, v23, v0.t

# Encoding: |000110|1|11001|00110|101|11010|1010111|
# CHECK: vfmax.vf v26, v25, ft6
# CHECK-SAME: [0x57,0x5d,0x93,0x1b]
vfmax.vf v26, v25, ft6
# Encoding: |000110|0|11011|00111|101|11100|1010111|
# CHECK: vfmax.vf v28, v27, ft7, v0.t
# CHECK-SAME: [0x57,0xde,0xb3,0x19]
vfmax.vf v28, v27, ft7, v0.t

# Encoding: |000111|1|11101|11110|001|11111|1010111|
# CHECK: vfredmax.vs v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0x1f]
vfredmax.vs v31, v29, v30
# Encoding: |000111|0|00000|00001|001|00010|1010111|
# CHECK: vfredmax.vs v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0x1c]
vfredmax.vs v2, v0, v1, v0.t

# Encoding: |001000|1|00011|00100|001|00101|1010111|
# CHECK: vfsgnj.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x12,0x32,0x22]
vfsgnj.vv v5, v3, v4
# Encoding: |001000|0|00110|00111|001|01000|1010111|
# CHECK: vfsgnj.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0x20]
vfsgnj.vv v8, v6, v7, v0.t

# Encoding: |001000|1|01001|01000|101|01010|1010111|
# CHECK: vfsgnj.vf v10, v9, fs0
# CHECK-SAME: [0x57,0x55,0x94,0x22]
vfsgnj.vf v10, v9, fs0
# Encoding: |001000|0|01011|01001|101|01100|1010111|
# CHECK: vfsgnj.vf v12, v11, fs1, v0.t
# CHECK-SAME: [0x57,0xd6,0xb4,0x20]
vfsgnj.vf v12, v11, fs1, v0.t

# Encoding: |001001|1|01101|01110|001|01111|1010111|
# CHECK: vfsgnjn.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0x26]
vfsgnjn.vv v15, v13, v14
# Encoding: |001001|0|10000|10001|001|10010|1010111|
# CHECK: vfsgnjn.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0x25]
vfsgnjn.vv v18, v16, v17, v0.t

# Encoding: |001001|1|10011|01010|101|10100|1010111|
# CHECK: vfsgnjn.vf v20, v19, fa0
# CHECK-SAME: [0x57,0x5a,0x35,0x27]
vfsgnjn.vf v20, v19, fa0
# Encoding: |001001|0|10101|01011|101|10110|1010111|
# CHECK: vfsgnjn.vf v22, v21, fa1, v0.t
# CHECK-SAME: [0x57,0xdb,0x55,0x25]
vfsgnjn.vf v22, v21, fa1, v0.t

# Encoding: |001010|1|10111|11000|001|11001|1010111|
# CHECK: vfsgnjx.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x1c,0x7c,0x2b]
vfsgnjx.vv v25, v23, v24
# Encoding: |001010|0|11010|11011|001|11100|1010111|
# CHECK: vfsgnjx.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0x29]
vfsgnjx.vv v28, v26, v27, v0.t

# Encoding: |001010|1|11101|01100|101|11110|1010111|
# CHECK: vfsgnjx.vf v30, v29, fa2
# CHECK-SAME: [0x57,0x5f,0xd6,0x2b]
vfsgnjx.vf v30, v29, fa2
# Encoding: |001010|0|11111|01101|101|00000|1010111|
# CHECK: vfsgnjx.vf v0, v31, fa3, v0.t
# CHECK-SAME: [0x57,0xd0,0xf6,0x29]
vfsgnjx.vf v0, v31, fa3, v0.t

# Encoding: |010111|1|00000|01110|101|00001|1010111|
# CHECK: vfmv.v.f v1, fa4
# CHECK-SAME: [0xd7,0x50,0x07,0x5e]
vfmv.v.f v1, fa4

# Encoding: |010000|1|00010|00000|001|01111|1010111|
# CHECK: vfmv.f.s fa5, v2
# CHECK-SAME: [0xd7,0x17,0x20,0x42]
vfmv.f.s fa5, v2

# Encoding: |010000|1|00000|10000|101|00011|1010111|
# CHECK: vfmv.s.f v3, fa6
# CHECK-SAME: [0xd7,0x51,0x08,0x42]
vfmv.s.f v3, fa6

# Encoding: |010111|0|00100|10001|101|00101|1010111|
# CHECK: vfmerge.vfm v5, v4, fa7, v0
# CHECK-SAME: [0xd7,0xd2,0x48,0x5c]
vfmerge.vfm v5, v4, fa7, v0

# Encoding: |011000|1|00110|00111|001|01000|1010111|
# CHECK: vmfeq.vv v8, v6, v7
# CHECK-SAME: [0x57,0x94,0x63,0x62]
vmfeq.vv v8, v6, v7
# Encoding: |011000|0|01001|01010|001|01011|1010111|
# CHECK: vmfeq.vv v11, v9, v10, v0.t
# CHECK-SAME: [0xd7,0x15,0x95,0x60]
vmfeq.vv v11, v9, v10, v0.t

# Encoding: |011000|1|01100|10010|101|01101|1010111|
# CHECK: vmfeq.vf v13, v12, fs2
# CHECK-SAME: [0xd7,0x56,0xc9,0x62]
vmfeq.vf v13, v12, fs2
# Encoding: |011000|0|01110|10011|101|01111|1010111|
# CHECK: vmfeq.vf v15, v14, fs3, v0.t
# CHECK-SAME: [0xd7,0xd7,0xe9,0x60]
vmfeq.vf v15, v14, fs3, v0.t

# Encoding: |011001|1|10000|10001|001|10010|1010111|
# CHECK: vmfle.vv v18, v16, v17
# CHECK-SAME: [0x57,0x99,0x08,0x67]
vmfle.vv v18, v16, v17
# Encoding: |011001|0|10011|10100|001|10101|1010111|
# CHECK: vmfle.vv v21, v19, v20, v0.t
# CHECK-SAME: [0xd7,0x1a,0x3a,0x65]
vmfle.vv v21, v19, v20, v0.t

# Encoding: |011001|1|10110|10100|101|10111|1010111|
# CHECK: vmfle.vf v23, v22, fs4
# CHECK-SAME: [0xd7,0x5b,0x6a,0x67]
vmfle.vf v23, v22, fs4
# Encoding: |011001|0|11000|10101|101|11001|1010111|
# CHECK: vmfle.vf v25, v24, fs5, v0.t
# CHECK-SAME: [0xd7,0xdc,0x8a,0x65]
vmfle.vf v25, v24, fs5, v0.t

# Encoding: |011011|1|11010|11011|001|11100|1010111|
# CHECK: vmflt.vv v28, v26, v27
# CHECK-SAME: [0x57,0x9e,0xad,0x6f]
vmflt.vv v28, v26, v27
# Encoding: |011011|0|11101|11110|001|11111|1010111|
# CHECK: vmflt.vv v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x1f,0xdf,0x6d]
vmflt.vv v31, v29, v30, v0.t

# Encoding: |011011|1|00000|10110|101|00001|1010111|
# CHECK: vmflt.vf v1, v0, fs6
# CHECK-SAME: [0xd7,0x50,0x0b,0x6e]
vmflt.vf v1, v0, fs6
# Encoding: |011011|0|00010|10111|101|00011|1010111|
# CHECK: vmflt.vf v3, v2, fs7, v0.t
# CHECK-SAME: [0xd7,0xd1,0x2b,0x6c]
vmflt.vf v3, v2, fs7, v0.t

# Encoding: |011100|1|00100|00101|001|00110|1010111|
# CHECK: vmfne.vv v6, v4, v5
# CHECK-SAME: [0x57,0x93,0x42,0x72]
vmfne.vv v6, v4, v5
# Encoding: |011100|0|00111|01000|001|01001|1010111|
# CHECK: vmfne.vv v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x14,0x74,0x70]
vmfne.vv v9, v7, v8, v0.t

# Encoding: |011100|1|01010|11000|101|01011|1010111|
# CHECK: vmfne.vf v11, v10, fs8
# CHECK-SAME: [0xd7,0x55,0xac,0x72]
vmfne.vf v11, v10, fs8
# Encoding: |011100|0|01100|11001|101|01101|1010111|
# CHECK: vmfne.vf v13, v12, fs9, v0.t
# CHECK-SAME: [0xd7,0xd6,0xcc,0x70]
vmfne.vf v13, v12, fs9, v0.t

# Encoding: |011101|1|01110|11010|101|01111|1010111|
# CHECK: vmfgt.vf v15, v14, fs10
# CHECK-SAME: [0xd7,0x57,0xed,0x76]
vmfgt.vf v15, v14, fs10
# Encoding: |011101|0|10000|11011|101|10001|1010111|
# CHECK: vmfgt.vf v17, v16, fs11, v0.t
# CHECK-SAME: [0xd7,0xd8,0x0d,0x75]
vmfgt.vf v17, v16, fs11, v0.t

# Encoding: |011111|1|10010|11100|101|10011|1010111|
# CHECK: vmfge.vf v19, v18, ft8
# CHECK-SAME: [0xd7,0x59,0x2e,0x7f]
vmfge.vf v19, v18, ft8
# Encoding: |011111|0|10100|11101|101|10101|1010111|
# CHECK: vmfge.vf v21, v20, ft9, v0.t
# CHECK-SAME: [0xd7,0xda,0x4e,0x7d]
vmfge.vf v21, v20, ft9, v0.t

# Encoding: |100000|1|10110|10111|001|11000|1010111|
# CHECK: vfdiv.vv v24, v22, v23
# CHECK-SAME: [0x57,0x9c,0x6b,0x83]
vfdiv.vv v24, v22, v23
# Encoding: |100000|0|11001|11010|001|11011|1010111|
# CHECK: vfdiv.vv v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x1d,0x9d,0x81]
vfdiv.vv v27, v25, v26, v0.t

# Encoding: |100000|1|11100|11110|101|11101|1010111|
# CHECK: vfdiv.vf v29, v28, ft10
# CHECK-SAME: [0xd7,0x5e,0xcf,0x83]
vfdiv.vf v29, v28, ft10
# Encoding: |100000|0|11110|00000|101|11111|1010111|
# CHECK: vfdiv.vf v31, v30, ft0, v0.t
# CHECK-SAME: [0xd7,0x5f,0xe0,0x81]
vfdiv.vf v31, v30, ft0, v0.t

# Encoding: |100001|1|00000|00001|101|00001|1010111|
# CHECK: vfrdiv.vf v1, v0, ft1
# CHECK-SAME: [0xd7,0xd0,0x00,0x86]
vfrdiv.vf v1, v0, ft1
# Encoding: |100001|0|00010|00010|101|00011|1010111|
# CHECK: vfrdiv.vf v3, v2, ft2, v0.t
# CHECK-SAME: [0xd7,0x51,0x21,0x84]
vfrdiv.vf v3, v2, ft2, v0.t

# Encoding: |100100|1|00100|00101|001|00110|1010111|
# CHECK: vfmul.vv v6, v4, v5
# CHECK-SAME: [0x57,0x93,0x42,0x92]
vfmul.vv v6, v4, v5
# Encoding: |100100|0|00111|01000|001|01001|1010111|
# CHECK: vfmul.vv v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x14,0x74,0x90]
vfmul.vv v9, v7, v8, v0.t

# Encoding: |100100|1|01010|00011|101|01011|1010111|
# CHECK: vfmul.vf v11, v10, ft3
# CHECK-SAME: [0xd7,0xd5,0xa1,0x92]
vfmul.vf v11, v10, ft3
# Encoding: |100100|0|01100|00100|101|01101|1010111|
# CHECK: vfmul.vf v13, v12, ft4, v0.t
# CHECK-SAME: [0xd7,0x56,0xc2,0x90]
vfmul.vf v13, v12, ft4, v0.t

# Encoding: |101000|1|01110|01111|001|10000|1010111|
# CHECK: vfmadd.vv v16, v15, v14
# CHECK-SAME: [0x57,0x98,0xe7,0xa2]
vfmadd.vv v16, v15, v14
# Encoding: |101000|0|10001|10010|001|10011|1010111|
# CHECK: vfmadd.vv v19, v18, v17, v0.t
# CHECK-SAME: [0xd7,0x19,0x19,0xa1]
vfmadd.vv v19, v18, v17, v0.t

# Encoding: |101000|1|10100|00101|101|10101|1010111|
# CHECK: vfmadd.vf v21, ft5, v20
# CHECK-SAME: [0xd7,0xda,0x42,0xa3]
vfmadd.vf v21, ft5, v20
# Encoding: |101000|0|10110|00110|101|10111|1010111|
# CHECK: vfmadd.vf v23, ft6, v22, v0.t
# CHECK-SAME: [0xd7,0x5b,0x63,0xa1]
vfmadd.vf v23, ft6, v22, v0.t

# Encoding: |101001|1|11000|11001|001|11010|1010111|
# CHECK: vfnmadd.vv v26, v25, v24
# CHECK-SAME: [0x57,0x9d,0x8c,0xa7]
vfnmadd.vv v26, v25, v24
# Encoding: |101001|0|11011|11100|001|11101|1010111|
# CHECK: vfnmadd.vv v29, v28, v27, v0.t
# CHECK-SAME: [0xd7,0x1e,0xbe,0xa5]
vfnmadd.vv v29, v28, v27, v0.t

# Encoding: |101001|1|11110|00111|101|11111|1010111|
# CHECK: vfnmadd.vf v31, ft7, v30
# CHECK-SAME: [0xd7,0xdf,0xe3,0xa7]
vfnmadd.vf v31, ft7, v30
# Encoding: |101001|0|00000|01000|101|00001|1010111|
# CHECK: vfnmadd.vf v1, fs0, v0, v0.t
# CHECK-SAME: [0xd7,0x50,0x04,0xa4]
vfnmadd.vf v1, fs0, v0, v0.t

# Encoding: |101010|1|00010|00011|001|00100|1010111|
# CHECK: vfmsub.vv v4, v3, v2
# CHECK-SAME: [0x57,0x92,0x21,0xaa]
vfmsub.vv v4, v3, v2
# Encoding: |101010|0|00101|00110|001|00111|1010111|
# CHECK: vfmsub.vv v7, v6, v5, v0.t
# CHECK-SAME: [0xd7,0x13,0x53,0xa8]
vfmsub.vv v7, v6, v5, v0.t

# Encoding: |101010|1|01000|01001|101|01001|1010111|
# CHECK: vfmsub.vf v9, fs1, v8
# CHECK-SAME: [0xd7,0xd4,0x84,0xaa]
vfmsub.vf v9, fs1, v8
# Encoding: |101010|0|01010|01010|101|01011|1010111|
# CHECK: vfmsub.vf v11, fa0, v10, v0.t
# CHECK-SAME: [0xd7,0x55,0xa5,0xa8]
vfmsub.vf v11, fa0, v10, v0.t

# Encoding: |101011|1|01100|01101|001|01110|1010111|
# CHECK: vfnmsub.vv v14, v13, v12
# CHECK-SAME: [0x57,0x97,0xc6,0xae]
vfnmsub.vv v14, v13, v12
# Encoding: |101011|0|01111|10000|001|10001|1010111|
# CHECK: vfnmsub.vv v17, v16, v15, v0.t
# CHECK-SAME: [0xd7,0x18,0xf8,0xac]
vfnmsub.vv v17, v16, v15, v0.t

# Encoding: |101011|1|10010|01011|101|10011|1010111|
# CHECK: vfnmsub.vf v19, fa1, v18
# CHECK-SAME: [0xd7,0xd9,0x25,0xaf]
vfnmsub.vf v19, fa1, v18
# Encoding: |101011|0|10100|01100|101|10101|1010111|
# CHECK: vfnmsub.vf v21, fa2, v20, v0.t
# CHECK-SAME: [0xd7,0x5a,0x46,0xad]
vfnmsub.vf v21, fa2, v20, v0.t

# Encoding: |101100|1|10110|10111|001|11000|1010111|
# CHECK: vfmacc.vv v24, v23, v22
# CHECK-SAME: [0x57,0x9c,0x6b,0xb3]
vfmacc.vv v24, v23, v22
# Encoding: |101100|0|11001|11010|001|11011|1010111|
# CHECK: vfmacc.vv v27, v26, v25, v0.t
# CHECK-SAME: [0xd7,0x1d,0x9d,0xb1]
vfmacc.vv v27, v26, v25, v0.t

# Encoding: |101100|1|11100|01101|101|11101|1010111|
# CHECK: vfmacc.vf v29, fa3, v28
# CHECK-SAME: [0xd7,0xde,0xc6,0xb3]
vfmacc.vf v29, fa3, v28
# Encoding: |101100|0|11110|01110|101|11111|1010111|
# CHECK: vfmacc.vf v31, fa4, v30, v0.t
# CHECK-SAME: [0xd7,0x5f,0xe7,0xb1]
vfmacc.vf v31, fa4, v30, v0.t

# Encoding: |101101|1|00000|00001|001|00010|1010111|
# CHECK: vfnmacc.vv v2, v1, v0
# CHECK-SAME: [0x57,0x91,0x00,0xb6]
vfnmacc.vv v2, v1, v0
# Encoding: |101101|0|00011|00100|001|00101|1010111|
# CHECK: vfnmacc.vv v5, v4, v3, v0.t
# CHECK-SAME: [0xd7,0x12,0x32,0xb4]
vfnmacc.vv v5, v4, v3, v0.t

# Encoding: |101101|1|00110|01111|101|00111|1010111|
# CHECK: vfnmacc.vf v7, fa5, v6
# CHECK-SAME: [0xd7,0xd3,0x67,0xb6]
vfnmacc.vf v7, fa5, v6
# Encoding: |101101|0|01000|10000|101|01001|1010111|
# CHECK: vfnmacc.vf v9, fa6, v8, v0.t
# CHECK-SAME: [0xd7,0x54,0x88,0xb4]
vfnmacc.vf v9, fa6, v8, v0.t

# Encoding: |101110|1|01010|01011|001|01100|1010111|
# CHECK: vfmsac.vv v12, v11, v10
# CHECK-SAME: [0x57,0x96,0xa5,0xba]
vfmsac.vv v12, v11, v10
# Encoding: |101110|0|01101|01110|001|01111|1010111|
# CHECK: vfmsac.vv v15, v14, v13, v0.t
# CHECK-SAME: [0xd7,0x17,0xd7,0xb8]
vfmsac.vv v15, v14, v13, v0.t

# Encoding: |101110|1|10000|10001|101|10001|1010111|
# CHECK: vfmsac.vf v17, fa7, v16
# CHECK-SAME: [0xd7,0xd8,0x08,0xbb]
vfmsac.vf v17, fa7, v16
# Encoding: |101110|0|10010|10010|101|10011|1010111|
# CHECK: vfmsac.vf v19, fs2, v18, v0.t
# CHECK-SAME: [0xd7,0x59,0x29,0xb9]
vfmsac.vf v19, fs2, v18, v0.t

# Encoding: |101111|1|10100|10101|001|10110|1010111|
# CHECK: vfnmsac.vv v22, v21, v20
# CHECK-SAME: [0x57,0x9b,0x4a,0xbf]
vfnmsac.vv v22, v21, v20
# Encoding: |101111|0|10111|11000|001|11001|1010111|
# CHECK: vfnmsac.vv v25, v24, v23, v0.t
# CHECK-SAME: [0xd7,0x1c,0x7c,0xbd]
vfnmsac.vv v25, v24, v23, v0.t

# Encoding: |101111|1|11010|10011|101|11011|1010111|
# CHECK: vfnmsac.vf v27, fs3, v26
# CHECK-SAME: [0xd7,0xdd,0xa9,0xbf]
vfnmsac.vf v27, fs3, v26
# Encoding: |101111|0|11100|10100|101|11101|1010111|
# CHECK: vfnmsac.vf v29, fs4, v28, v0.t
# CHECK-SAME: [0xd7,0x5e,0xca,0xbd]
vfnmsac.vf v29, fs4, v28, v0.t

# Encoding: |110000|1|11110|11111|001|00010|1010111|
# CHECK: vfwadd.vv v2, v30, v31
# CHECK-SAME: [0x57,0x91,0xef,0xc3]
vfwadd.vv v2, v30, v31
# Encoding: |110000|0|00011|00100|001|00110|1010111|
# CHECK: vfwadd.vv v6, v3, v4, v0.t
# CHECK-SAME: [0x57,0x13,0x32,0xc0]
vfwadd.vv v6, v3, v4, v0.t

# Encoding: |110000|1|00111|10101|101|01000|1010111|
# CHECK: vfwadd.vf v8, v7, fs5
# CHECK-SAME: [0x57,0xd4,0x7a,0xc2]
vfwadd.vf v8, v7, fs5
# Encoding: |110000|0|01001|10110|101|01010|1010111|
# CHECK: vfwadd.vf v10, v9, fs6, v0.t
# CHECK-SAME: [0x57,0x55,0x9b,0xc0]
vfwadd.vf v10, v9, fs6, v0.t

# Encoding: |110001|1|01011|01100|001|01110|1010111|
# CHECK: vfwredsum.vs v14, v11, v12
# CHECK-SAME: [0x57,0x17,0xb6,0xc6]
vfwredsum.vs v14, v11, v12
# Encoding: |110001|0|01111|10000|001|10010|1010111|
# CHECK: vfwredsum.vs v18, v15, v16, v0.t
# CHECK-SAME: [0x57,0x19,0xf8,0xc4]
vfwredsum.vs v18, v15, v16, v0.t

# Encoding: |110010|1|10011|10100|001|10110|1010111|
# CHECK: vfwsub.vv v22, v19, v20
# CHECK-SAME: [0x57,0x1b,0x3a,0xcb]
vfwsub.vv v22, v19, v20
# Encoding: |110010|0|10111|11000|001|11010|1010111|
# CHECK: vfwsub.vv v26, v23, v24, v0.t
# CHECK-SAME: [0x57,0x1d,0x7c,0xc9]
vfwsub.vv v26, v23, v24, v0.t

# Encoding: |110010|1|11011|10111|101|11100|1010111|
# CHECK: vfwsub.vf v28, v27, fs7
# CHECK-SAME: [0x57,0xde,0xbb,0xcb]
vfwsub.vf v28, v27, fs7
# Encoding: |110010|0|11101|11000|101|11110|1010111|
# CHECK: vfwsub.vf v30, v29, fs8, v0.t
# CHECK-SAME: [0x57,0x5f,0xdc,0xc9]
vfwsub.vf v30, v29, fs8, v0.t

# Encoding: |110011|1|11111|00001|001|00010|1010111|
# CHECK: vfwredosum.vs v2, v31, v1
# CHECK-SAME: [0x57,0x91,0xf0,0xcf]
vfwredosum.vs v2, v31, v1
# Encoding: |110011|0|00011|00100|001|00110|1010111|
# CHECK: vfwredosum.vs v6, v3, v4, v0.t
# CHECK-SAME: [0x57,0x13,0x32,0xcc]
vfwredosum.vs v6, v3, v4, v0.t

# Encoding: |110100|1|00111|01000|001|01010|1010111|
# CHECK: vfwadd.wv v10, v7, v8
# CHECK-SAME: [0x57,0x15,0x74,0xd2]
vfwadd.wv v10, v7, v8
# Encoding: |110100|0|01011|01100|001|01110|1010111|
# CHECK: vfwadd.wv v14, v11, v12, v0.t
# CHECK-SAME: [0x57,0x17,0xb6,0xd0]
vfwadd.wv v14, v11, v12, v0.t

# Encoding: |110100|1|01111|11001|101|10000|1010111|
# CHECK: vfwadd.wf v16, v15, fs9
# CHECK-SAME: [0x57,0xd8,0xfc,0xd2]
vfwadd.wf v16, v15, fs9
# Encoding: |110100|0|10001|11010|101|10010|1010111|
# CHECK: vfwadd.wf v18, v17, fs10, v0.t
# CHECK-SAME: [0x57,0x59,0x1d,0xd1]
vfwadd.wf v18, v17, fs10, v0.t

# Encoding: |110110|1|10011|10100|001|10110|1010111|
# CHECK: vfwsub.wv v22, v19, v20
# CHECK-SAME: [0x57,0x1b,0x3a,0xdb]
vfwsub.wv v22, v19, v20
# Encoding: |110110|0|10111|11000|001|11010|1010111|
# CHECK: vfwsub.wv v26, v23, v24, v0.t
# CHECK-SAME: [0x57,0x1d,0x7c,0xd9]
vfwsub.wv v26, v23, v24, v0.t

# Encoding: |110110|1|11011|11011|101|11100|1010111|
# CHECK: vfwsub.wf v28, v27, fs11
# CHECK-SAME: [0x57,0xde,0xbd,0xdb]
vfwsub.wf v28, v27, fs11
# Encoding: |110110|0|11101|11100|101|11110|1010111|
# CHECK: vfwsub.wf v30, v29, ft8, v0.t
# CHECK-SAME: [0x57,0x5f,0xde,0xd9]
vfwsub.wf v30, v29, ft8, v0.t

# Encoding: |111000|1|11111|00001|001|00010|1010111|
# CHECK: vfwmul.vv v2, v31, v1
# CHECK-SAME: [0x57,0x91,0xf0,0xe3]
vfwmul.vv v2, v31, v1
# Encoding: |111000|0|00011|00100|001|00110|1010111|
# CHECK: vfwmul.vv v6, v3, v4, v0.t
# CHECK-SAME: [0x57,0x13,0x32,0xe0]
vfwmul.vv v6, v3, v4, v0.t

# Encoding: |111000|1|00111|11101|101|01000|1010111|
# CHECK: vfwmul.vf v8, v7, ft9
# CHECK-SAME: [0x57,0xd4,0x7e,0xe2]
vfwmul.vf v8, v7, ft9
# Encoding: |111000|0|01001|11110|101|01010|1010111|
# CHECK: vfwmul.vf v10, v9, ft10, v0.t
# CHECK-SAME: [0x57,0x55,0x9f,0xe0]
vfwmul.vf v10, v9, ft10, v0.t

# Encoding: |111100|1|01011|01100|001|01110|1010111|
# CHECK: vfwmacc.vv v14, v12, v11
# CHECK-SAME: [0x57,0x17,0xb6,0xf2]
vfwmacc.vv v14, v12, v11
# Encoding: |111100|0|01111|10000|001|10010|1010111|
# CHECK: vfwmacc.vv v18, v16, v15, v0.t
# CHECK-SAME: [0x57,0x19,0xf8,0xf0]
vfwmacc.vv v18, v16, v15, v0.t

# Encoding: |111100|1|10011|00000|101|10100|1010111|
# CHECK: vfwmacc.vf v20, ft0, v19
# CHECK-SAME: [0x57,0x5a,0x30,0xf3]
vfwmacc.vf v20, ft0, v19
# Encoding: |111100|0|10101|00001|101|10110|1010111|
# CHECK: vfwmacc.vf v22, ft1, v21, v0.t
# CHECK-SAME: [0x57,0xdb,0x50,0xf1]
vfwmacc.vf v22, ft1, v21, v0.t

# Encoding: |111101|1|10111|11000|001|11010|1010111|
# CHECK: vfwnmacc.vv v26, v24, v23
# CHECK-SAME: [0x57,0x1d,0x7c,0xf7]
vfwnmacc.vv v26, v24, v23
# Encoding: |111101|0|11011|11100|001|11110|1010111|
# CHECK: vfwnmacc.vv v30, v28, v27, v0.t
# CHECK-SAME: [0x57,0x1f,0xbe,0xf5]
vfwnmacc.vv v30, v28, v27, v0.t

# Encoding: |111101|1|11111|00010|101|00010|1010111|
# CHECK: vfwnmacc.vf v2, ft2, v31
# CHECK-SAME: [0x57,0x51,0xf1,0xf7]
vfwnmacc.vf v2, ft2, v31
# Encoding: |111101|0|00011|00011|101|00100|1010111|
# CHECK: vfwnmacc.vf v4, ft3, v3, v0.t
# CHECK-SAME: [0x57,0xd2,0x31,0xf4]
vfwnmacc.vf v4, ft3, v3, v0.t

# Encoding: |111110|1|00101|00110|001|01000|1010111|
# CHECK: vfwmsac.vv v8, v6, v5
# CHECK-SAME: [0x57,0x14,0x53,0xfa]
vfwmsac.vv v8, v6, v5
# Encoding: |111110|0|01001|01010|001|01100|1010111|
# CHECK: vfwmsac.vv v12, v10, v9, v0.t
# CHECK-SAME: [0x57,0x16,0x95,0xf8]
vfwmsac.vv v12, v10, v9, v0.t

# Encoding: |111110|1|01101|00100|101|01110|1010111|
# CHECK: vfwmsac.vf v14, ft4, v13
# CHECK-SAME: [0x57,0x57,0xd2,0xfa]
vfwmsac.vf v14, ft4, v13
# Encoding: |111110|0|01111|00101|101|10000|1010111|
# CHECK: vfwmsac.vf v16, ft5, v15, v0.t
# CHECK-SAME: [0x57,0xd8,0xf2,0xf8]
vfwmsac.vf v16, ft5, v15, v0.t

# Encoding: |111111|1|10001|10010|001|10100|1010111|
# CHECK: vfwnmsac.vv v20, v18, v17
# CHECK-SAME: [0x57,0x1a,0x19,0xff]
vfwnmsac.vv v20, v18, v17
# Encoding: |111111|0|10101|10110|001|11000|1010111|
# CHECK: vfwnmsac.vv v24, v22, v21, v0.t
# CHECK-SAME: [0x57,0x1c,0x5b,0xfd]
vfwnmsac.vv v24, v22, v21, v0.t

# Encoding: |111111|1|11001|00110|101|11010|1010111|
# CHECK: vfwnmsac.vf v26, ft6, v25
# CHECK-SAME: [0x57,0x5d,0x93,0xff]
vfwnmsac.vf v26, ft6, v25
# Encoding: |111111|0|11011|00111|101|11100|1010111|
# CHECK: vfwnmsac.vf v28, ft7, v27, v0.t
# CHECK-SAME: [0x57,0xde,0xb3,0xfd]
vfwnmsac.vf v28, ft7, v27, v0.t

# Encoding: |100011|1|11101|00000|001|11110|1010111|
# CHECK: vfsqrt.v v30, v29
# CHECK-SAME: [0x57,0x1f,0xd0,0x8f]
vfsqrt.v v30, v29
# Encoding: |100011|0|11111|00000|001|00000|1010111|
# CHECK: vfsqrt.v v0, v31, v0.t
# CHECK-SAME: [0x57,0x10,0xf0,0x8d]
vfsqrt.v v0, v31, v0.t

# Encoding: |100011|1|00001|10000|001|00010|1010111|
# CHECK: vfclass.v v2, v1
# CHECK-SAME: [0x57,0x11,0x18,0x8e]
vfclass.v v2, v1
# Encoding: |100011|0|00011|10000|001|00100|1010111|
# CHECK: vfclass.v v4, v3, v0.t
# CHECK-SAME: [0x57,0x12,0x38,0x8c]
vfclass.v v4, v3, v0.t

# Encoding: |100010|1|00101|00000|001|00110|1010111|
# CHECK: vfcvt.xu.f.v v6, v5
# CHECK-SAME: [0x57,0x13,0x50,0x8a]
vfcvt.xu.f.v v6, v5
# Encoding: |100010|0|00111|00000|001|01000|1010111|
# CHECK: vfcvt.xu.f.v v8, v7, v0.t
# CHECK-SAME: [0x57,0x14,0x70,0x88]
vfcvt.xu.f.v v8, v7, v0.t

# Encoding: |100010|1|01001|00001|001|01010|1010111|
# CHECK: vfcvt.x.f.v v10, v9
# CHECK-SAME: [0x57,0x95,0x90,0x8a]
vfcvt.x.f.v v10, v9
# Encoding: |100010|0|01011|00001|001|01100|1010111|
# CHECK: vfcvt.x.f.v v12, v11, v0.t
# CHECK-SAME: [0x57,0x96,0xb0,0x88]
vfcvt.x.f.v v12, v11, v0.t

# Encoding: |100010|1|01101|00010|001|01110|1010111|
# CHECK: vfcvt.f.xu.v v14, v13
# CHECK-SAME: [0x57,0x17,0xd1,0x8a]
vfcvt.f.xu.v v14, v13
# Encoding: |100010|0|01111|00010|001|10000|1010111|
# CHECK: vfcvt.f.xu.v v16, v15, v0.t
# CHECK-SAME: [0x57,0x18,0xf1,0x88]
vfcvt.f.xu.v v16, v15, v0.t

# Encoding: |100010|1|10001|00011|001|10010|1010111|
# CHECK: vfcvt.f.x.v v18, v17
# CHECK-SAME: [0x57,0x99,0x11,0x8b]
vfcvt.f.x.v v18, v17
# Encoding: |100010|0|10011|00011|001|10100|1010111|
# CHECK: vfcvt.f.x.v v20, v19, v0.t
# CHECK-SAME: [0x57,0x9a,0x31,0x89]
vfcvt.f.x.v v20, v19, v0.t

# Encoding: |100010|1|10101|01000|001|10110|1010111|
# CHECK: vfwcvt.xu.f.v v22, v21
# CHECK-SAME: [0x57,0x1b,0x54,0x8b]
vfwcvt.xu.f.v v22, v21
# Encoding: |100010|0|10111|01000|001|11000|1010111|
# CHECK: vfwcvt.xu.f.v v24, v23, v0.t
# CHECK-SAME: [0x57,0x1c,0x74,0x89]
vfwcvt.xu.f.v v24, v23, v0.t

# Encoding: |100010|1|11001|01001|001|11010|1010111|
# CHECK: vfwcvt.x.f.v v26, v25
# CHECK-SAME: [0x57,0x9d,0x94,0x8b]
vfwcvt.x.f.v v26, v25
# Encoding: |100010|0|11011|01001|001|11100|1010111|
# CHECK: vfwcvt.x.f.v v28, v27, v0.t
# CHECK-SAME: [0x57,0x9e,0xb4,0x89]
vfwcvt.x.f.v v28, v27, v0.t

# Encoding: |100010|1|11101|01010|001|11110|1010111|
# CHECK: vfwcvt.f.xu.v v30, v29
# CHECK-SAME: [0x57,0x1f,0xd5,0x8b]
vfwcvt.f.xu.v v30, v29
# Encoding: |100010|0|11111|01010|001|00010|1010111|
# CHECK: vfwcvt.f.xu.v v2, v31, v0.t
# CHECK-SAME: [0x57,0x11,0xf5,0x89]
vfwcvt.f.xu.v v2, v31, v0.t

# Encoding: |100010|1|00011|01011|001|00100|1010111|
# CHECK: vfwcvt.f.x.v v4, v3
# CHECK-SAME: [0x57,0x92,0x35,0x8a]
vfwcvt.f.x.v v4, v3
# Encoding: |100010|0|00101|01011|001|00110|1010111|
# CHECK: vfwcvt.f.x.v v6, v5, v0.t
# CHECK-SAME: [0x57,0x93,0x55,0x88]
vfwcvt.f.x.v v6, v5, v0.t

# Encoding: |100010|1|00111|01100|001|01000|1010111|
# CHECK: vfwcvt.f.f.v v8, v7
# CHECK-SAME: [0x57,0x14,0x76,0x8a]
vfwcvt.f.f.v v8, v7
# Encoding: |100010|0|01001|01100|001|01010|1010111|
# CHECK: vfwcvt.f.f.v v10, v9, v0.t
# CHECK-SAME: [0x57,0x15,0x96,0x88]
vfwcvt.f.f.v v10, v9, v0.t

# Encoding: |100010|1|01011|10000|001|01100|1010111|
# CHECK: vfncvt.xu.f.w v12, v11
# CHECK-SAME: [0x57,0x16,0xb8,0x8a]
vfncvt.xu.f.w v12, v11
# Encoding: |100010|0|01101|10000|001|01110|1010111|
# CHECK: vfncvt.xu.f.w v14, v13, v0.t
# CHECK-SAME: [0x57,0x17,0xd8,0x88]
vfncvt.xu.f.w v14, v13, v0.t

# Encoding: |100010|1|01111|10001|001|10000|1010111|
# CHECK: vfncvt.x.f.w v16, v15
# CHECK-SAME: [0x57,0x98,0xf8,0x8a]
vfncvt.x.f.w v16, v15
# Encoding: |100010|0|10001|10001|001|10010|1010111|
# CHECK: vfncvt.x.f.w v18, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x18,0x89]
vfncvt.x.f.w v18, v17, v0.t

# Encoding: |100010|1|10011|10010|001|10100|1010111|
# CHECK: vfncvt.f.xu.w v20, v19
# CHECK-SAME: [0x57,0x1a,0x39,0x8b]
vfncvt.f.xu.w v20, v19
# Encoding: |100010|0|10101|10010|001|10110|1010111|
# CHECK: vfncvt.f.xu.w v22, v21, v0.t
# CHECK-SAME: [0x57,0x1b,0x59,0x89]
vfncvt.f.xu.w v22, v21, v0.t

# Encoding: |100010|1|10111|10011|001|11000|1010111|
# CHECK: vfncvt.f.x.w v24, v23
# CHECK-SAME: [0x57,0x9c,0x79,0x8b]
vfncvt.f.x.w v24, v23
# Encoding: |100010|0|11001|10011|001|11010|1010111|
# CHECK: vfncvt.f.x.w v26, v25, v0.t
# CHECK-SAME: [0x57,0x9d,0x99,0x89]
vfncvt.f.x.w v26, v25, v0.t

# Encoding: |100010|1|11011|10100|001|11100|1010111|
# CHECK: vfncvt.f.f.w v28, v27
# CHECK-SAME: [0x57,0x1e,0xba,0x8b]
vfncvt.f.f.w v28, v27
# Encoding: |100010|0|11101|10100|001|11110|1010111|
# CHECK: vfncvt.f.f.w v30, v29, v0.t
# CHECK-SAME: [0x57,0x1f,0xda,0x89]
vfncvt.f.f.w v30, v29, v0.t

# Encoding: |010100|1|11111|00001|010|00000|1010111|
# CHECK: vmsbf.m v0, v31
# CHECK-SAME: [0x57,0xa0,0xf0,0x53]
vmsbf.m v0, v31
# Encoding: |010100|0|00001|00001|010|00010|1010111|
# CHECK: vmsbf.m v2, v1, v0.t
# CHECK-SAME: [0x57,0xa1,0x10,0x50]
vmsbf.m v2, v1, v0.t

# Encoding: |010100|1|00011|00010|010|00100|1010111|
# CHECK: vmsof.m v4, v3
# CHECK-SAME: [0x57,0x22,0x31,0x52]
vmsof.m v4, v3
# Encoding: |010100|0|00101|00010|010|00110|1010111|
# CHECK: vmsof.m v6, v5, v0.t
# CHECK-SAME: [0x57,0x23,0x51,0x50]
vmsof.m v6, v5, v0.t

# Encoding: |010100|1|00111|00011|010|01000|1010111|
# CHECK: vmsif.m v8, v7
# CHECK-SAME: [0x57,0xa4,0x71,0x52]
vmsif.m v8, v7
# Encoding: |010100|0|01001|00011|010|01010|1010111|
# CHECK: vmsif.m v10, v9, v0.t
# CHECK-SAME: [0x57,0xa5,0x91,0x50]
vmsif.m v10, v9, v0.t

# Encoding: |010100|1|01011|10000|010|01100|1010111|
# CHECK: viota.m v12, v11
# CHECK-SAME: [0x57,0x26,0xb8,0x52]
viota.m v12, v11
# Encoding: |010100|0|01101|10000|010|01110|1010111|
# CHECK: viota.m v14, v13, v0.t
# CHECK-SAME: [0x57,0x27,0xd8,0x50]
viota.m v14, v13, v0.t

# Encoding: |010100|1|00000|10001|010|01111|1010111|
# CHECK: vid.v v15
# CHECK-SAME: [0xd7,0xa7,0x08,0x52]
vid.v v15
# Encoding: |010100|0|00000|10001|010|10000|1010111|
# CHECK: vid.v v16, v0.t
# CHECK-SAME: [0x57,0xa8,0x08,0x50]
vid.v v16, v0.t

# Encoding: |000|100|1|00000|00110|000|10001|0000111|
# CHECK: vlb.v v17, (t1)
# CHECK-SAME: [0x87,0x08,0x03,0x12]
vlb.v v17, (t1)
# Encoding: |000|100|0|00000|01000|000|10010|0000111|
# CHECK: vlb.v v18, (s0), v0.t
# CHECK-SAME: [0x07,0x09,0x04,0x10]
vlb.v v18, (s0), v0.t

# Encoding: |000|100|1|00000|01010|101|10011|0000111|
# CHECK: vlh.v v19, (a0)
# CHECK-SAME: [0x87,0x59,0x05,0x12]
vlh.v v19, (a0)
# Encoding: |000|100|0|00000|01100|101|10100|0000111|
# CHECK: vlh.v v20, (a2), v0.t
# CHECK-SAME: [0x07,0x5a,0x06,0x10]
vlh.v v20, (a2), v0.t

# Encoding: |000|100|1|00000|01110|110|10101|0000111|
# CHECK: vlw.v v21, (a4)
# CHECK-SAME: [0x87,0x6a,0x07,0x12]
vlw.v v21, (a4)
# Encoding: |000|100|0|00000|10000|110|10110|0000111|
# CHECK: vlw.v v22, (a6), v0.t
# CHECK-SAME: [0x07,0x6b,0x08,0x10]
vlw.v v22, (a6), v0.t

# Encoding: |000|000|1|00000|10010|000|10111|0000111|
# CHECK: vlbu.v v23, (s2)
# CHECK-SAME: [0x87,0x0b,0x09,0x02]
vlbu.v v23, (s2)
# Encoding: |000|000|0|00000|10100|000|11000|0000111|
# CHECK: vlbu.v v24, (s4), v0.t
# CHECK-SAME: [0x07,0x0c,0x0a,0x00]
vlbu.v v24, (s4), v0.t

# Encoding: |000|000|1|00000|10110|101|11001|0000111|
# CHECK: vlhu.v v25, (s6)
# CHECK-SAME: [0x87,0x5c,0x0b,0x02]
vlhu.v v25, (s6)
# Encoding: |000|000|0|00000|11000|101|11010|0000111|
# CHECK: vlhu.v v26, (s8), v0.t
# CHECK-SAME: [0x07,0x5d,0x0c,0x00]
vlhu.v v26, (s8), v0.t

# Encoding: |000|000|1|00000|11010|110|11011|0000111|
# CHECK: vlwu.v v27, (s10)
# CHECK-SAME: [0x87,0x6d,0x0d,0x02]
vlwu.v v27, (s10)
# Encoding: |000|000|0|00000|11100|110|11100|0000111|
# CHECK: vlwu.v v28, (t3), v0.t
# CHECK-SAME: [0x07,0x6e,0x0e,0x00]
vlwu.v v28, (t3), v0.t

# Encoding: |000|000|1|00000|11110|111|11101|0000111|
# CHECK: vle.v v29, (t5)
# CHECK-SAME: [0x87,0x7e,0x0f,0x02]
vle.v v29, (t5)
# Encoding: |000|000|0|00000|00001|111|11110|0000111|
# CHECK: vle.v v30, (ra), v0.t
# CHECK-SAME: [0x07,0xff,0x00,0x00]
vle.v v30, (ra), v0.t

# Encoding: |000|000|1|00000|00011|000|11111|0100111|
# CHECK: vsb.v v31, (gp)
# CHECK-SAME: [0xa7,0x8f,0x01,0x02]
vsb.v v31, (gp)
# Encoding: |000|000|0|00000|00101|000|00000|0100111|
# CHECK: vsb.v v0, (t0), v0.t
# CHECK-SAME: [0x27,0x80,0x02,0x00]
vsb.v v0, (t0), v0.t

# Encoding: |000|000|1|00000|00111|101|00001|0100111|
# CHECK: vsh.v v1, (t2)
# CHECK-SAME: [0xa7,0xd0,0x03,0x02]
vsh.v v1, (t2)
# Encoding: |000|000|0|00000|01001|101|00010|0100111|
# CHECK: vsh.v v2, (s1), v0.t
# CHECK-SAME: [0x27,0xd1,0x04,0x00]
vsh.v v2, (s1), v0.t

# Encoding: |000|000|1|00000|01011|110|00011|0100111|
# CHECK: vsw.v v3, (a1)
# CHECK-SAME: [0xa7,0xe1,0x05,0x02]
vsw.v v3, (a1)
# Encoding: |000|000|0|00000|01101|110|00100|0100111|
# CHECK: vsw.v v4, (a3), v0.t
# CHECK-SAME: [0x27,0xe2,0x06,0x00]
vsw.v v4, (a3), v0.t

# Encoding: |000|000|1|00000|01111|111|00101|0100111|
# CHECK: vse.v v5, (a5)
# CHECK-SAME: [0xa7,0xf2,0x07,0x02]
vse.v v5, (a5)
# Encoding: |000|000|0|00000|10001|111|00110|0100111|
# CHECK: vse.v v6, (a7), v0.t
# CHECK-SAME: [0x27,0xf3,0x08,0x00]
vse.v v6, (a7), v0.t

# Encoding: |000|110|1|10011|10101|000|00111|0000111|
# CHECK: vlsb.v v7, (s5), s3
# CHECK-SAME: [0x87,0x83,0x3a,0x1b]
vlsb.v v7, (s5), s3
# Encoding: |000|110|0|10111|11001|000|01000|0000111|
# CHECK: vlsb.v v8, (s9), s7, v0.t
# CHECK-SAME: [0x07,0x84,0x7c,0x19]
vlsb.v v8, (s9), s7, v0.t

# Encoding: |000|110|1|11011|11101|101|01001|0000111|
# CHECK: vlsh.v v9, (t4), s11
# CHECK-SAME: [0x87,0xd4,0xbe,0x1b]
vlsh.v v9, (t4), s11
# Encoding: |000|110|0|11111|00010|101|01010|0000111|
# CHECK: vlsh.v v10, (sp), t6, v0.t
# CHECK-SAME: [0x07,0x55,0xf1,0x19]
vlsh.v v10, (sp), t6, v0.t

# Encoding: |000|110|1|00100|00110|110|01011|0000111|
# CHECK: vlsw.v v11, (t1), tp
# CHECK-SAME: [0x87,0x65,0x43,0x1a]
vlsw.v v11, (t1), tp
# Encoding: |000|110|0|01000|01010|110|01100|0000111|
# CHECK: vlsw.v v12, (a0), s0, v0.t
# CHECK-SAME: [0x07,0x66,0x85,0x18]
vlsw.v v12, (a0), s0, v0.t

# Encoding: |000|010|1|01100|01110|000|01101|0000111|
# CHECK: vlsbu.v v13, (a4), a2
# CHECK-SAME: [0x87,0x06,0xc7,0x0a]
vlsbu.v v13, (a4), a2
# Encoding: |000|010|0|10000|10010|000|01110|0000111|
# CHECK: vlsbu.v v14, (s2), a6, v0.t
# CHECK-SAME: [0x07,0x07,0x09,0x09]
vlsbu.v v14, (s2), a6, v0.t

# Encoding: |000|010|1|10100|10110|101|01111|0000111|
# CHECK: vlshu.v v15, (s6), s4
# CHECK-SAME: [0x87,0x57,0x4b,0x0b]
vlshu.v v15, (s6), s4
# Encoding: |000|010|0|11000|11010|101|10000|0000111|
# CHECK: vlshu.v v16, (s10), s8, v0.t
# CHECK-SAME: [0x07,0x58,0x8d,0x09]
vlshu.v v16, (s10), s8, v0.t

# Encoding: |000|010|1|11100|11110|110|10001|0000111|
# CHECK: vlswu.v v17, (t5), t3
# CHECK-SAME: [0x87,0x68,0xcf,0x0b]
vlswu.v v17, (t5), t3
# Encoding: |000|010|0|00001|00011|110|10010|0000111|
# CHECK: vlswu.v v18, (gp), ra, v0.t
# CHECK-SAME: [0x07,0xe9,0x11,0x08]
vlswu.v v18, (gp), ra, v0.t

# Encoding: |000|010|1|00101|00111|111|10011|0000111|
# CHECK: vlse.v v19, (t2), t0
# CHECK-SAME: [0x87,0xf9,0x53,0x0a]
vlse.v v19, (t2), t0
# Encoding: |000|010|0|01001|01011|111|10100|0000111|
# CHECK: vlse.v v20, (a1), s1, v0.t
# CHECK-SAME: [0x07,0xfa,0x95,0x08]
vlse.v v20, (a1), s1, v0.t

# Encoding: |000|010|1|01101|01111|000|10101|0100111|
# CHECK: vssb.v v21, (a5), a3
# CHECK-SAME: [0xa7,0x8a,0xd7,0x0a]
vssb.v v21, (a5), a3
# Encoding: |000|010|0|10001|10011|000|10110|0100111|
# CHECK: vssb.v v22, (s3), a7, v0.t
# CHECK-SAME: [0x27,0x8b,0x19,0x09]
vssb.v v22, (s3), a7, v0.t

# Encoding: |000|010|1|10101|10111|101|10111|0100111|
# CHECK: vssh.v v23, (s7), s5
# CHECK-SAME: [0xa7,0xdb,0x5b,0x0b]
vssh.v v23, (s7), s5
# Encoding: |000|010|0|11001|11011|101|11000|0100111|
# CHECK: vssh.v v24, (s11), s9, v0.t
# CHECK-SAME: [0x27,0xdc,0x9d,0x09]
vssh.v v24, (s11), s9, v0.t

# Encoding: |000|010|1|11101|11111|110|11001|0100111|
# CHECK: vssw.v v25, (t6), t4
# CHECK-SAME: [0xa7,0xec,0xdf,0x0b]
vssw.v v25, (t6), t4
# Encoding: |000|010|0|00010|00100|110|11010|0100111|
# CHECK: vssw.v v26, (tp), sp, v0.t
# CHECK-SAME: [0x27,0x6d,0x22,0x08]
vssw.v v26, (tp), sp, v0.t

# Encoding: |000|010|1|00110|01000|111|11011|0100111|
# CHECK: vsse.v v27, (s0), t1
# CHECK-SAME: [0xa7,0x7d,0x64,0x0a]
vsse.v v27, (s0), t1
# Encoding: |000|010|0|01010|01100|111|11100|0100111|
# CHECK: vsse.v v28, (a2), a0, v0.t
# CHECK-SAME: [0x27,0x7e,0xa6,0x08]
vsse.v v28, (a2), a0, v0.t

# Encoding: |000|111|1|11101|01110|000|11110|0000111|
# CHECK: vlxb.v v30, (a4), v29
# CHECK-SAME: [0x07,0x0f,0xd7,0x1f]
vlxb.v v30, (a4), v29
# Encoding: |000|111|0|11111|10000|000|00000|0000111|
# CHECK: vlxb.v v0, (a6), v31, v0.t
# CHECK-SAME: [0x07,0x00,0xf8,0x1d]
vlxb.v v0, (a6), v31, v0.t

# Encoding: |000|111|1|00001|10010|101|00010|0000111|
# CHECK: vlxh.v v2, (s2), v1
# CHECK-SAME: [0x07,0x51,0x19,0x1e]
vlxh.v v2, (s2), v1
# Encoding: |000|111|0|00011|10100|101|00100|0000111|
# CHECK: vlxh.v v4, (s4), v3, v0.t
# CHECK-SAME: [0x07,0x52,0x3a,0x1c]
vlxh.v v4, (s4), v3, v0.t

# Encoding: |000|111|1|00101|10110|110|00110|0000111|
# CHECK: vlxw.v v6, (s6), v5
# CHECK-SAME: [0x07,0x63,0x5b,0x1e]
vlxw.v v6, (s6), v5
# Encoding: |000|111|0|00111|11000|110|01000|0000111|
# CHECK: vlxw.v v8, (s8), v7, v0.t
# CHECK-SAME: [0x07,0x64,0x7c,0x1c]
vlxw.v v8, (s8), v7, v0.t

# Encoding: |000|011|1|01001|11010|000|01010|0000111|
# CHECK: vlxbu.v v10, (s10), v9
# CHECK-SAME: [0x07,0x05,0x9d,0x0e]
vlxbu.v v10, (s10), v9
# Encoding: |000|011|0|01011|11100|000|01100|0000111|
# CHECK: vlxbu.v v12, (t3), v11, v0.t
# CHECK-SAME: [0x07,0x06,0xbe,0x0c]
vlxbu.v v12, (t3), v11, v0.t

# Encoding: |000|011|1|01101|11110|101|01110|0000111|
# CHECK: vlxhu.v v14, (t5), v13
# CHECK-SAME: [0x07,0x57,0xdf,0x0e]
vlxhu.v v14, (t5), v13
# Encoding: |000|011|0|01111|00001|101|10000|0000111|
# CHECK: vlxhu.v v16, (ra), v15, v0.t
# CHECK-SAME: [0x07,0xd8,0xf0,0x0c]
vlxhu.v v16, (ra), v15, v0.t

# Encoding: |000|011|1|10001|00011|110|10010|0000111|
# CHECK: vlxwu.v v18, (gp), v17
# CHECK-SAME: [0x07,0xe9,0x11,0x0f]
vlxwu.v v18, (gp), v17
# Encoding: |000|011|0|10011|00101|110|10100|0000111|
# CHECK: vlxwu.v v20, (t0), v19, v0.t
# CHECK-SAME: [0x07,0xea,0x32,0x0d]
vlxwu.v v20, (t0), v19, v0.t

# Encoding: |000|011|1|10101|00111|111|10110|0000111|
# CHECK: vlxe.v v22, (t2), v21
# CHECK-SAME: [0x07,0xfb,0x53,0x0f]
vlxe.v v22, (t2), v21
# Encoding: |000|011|0|10111|01001|111|11000|0000111|
# CHECK: vlxe.v v24, (s1), v23, v0.t
# CHECK-SAME: [0x07,0xfc,0x74,0x0d]
vlxe.v v24, (s1), v23, v0.t

# Encoding: |000|011|1|11001|01011|000|11010|0100111|
# CHECK: vsxb.v v26, (a1), v25
# CHECK-SAME: [0x27,0x8d,0x95,0x0f]
vsxb.v v26, (a1), v25
# Encoding: |000|011|0|11011|01101|000|11100|0100111|
# CHECK: vsxb.v v28, (a3), v27, v0.t
# CHECK-SAME: [0x27,0x8e,0xb6,0x0d]
vsxb.v v28, (a3), v27, v0.t

# Encoding: |000|011|1|11101|01111|101|11110|0100111|
# CHECK: vsxh.v v30, (a5), v29
# CHECK-SAME: [0x27,0xdf,0xd7,0x0f]
vsxh.v v30, (a5), v29
# Encoding: |000|011|0|11111|10001|101|00000|0100111|
# CHECK: vsxh.v v0, (a7), v31, v0.t
# CHECK-SAME: [0x27,0xd0,0xf8,0x0d]
vsxh.v v0, (a7), v31, v0.t

# Encoding: |000|011|1|00001|10011|110|00010|0100111|
# CHECK: vsxw.v v2, (s3), v1
# CHECK-SAME: [0x27,0xe1,0x19,0x0e]
vsxw.v v2, (s3), v1
# Encoding: |000|011|0|00011|10101|110|00100|0100111|
# CHECK: vsxw.v v4, (s5), v3, v0.t
# CHECK-SAME: [0x27,0xe2,0x3a,0x0c]
vsxw.v v4, (s5), v3, v0.t

# Encoding: |000|011|1|00101|10111|111|00110|0100111|
# CHECK: vsxe.v v6, (s7), v5
# CHECK-SAME: [0x27,0xf3,0x5b,0x0e]
vsxe.v v6, (s7), v5
# Encoding: |000|011|0|00111|11001|111|01000|0100111|
# CHECK: vsxe.v v8, (s9), v7, v0.t
# CHECK-SAME: [0x27,0xf4,0x7c,0x0c]
vsxe.v v8, (s9), v7, v0.t

# Encoding: |000|111|1|01001|11011|000|01010|0100111|
# CHECK: vsuxb.v v10, (s11), v9
# CHECK-SAME: [0x27,0x85,0x9d,0x1e]
vsuxb.v v10, (s11), v9
# Encoding: |000|111|0|01011|11101|000|01100|0100111|
# CHECK: vsuxb.v v12, (t4), v11, v0.t
# CHECK-SAME: [0x27,0x86,0xbe,0x1c]
vsuxb.v v12, (t4), v11, v0.t

# Encoding: |000|111|1|01101|11111|101|01110|0100111|
# CHECK: vsuxh.v v14, (t6), v13
# CHECK-SAME: [0x27,0xd7,0xdf,0x1e]
vsuxh.v v14, (t6), v13
# Encoding: |000|111|0|01111|00010|101|10000|0100111|
# CHECK: vsuxh.v v16, (sp), v15, v0.t
# CHECK-SAME: [0x27,0x58,0xf1,0x1c]
vsuxh.v v16, (sp), v15, v0.t

# Encoding: |000|111|1|10001|00100|110|10010|0100111|
# CHECK: vsuxw.v v18, (tp), v17
# CHECK-SAME: [0x27,0x69,0x12,0x1f]
vsuxw.v v18, (tp), v17
# Encoding: |000|111|0|10011|00110|110|10100|0100111|
# CHECK: vsuxw.v v20, (t1), v19, v0.t
# CHECK-SAME: [0x27,0x6a,0x33,0x1d]
vsuxw.v v20, (t1), v19, v0.t

# Encoding: |000|111|1|10101|01000|111|10110|0100111|
# CHECK: vsuxe.v v22, (s0), v21
# CHECK-SAME: [0x27,0x7b,0x54,0x1f]
vsuxe.v v22, (s0), v21
# Encoding: |000|111|0|10111|01010|111|11000|0100111|
# CHECK: vsuxe.v v24, (a0), v23, v0.t
# CHECK-SAME: [0x27,0x7c,0x75,0x1d]
vsuxe.v v24, (a0), v23, v0.t

