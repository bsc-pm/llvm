# Generated witu utils/EPI/process.py
# RUN: llvm-mc < %s -arch=riscv64 -mattr=+m,+f,+d,+a,+epi -show-encoding | FileCheck %s

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

# Encoding: |010000|1|10000|10001|000|10010|1010111|
# CHECK: vadc.vvm v18, v16, v17, v0
# CHECK-SAME: [0x57,0x89,0x08,0x43]
vadc.vvm v18, v16, v17, v0

# Encoding: |010000|1|10011|10110|100|10100|1010111|
# CHECK: vadc.vxm v20, v19, s6, v0
# CHECK-SAME: [0x57,0x4a,0x3b,0x43]
vadc.vxm v20, v19, s6, v0

# Encoding: |010000|1|10101|10000|011|10110|1010111|
# CHECK: vadc.vim v22, v21, -16, v0
# CHECK-SAME: [0x57,0x3b,0x58,0x43]
vadc.vim v22, v21, -16, v0

# Encoding: |010010|1|10111|11000|000|11001|1010111|
# CHECK: vsbc.vvm v25, v23, v24, v0
# CHECK-SAME: [0xd7,0x0c,0x7c,0x4b]
vsbc.vvm v25, v23, v24, v0

# Encoding: |010010|1|11010|11000|100|11011|1010111|
# CHECK: vsbc.vxm v27, v26, s8, v0
# CHECK-SAME: [0xd7,0x4d,0xac,0x4b]
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
# CHECK: vseq.vv v9, v7, v8
# CHECK-SAME: [0xd7,0x04,0x74,0x62]
vseq.vv v9, v7, v8
# Encoding: |011000|0|01010|01011|000|01100|1010111|
# CHECK: vseq.vv v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0x86,0xa5,0x60]
vseq.vv v12, v10, v11, v0.t

# Encoding: |011000|1|01101|11110|100|01110|1010111|
# CHECK: vseq.vx v14, v13, t5
# CHECK-SAME: [0x57,0x47,0xdf,0x62]
vseq.vx v14, v13, t5
# Encoding: |011000|0|01111|00001|100|10000|1010111|
# CHECK: vseq.vx v16, v15, ra, v0.t
# CHECK-SAME: [0x57,0xc8,0xf0,0x60]
vseq.vx v16, v15, ra, v0.t

# Encoding: |011000|1|10001|10011|011|10010|1010111|
# CHECK: vseq.vi v18, v17, -13
# CHECK-SAME: [0x57,0xb9,0x19,0x63]
vseq.vi v18, v17, -13
# Encoding: |011000|0|10011|10100|011|10100|1010111|
# CHECK: vseq.vi v20, v19, -12, v0.t
# CHECK-SAME: [0x57,0x3a,0x3a,0x61]
vseq.vi v20, v19, -12, v0.t

# Encoding: |011001|1|10101|10110|000|10111|1010111|
# CHECK: vsne.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x0b,0x5b,0x67]
vsne.vv v23, v21, v22
# Encoding: |011001|0|11000|11001|000|11010|1010111|
# CHECK: vsne.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0x8d,0x8c,0x65]
vsne.vv v26, v24, v25, v0.t

# Encoding: |011001|1|11011|00011|100|11100|1010111|
# CHECK: vsne.vx v28, v27, gp
# CHECK-SAME: [0x57,0xce,0xb1,0x67]
vsne.vx v28, v27, gp
# Encoding: |011001|0|11101|00101|100|11110|1010111|
# CHECK: vsne.vx v30, v29, t0, v0.t
# CHECK-SAME: [0x57,0xcf,0xd2,0x65]
vsne.vx v30, v29, t0, v0.t

# Encoding: |011001|1|11111|10101|011|00000|1010111|
# CHECK: vsne.vi v0, v31, -11
# CHECK-SAME: [0x57,0xb0,0xfa,0x67]
vsne.vi v0, v31, -11
# Encoding: |011001|0|00001|10110|011|00010|1010111|
# CHECK: vsne.vi v2, v1, -10, v0.t
# CHECK-SAME: [0x57,0x31,0x1b,0x64]
vsne.vi v2, v1, -10, v0.t

# Encoding: |011010|1|00011|00100|000|00101|1010111|
# CHECK: vsltu.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x02,0x32,0x6a]
vsltu.vv v5, v3, v4
# Encoding: |011010|0|00110|00111|000|01000|1010111|
# CHECK: vsltu.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x84,0x63,0x68]
vsltu.vv v8, v6, v7, v0.t

# Encoding: |011010|1|01001|00111|100|01010|1010111|
# CHECK: vsltu.vx v10, v9, t2
# CHECK-SAME: [0x57,0xc5,0x93,0x6a]
vsltu.vx v10, v9, t2
# Encoding: |011010|0|01011|01001|100|01100|1010111|
# CHECK: vsltu.vx v12, v11, s1, v0.t
# CHECK-SAME: [0x57,0xc6,0xb4,0x68]
vsltu.vx v12, v11, s1, v0.t

# Encoding: |011011|1|01101|01110|000|01111|1010111|
# CHECK: vslt.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x07,0xd7,0x6e]
vslt.vv v15, v13, v14
# Encoding: |011011|0|10000|10001|000|10010|1010111|
# CHECK: vslt.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x89,0x08,0x6d]
vslt.vv v18, v16, v17, v0.t

# Encoding: |011011|1|10011|01011|100|10100|1010111|
# CHECK: vslt.vx v20, v19, a1
# CHECK-SAME: [0x57,0xca,0x35,0x6f]
vslt.vx v20, v19, a1
# Encoding: |011011|0|10101|01101|100|10110|1010111|
# CHECK: vslt.vx v22, v21, a3, v0.t
# CHECK-SAME: [0x57,0xcb,0x56,0x6d]
vslt.vx v22, v21, a3, v0.t

# Encoding: |011100|1|10111|11000|000|11001|1010111|
# CHECK: vsleu.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x0c,0x7c,0x73]
vsleu.vv v25, v23, v24
# Encoding: |011100|0|11010|11011|000|11100|1010111|
# CHECK: vsleu.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x8e,0xad,0x71]
vsleu.vv v28, v26, v27, v0.t

# Encoding: |011100|1|11101|01111|100|11110|1010111|
# CHECK: vsleu.vx v30, v29, a5
# CHECK-SAME: [0x57,0xcf,0xd7,0x73]
vsleu.vx v30, v29, a5
# Encoding: |011100|0|11111|10001|100|00000|1010111|
# CHECK: vsleu.vx v0, v31, a7, v0.t
# CHECK-SAME: [0x57,0xc0,0xf8,0x71]
vsleu.vx v0, v31, a7, v0.t

# Encoding: |011100|1|00001|10111|011|00010|1010111|
# CHECK: vsleu.vi v2, v1, -9
# CHECK-SAME: [0x57,0xb1,0x1b,0x72]
vsleu.vi v2, v1, -9
# Encoding: |011100|0|00011|11000|011|00100|1010111|
# CHECK: vsleu.vi v4, v3, -8, v0.t
# CHECK-SAME: [0x57,0x32,0x3c,0x70]
vsleu.vi v4, v3, -8, v0.t

# Encoding: |011101|1|00101|00110|000|00111|1010111|
# CHECK: vsle.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x03,0x53,0x76]
vsle.vv v7, v5, v6
# Encoding: |011101|0|01000|01001|000|01010|1010111|
# CHECK: vsle.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x85,0x84,0x74]
vsle.vv v10, v8, v9, v0.t

# Encoding: |011101|1|01011|10011|100|01100|1010111|
# CHECK: vsle.vx v12, v11, s3
# CHECK-SAME: [0x57,0xc6,0xb9,0x76]
vsle.vx v12, v11, s3
# Encoding: |011101|0|01101|10101|100|01110|1010111|
# CHECK: vsle.vx v14, v13, s5, v0.t
# CHECK-SAME: [0x57,0xc7,0xda,0x74]
vsle.vx v14, v13, s5, v0.t

# Encoding: |011101|1|01111|11001|011|10000|1010111|
# CHECK: vsle.vi v16, v15, -7
# CHECK-SAME: [0x57,0xb8,0xfc,0x76]
vsle.vi v16, v15, -7
# Encoding: |011101|0|10001|11010|011|10010|1010111|
# CHECK: vsle.vi v18, v17, -6, v0.t
# CHECK-SAME: [0x57,0x39,0x1d,0x75]
vsle.vi v18, v17, -6, v0.t

# Encoding: |011110|1|10011|10111|100|10100|1010111|
# CHECK: vsgtu.vx v20, v19, s7
# CHECK-SAME: [0x57,0xca,0x3b,0x7b]
vsgtu.vx v20, v19, s7
# Encoding: |011110|0|10101|11001|100|10110|1010111|
# CHECK: vsgtu.vx v22, v21, s9, v0.t
# CHECK-SAME: [0x57,0xcb,0x5c,0x79]
vsgtu.vx v22, v21, s9, v0.t

# Encoding: |011110|1|10111|11011|011|11000|1010111|
# CHECK: vsgtu.vi v24, v23, 27
# CHECK-SAME: [0x57,0xbc,0x7d,0x7b]
vsgtu.vi v24, v23, 27
# Encoding: |011110|0|11001|11100|011|11010|1010111|
# CHECK: vsgtu.vi v26, v25, 28, v0.t
# CHECK-SAME: [0x57,0x3d,0x9e,0x79]
vsgtu.vi v26, v25, 28, v0.t

# Encoding: |011111|1|11011|11011|100|11100|1010111|
# CHECK: vsgt.vx v28, v27, s11
# CHECK-SAME: [0x57,0xce,0xbd,0x7f]
vsgt.vx v28, v27, s11
# Encoding: |011111|0|11101|11101|100|11110|1010111|
# CHECK: vsgt.vx v30, v29, t4, v0.t
# CHECK-SAME: [0x57,0xcf,0xde,0x7d]
vsgt.vx v30, v29, t4, v0.t

# Encoding: |011111|1|11111|11101|011|00000|1010111|
# CHECK: vsgt.vi v0, v31, -3
# CHECK-SAME: [0x57,0xb0,0xfe,0x7f]
vsgt.vi v0, v31, -3
# Encoding: |011111|0|00001|11110|011|00010|1010111|
# CHECK: vsgt.vi v2, v1, -2, v0.t
# CHECK-SAME: [0x57,0x31,0x1f,0x7c]
vsgt.vi v2, v1, -2, v0.t

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

# Encoding: |100100|1|10011|10100|000|10101|1010111|
# CHECK: vaadd.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x0a,0x3a,0x93]
vaadd.vv v21, v19, v20
# Encoding: |100100|0|10110|10111|000|11000|1010111|
# CHECK: vaadd.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x8c,0x6b,0x91]
vaadd.vv v24, v22, v23, v0.t

# Encoding: |100100|1|11001|10000|100|11010|1010111|
# CHECK: vaadd.vx v26, v25, a6
# CHECK-SAME: [0x57,0x4d,0x98,0x93]
vaadd.vx v26, v25, a6
# Encoding: |100100|0|11011|10010|100|11100|1010111|
# CHECK: vaadd.vx v28, v27, s2, v0.t
# CHECK-SAME: [0x57,0x4e,0xb9,0x91]
vaadd.vx v28, v27, s2, v0.t

# Encoding: |100100|1|11101|00011|011|11110|1010111|
# CHECK: vaadd.vi v30, v29, 3
# CHECK-SAME: [0x57,0xbf,0xd1,0x93]
vaadd.vi v30, v29, 3
# Encoding: |100100|0|11111|00100|011|00000|1010111|
# CHECK: vaadd.vi v0, v31, 4, v0.t
# CHECK-SAME: [0x57,0x30,0xf2,0x91]
vaadd.vi v0, v31, 4, v0.t

# Encoding: |100101|1|00001|00010|000|00011|1010111|
# CHECK: vsll.vv v3, v1, v2
# CHECK-SAME: [0xd7,0x01,0x11,0x96]
vsll.vv v3, v1, v2
# Encoding: |100101|0|00100|00101|000|00110|1010111|
# CHECK: vsll.vv v6, v4, v5, v0.t
# CHECK-SAME: [0x57,0x83,0x42,0x94]
vsll.vv v6, v4, v5, v0.t

# Encoding: |100101|1|00111|10100|100|01000|1010111|
# CHECK: vsll.vx v8, v7, s4
# CHECK-SAME: [0x57,0x44,0x7a,0x96]
vsll.vx v8, v7, s4
# Encoding: |100101|0|01001|10110|100|01010|1010111|
# CHECK: vsll.vx v10, v9, s6, v0.t
# CHECK-SAME: [0x57,0x45,0x9b,0x94]
vsll.vx v10, v9, s6, v0.t

# Encoding: |100101|1|01011|00101|011|01100|1010111|
# CHECK: vsll.vi v12, v11, 5
# CHECK-SAME: [0x57,0xb6,0xb2,0x96]
vsll.vi v12, v11, 5
# Encoding: |100101|0|01101|00110|011|01110|1010111|
# CHECK: vsll.vi v14, v13, 6, v0.t
# CHECK-SAME: [0x57,0x37,0xd3,0x94]
vsll.vi v14, v13, 6, v0.t

# Encoding: |100110|1|01111|10000|000|10001|1010111|
# CHECK: vasub.vv v17, v15, v16
# CHECK-SAME: [0xd7,0x08,0xf8,0x9a]
vasub.vv v17, v15, v16
# Encoding: |100110|0|10010|10011|000|10100|1010111|
# CHECK: vasub.vv v20, v18, v19, v0.t
# CHECK-SAME: [0x57,0x8a,0x29,0x99]
vasub.vv v20, v18, v19, v0.t

# Encoding: |100110|1|10101|11000|100|10110|1010111|
# CHECK: vasub.vx v22, v21, s8
# CHECK-SAME: [0x57,0x4b,0x5c,0x9b]
vasub.vx v22, v21, s8
# Encoding: |100110|0|10111|11010|100|11000|1010111|
# CHECK: vasub.vx v24, v23, s10, v0.t
# CHECK-SAME: [0x57,0x4c,0x7d,0x99]
vasub.vx v24, v23, s10, v0.t

# Encoding: |100111|1|11001|11010|000|11011|1010111|
# CHECK: vsmul.vv v27, v25, v26
# CHECK-SAME: [0xd7,0x0d,0x9d,0x9f]
vsmul.vv v27, v25, v26
# Encoding: |100111|0|11100|11101|000|11110|1010111|
# CHECK: vsmul.vv v30, v28, v29, v0.t
# CHECK-SAME: [0x57,0x8f,0xce,0x9d]
vsmul.vv v30, v28, v29, v0.t

# Encoding: |100111|1|11111|11100|100|00000|1010111|
# CHECK: vsmul.vx v0, v31, t3
# CHECK-SAME: [0x57,0x40,0xfe,0x9f]
vsmul.vx v0, v31, t3
# Encoding: |100111|0|00001|11110|100|00010|1010111|
# CHECK: vsmul.vx v2, v1, t5, v0.t
# CHECK-SAME: [0x57,0x41,0x1f,0x9c]
vsmul.vx v2, v1, t5, v0.t

# Encoding: |101000|1|00011|00100|000|00101|1010111|
# CHECK: vsrl.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x02,0x32,0xa2]
vsrl.vv v5, v3, v4
# Encoding: |101000|0|00110|00111|000|01000|1010111|
# CHECK: vsrl.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x84,0x63,0xa0]
vsrl.vv v8, v6, v7, v0.t

# Encoding: |101000|1|01001|00001|100|01010|1010111|
# CHECK: vsrl.vx v10, v9, ra
# CHECK-SAME: [0x57,0xc5,0x90,0xa2]
vsrl.vx v10, v9, ra
# Encoding: |101000|0|01011|00011|100|01100|1010111|
# CHECK: vsrl.vx v12, v11, gp, v0.t
# CHECK-SAME: [0x57,0xc6,0xb1,0xa0]
vsrl.vx v12, v11, gp, v0.t

# Encoding: |101000|1|01101|00111|011|01110|1010111|
# CHECK: vsrl.vi v14, v13, 7
# CHECK-SAME: [0x57,0xb7,0xd3,0xa2]
vsrl.vi v14, v13, 7
# Encoding: |101000|0|01111|01000|011|10000|1010111|
# CHECK: vsrl.vi v16, v15, 8, v0.t
# CHECK-SAME: [0x57,0x38,0xf4,0xa0]
vsrl.vi v16, v15, 8, v0.t

# Encoding: |101001|1|10001|10010|000|10011|1010111|
# CHECK: vsra.vv v19, v17, v18
# CHECK-SAME: [0xd7,0x09,0x19,0xa7]
vsra.vv v19, v17, v18
# Encoding: |101001|0|10100|10101|000|10110|1010111|
# CHECK: vsra.vv v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0x8b,0x4a,0xa5]
vsra.vv v22, v20, v21, v0.t

# Encoding: |101001|1|10111|00101|100|11000|1010111|
# CHECK: vsra.vx v24, v23, t0
# CHECK-SAME: [0x57,0xcc,0x72,0xa7]
vsra.vx v24, v23, t0
# Encoding: |101001|0|11001|00111|100|11010|1010111|
# CHECK: vsra.vx v26, v25, t2, v0.t
# CHECK-SAME: [0x57,0xcd,0x93,0xa5]
vsra.vx v26, v25, t2, v0.t

# Encoding: |101001|1|11011|01001|011|11100|1010111|
# CHECK: vsra.vi v28, v27, 9
# CHECK-SAME: [0x57,0xbe,0xb4,0xa7]
vsra.vi v28, v27, 9
# Encoding: |101001|0|11101|01010|011|11110|1010111|
# CHECK: vsra.vi v30, v29, 10, v0.t
# CHECK-SAME: [0x57,0x3f,0xd5,0xa5]
vsra.vi v30, v29, 10, v0.t

# Encoding: |101010|1|11111|00000|000|00001|1010111|
# CHECK: vssrl.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x00,0xf0,0xab]
vssrl.vv v1, v31, v0
# Encoding: |101010|0|00010|00011|000|00100|1010111|
# CHECK: vssrl.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x82,0x21,0xa8]
vssrl.vv v4, v2, v3, v0.t

# Encoding: |101010|1|00101|01001|100|00110|1010111|
# CHECK: vssrl.vx v6, v5, s1
# CHECK-SAME: [0x57,0xc3,0x54,0xaa]
vssrl.vx v6, v5, s1
# Encoding: |101010|0|00111|01011|100|01000|1010111|
# CHECK: vssrl.vx v8, v7, a1, v0.t
# CHECK-SAME: [0x57,0xc4,0x75,0xa8]
vssrl.vx v8, v7, a1, v0.t

# Encoding: |101010|1|01001|01011|011|01010|1010111|
# CHECK: vssrl.vi v10, v9, 11
# CHECK-SAME: [0x57,0xb5,0x95,0xaa]
vssrl.vi v10, v9, 11
# Encoding: |101010|0|01011|01100|011|01100|1010111|
# CHECK: vssrl.vi v12, v11, 12, v0.t
# CHECK-SAME: [0x57,0x36,0xb6,0xa8]
vssrl.vi v12, v11, 12, v0.t

# Encoding: |101011|1|01101|01110|000|01111|1010111|
# CHECK: vssra.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x07,0xd7,0xae]
vssra.vv v15, v13, v14
# Encoding: |101011|0|10000|10001|000|10010|1010111|
# CHECK: vssra.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x89,0x08,0xad]
vssra.vv v18, v16, v17, v0.t

# Encoding: |101011|1|10011|01101|100|10100|1010111|
# CHECK: vssra.vx v20, v19, a3
# CHECK-SAME: [0x57,0xca,0x36,0xaf]
vssra.vx v20, v19, a3
# Encoding: |101011|0|10101|01111|100|10110|1010111|
# CHECK: vssra.vx v22, v21, a5, v0.t
# CHECK-SAME: [0x57,0xcb,0x57,0xad]
vssra.vx v22, v21, a5, v0.t

# Encoding: |101011|1|10111|01101|011|11000|1010111|
# CHECK: vssra.vi v24, v23, 13
# CHECK-SAME: [0x57,0xbc,0x76,0xaf]
vssra.vi v24, v23, 13
# Encoding: |101011|0|11001|01110|011|11010|1010111|
# CHECK: vssra.vi v26, v25, 14, v0.t
# CHECK-SAME: [0x57,0x3d,0x97,0xad]
vssra.vi v26, v25, 14, v0.t

# Encoding: |101100|1|11011|11100|000|11101|1010111|
# CHECK: vnsrl.vv v29, v27, v28
# CHECK-SAME: [0xd7,0x0e,0xbe,0xb3]
vnsrl.vv v29, v27, v28
# Encoding: |101100|0|11110|11111|000|00000|1010111|
# CHECK: vnsrl.vv v0, v30, v31, v0.t
# CHECK-SAME: [0x57,0x80,0xef,0xb1]
vnsrl.vv v0, v30, v31, v0.t

# Encoding: |101100|1|00001|10001|100|00010|1010111|
# CHECK: vnsrl.vx v2, v1, a7
# CHECK-SAME: [0x57,0xc1,0x18,0xb2]
vnsrl.vx v2, v1, a7
# Encoding: |101100|0|00011|10011|100|00100|1010111|
# CHECK: vnsrl.vx v4, v3, s3, v0.t
# CHECK-SAME: [0x57,0xc2,0x39,0xb0]
vnsrl.vx v4, v3, s3, v0.t

# Encoding: |101100|1|00101|01111|011|00110|1010111|
# CHECK: vnsrl.vi v6, v5, 15
# CHECK-SAME: [0x57,0xb3,0x57,0xb2]
vnsrl.vi v6, v5, 15
# Encoding: |101100|0|00111|10000|011|01000|1010111|
# CHECK: vnsrl.vi v8, v7, -16, v0.t
# CHECK-SAME: [0x57,0x34,0x78,0xb0]
vnsrl.vi v8, v7, -16, v0.t

# Encoding: |101101|1|01001|01010|000|01011|1010111|
# CHECK: vnsra.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x05,0x95,0xb6]
vnsra.vv v11, v9, v10
# Encoding: |101101|0|01100|01101|000|01110|1010111|
# CHECK: vnsra.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x87,0xc6,0xb4]
vnsra.vv v14, v12, v13, v0.t

# Encoding: |101101|1|01111|10101|100|10000|1010111|
# CHECK: vnsra.vx v16, v15, s5
# CHECK-SAME: [0x57,0xc8,0xfa,0xb6]
vnsra.vx v16, v15, s5
# Encoding: |101101|0|10001|10111|100|10010|1010111|
# CHECK: vnsra.vx v18, v17, s7, v0.t
# CHECK-SAME: [0x57,0xc9,0x1b,0xb5]
vnsra.vx v18, v17, s7, v0.t

# Encoding: |101101|1|10011|10001|011|10100|1010111|
# CHECK: vnsra.vi v20, v19, -15
# CHECK-SAME: [0x57,0xba,0x38,0xb7]
vnsra.vi v20, v19, -15
# Encoding: |101101|0|10101|10010|011|10110|1010111|
# CHECK: vnsra.vi v22, v21, -14, v0.t
# CHECK-SAME: [0x57,0x3b,0x59,0xb5]
vnsra.vi v22, v21, -14, v0.t

# Encoding: |101110|1|10111|11000|000|11001|1010111|
# CHECK: vnclipu.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x0c,0x7c,0xbb]
vnclipu.vv v25, v23, v24
# Encoding: |101110|0|11010|11011|000|11100|1010111|
# CHECK: vnclipu.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x8e,0xad,0xb9]
vnclipu.vv v28, v26, v27, v0.t

# Encoding: |101110|1|11101|11001|100|11110|1010111|
# CHECK: vnclipu.vx v30, v29, s9
# CHECK-SAME: [0x57,0xcf,0xdc,0xbb]
vnclipu.vx v30, v29, s9
# Encoding: |101110|0|11111|11011|100|00000|1010111|
# CHECK: vnclipu.vx v0, v31, s11, v0.t
# CHECK-SAME: [0x57,0xc0,0xfd,0xb9]
vnclipu.vx v0, v31, s11, v0.t

# Encoding: |101110|1|00001|10011|011|00010|1010111|
# CHECK: vnclipu.vi v2, v1, -13
# CHECK-SAME: [0x57,0xb1,0x19,0xba]
vnclipu.vi v2, v1, -13
# Encoding: |101110|0|00011|10100|011|00100|1010111|
# CHECK: vnclipu.vi v4, v3, -12, v0.t
# CHECK-SAME: [0x57,0x32,0x3a,0xb8]
vnclipu.vi v4, v3, -12, v0.t

# Encoding: |101111|1|00101|00110|000|00111|1010111|
# CHECK: vnclip.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x03,0x53,0xbe]
vnclip.vv v7, v5, v6
# Encoding: |101111|0|01000|01001|000|01010|1010111|
# CHECK: vnclip.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x85,0x84,0xbc]
vnclip.vv v10, v8, v9, v0.t

# Encoding: |101111|1|01011|11101|100|01100|1010111|
# CHECK: vnclip.vx v12, v11, t4
# CHECK-SAME: [0x57,0xc6,0xbe,0xbe]
vnclip.vx v12, v11, t4
# Encoding: |101111|0|01101|11111|100|01110|1010111|
# CHECK: vnclip.vx v14, v13, t6, v0.t
# CHECK-SAME: [0x57,0xc7,0xdf,0xbc]
vnclip.vx v14, v13, t6, v0.t

# Encoding: |101111|1|01111|10101|011|10000|1010111|
# CHECK: vnclip.vi v16, v15, -11
# CHECK-SAME: [0x57,0xb8,0xfa,0xbe]
vnclip.vi v16, v15, -11
# Encoding: |101111|0|10001|10110|011|10010|1010111|
# CHECK: vnclip.vi v18, v17, -10, v0.t
# CHECK-SAME: [0x57,0x39,0x1b,0xbd]
vnclip.vi v18, v17, -10, v0.t

# Encoding: |110000|1|10011|10100|000|10101|1010111|
# CHECK: vwredsumu.vs v21, v19, v20
# CHECK-SAME: [0xd7,0x0a,0x3a,0xc3]
vwredsumu.vs v21, v19, v20
# Encoding: |110000|0|10110|10111|000|11000|1010111|
# CHECK: vwredsumu.vs v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x8c,0x6b,0xc1]
vwredsumu.vs v24, v22, v23, v0.t

# Encoding: |110001|1|11001|11010|000|11011|1010111|
# CHECK: vwredsum.vs v27, v25, v26
# CHECK-SAME: [0xd7,0x0d,0x9d,0xc7]
vwredsum.vs v27, v25, v26
# Encoding: |110001|0|11100|11101|000|11110|1010111|
# CHECK: vwredsum.vs v30, v28, v29, v0.t
# CHECK-SAME: [0x57,0x8f,0xce,0xc5]
vwredsum.vs v30, v28, v29, v0.t

# Encoding: |111000|1|11111|00000|000|00001|1010111|
# CHECK: vdotu.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x00,0xf0,0xe3]
vdotu.vv v1, v31, v0
# Encoding: |111000|0|00010|00011|000|00100|1010111|
# CHECK: vdotu.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x82,0x21,0xe0]
vdotu.vv v4, v2, v3, v0.t

# Encoding: |111001|1|00101|00110|000|00111|1010111|
# CHECK: vdot.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x03,0x53,0xe6]
vdot.vv v7, v5, v6
# Encoding: |111001|0|01000|01001|000|01010|1010111|
# CHECK: vdot.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x85,0x84,0xe4]
vdot.vv v10, v8, v9, v0.t

# Encoding: |111100|1|01011|01100|000|01101|1010111|
# CHECK: vwsmaccu.vv v13, v11, v12
# CHECK-SAME: [0xd7,0x06,0xb6,0xf2]
vwsmaccu.vv v13, v11, v12
# Encoding: |111100|0|01110|01111|000|10000|1010111|
# CHECK: vwsmaccu.vv v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0x88,0xe7,0xf0]
vwsmaccu.vv v16, v14, v15, v0.t

# Encoding: |111100|1|10001|00010|100|10010|1010111|
# CHECK: vwsmaccu.vx v18, v17, sp
# CHECK-SAME: [0x57,0x49,0x11,0xf3]
vwsmaccu.vx v18, v17, sp
# Encoding: |111100|0|10011|00100|100|10100|1010111|
# CHECK: vwsmaccu.vx v20, v19, tp, v0.t
# CHECK-SAME: [0x57,0x4a,0x32,0xf1]
vwsmaccu.vx v20, v19, tp, v0.t

# Encoding: |111101|1|10101|10110|000|10111|1010111|
# CHECK: vwsmacc.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x0b,0x5b,0xf7]
vwsmacc.vv v23, v21, v22
# Encoding: |111101|0|11000|11001|000|11010|1010111|
# CHECK: vwsmacc.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0x8d,0x8c,0xf5]
vwsmacc.vv v26, v24, v25, v0.t

# Encoding: |111101|1|11011|00110|100|11100|1010111|
# CHECK: vwsmacc.vx v28, v27, t1
# CHECK-SAME: [0x57,0x4e,0xb3,0xf7]
vwsmacc.vx v28, v27, t1
# Encoding: |111101|0|11101|01000|100|11110|1010111|
# CHECK: vwsmacc.vx v30, v29, s0, v0.t
# CHECK-SAME: [0x57,0x4f,0xd4,0xf5]
vwsmacc.vx v30, v29, s0, v0.t

# Encoding: |111110|1|11111|00000|000|00001|1010111|
# CHECK: vwsmsacu.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x00,0xf0,0xfb]
vwsmsacu.vv v1, v31, v0
# Encoding: |111110|0|00010|00011|000|00100|1010111|
# CHECK: vwsmsacu.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x82,0x21,0xf8]
vwsmsacu.vv v4, v2, v3, v0.t

# Encoding: |111110|1|00101|01010|100|00110|1010111|
# CHECK: vwsmsacu.vx v6, v5, a0
# CHECK-SAME: [0x57,0x43,0x55,0xfa]
vwsmsacu.vx v6, v5, a0
# Encoding: |111110|0|00111|01100|100|01000|1010111|
# CHECK: vwsmsacu.vx v8, v7, a2, v0.t
# CHECK-SAME: [0x57,0x44,0x76,0xf8]
vwsmsacu.vx v8, v7, a2, v0.t

# Encoding: |111111|1|01001|01010|000|01011|1010111|
# CHECK: vwsmsac.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x05,0x95,0xfe]
vwsmsac.vv v11, v9, v10
# Encoding: |111111|0|01100|01101|000|01110|1010111|
# CHECK: vwsmsac.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x87,0xc6,0xfc]
vwsmsac.vv v14, v12, v13, v0.t

# Encoding: |111111|1|01111|01110|100|10000|1010111|
# CHECK: vwsmsac.vx v16, v15, a4
# CHECK-SAME: [0x57,0x48,0xf7,0xfe]
vwsmsac.vx v16, v15, a4
# Encoding: |111111|0|10001|10000|100|10010|1010111|
# CHECK: vwsmsac.vx v18, v17, a6, v0.t
# CHECK-SAME: [0x57,0x49,0x18,0xfd]
vwsmsac.vx v18, v17, a6, v0.t

# Encoding: |000000|1|10011|10100|010|10101|1010111|
# CHECK: vredsum.vs v21, v19, v20
# CHECK-SAME: [0xd7,0x2a,0x3a,0x03]
vredsum.vs v21, v19, v20
# Encoding: |000000|0|10110|10111|010|11000|1010111|
# CHECK: vredsum.vs v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0xac,0x6b,0x01]
vredsum.vs v24, v22, v23, v0.t

# Encoding: |000001|1|11001|11010|010|11011|1010111|
# CHECK: vredand.vs v27, v25, v26
# CHECK-SAME: [0xd7,0x2d,0x9d,0x07]
vredand.vs v27, v25, v26
# Encoding: |000001|0|11100|11101|010|11110|1010111|
# CHECK: vredand.vs v30, v28, v29, v0.t
# CHECK-SAME: [0x57,0xaf,0xce,0x05]
vredand.vs v30, v28, v29, v0.t

# Encoding: |000010|1|11111|00000|010|00001|1010111|
# CHECK: vredor.vs v1, v31, v0
# CHECK-SAME: [0xd7,0x20,0xf0,0x0b]
vredor.vs v1, v31, v0
# Encoding: |000010|0|00010|00011|010|00100|1010111|
# CHECK: vredor.vs v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0xa2,0x21,0x08]
vredor.vs v4, v2, v3, v0.t

# Encoding: |000011|1|00101|00110|010|00111|1010111|
# CHECK: vredxor.vs v7, v5, v6
# CHECK-SAME: [0xd7,0x23,0x53,0x0e]
vredxor.vs v7, v5, v6
# Encoding: |000011|0|01000|01001|010|01010|1010111|
# CHECK: vredxor.vs v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0xa5,0x84,0x0c]
vredxor.vs v10, v8, v9, v0.t

# Encoding: |000100|1|01011|01100|010|01101|1010111|
# CHECK: vredminu.vs v13, v11, v12
# CHECK-SAME: [0xd7,0x26,0xb6,0x12]
vredminu.vs v13, v11, v12
# Encoding: |000100|0|01110|01111|010|10000|1010111|
# CHECK: vredminu.vs v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0xa8,0xe7,0x10]
vredminu.vs v16, v14, v15, v0.t

# Encoding: |000101|1|10001|10010|010|10011|1010111|
# CHECK: vredmin.vs v19, v17, v18
# CHECK-SAME: [0xd7,0x29,0x19,0x17]
vredmin.vs v19, v17, v18
# Encoding: |000101|0|10100|10101|010|10110|1010111|
# CHECK: vredmin.vs v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0xab,0x4a,0x15]
vredmin.vs v22, v20, v21, v0.t

# Encoding: |000110|1|10111|11000|010|11001|1010111|
# CHECK: vredmaxu.vs v25, v23, v24
# CHECK-SAME: [0xd7,0x2c,0x7c,0x1b]
vredmaxu.vs v25, v23, v24
# Encoding: |000110|0|11010|11011|010|11100|1010111|
# CHECK: vredmaxu.vs v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0xae,0xad,0x19]
vredmaxu.vs v28, v26, v27, v0.t

# Encoding: |000111|1|11101|11110|010|11111|1010111|
# CHECK: vredmax.vs v31, v29, v30
# CHECK-SAME: [0xd7,0x2f,0xdf,0x1f]
vredmax.vs v31, v29, v30
# Encoding: |000111|0|00000|00001|010|00010|1010111|
# CHECK: vredmax.vs v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0xa1,0x00,0x1c]
vredmax.vs v2, v0, v1, v0.t

# Encoding: |001100|1|00011|10010|010|10100|1010111|
# CHECK: vext.x.v s4, v3, s2
# CHECK-SAME: [0x57,0x2a,0x39,0x32]
vext.x.v s4, v3, s2

# Encoding: |001101|1|00000|10110|110|00100|1010111|
# CHECK: vmv.s.x v4, s6
# CHECK-SAME: [0x57,0x62,0x0b,0x36]
vmv.s.x v4, s6

# Encoding: |001110|1|00101|11000|110|00110|1010111|
# CHECK: vslide1up.vx v6, v5, s8
# CHECK-SAME: [0x57,0x63,0x5c,0x3a]
vslide1up.vx v6, v5, s8
# Encoding: |001110|0|00111|11010|110|01000|1010111|
# CHECK: vslide1up.vx v8, v7, s10, v0.t
# CHECK-SAME: [0x57,0x64,0x7d,0x38]
vslide1up.vx v8, v7, s10, v0.t

# Encoding: |001111|1|01001|11100|110|01010|1010111|
# CHECK: vslide1down.vx v10, v9, t3
# CHECK-SAME: [0x57,0x65,0x9e,0x3e]
vslide1down.vx v10, v9, t3
# Encoding: |001111|0|01011|11110|110|01100|1010111|
# CHECK: vslide1down.vx v12, v11, t5, v0.t
# CHECK-SAME: [0x57,0x66,0xbf,0x3c]
vslide1down.vx v12, v11, t5, v0.t

# Encoding: |010100|1|01101|00000|010|00001|1010111|
# CHECK: vmpopc.m ra, v13
# CHECK-SAME: [0xd7,0x20,0xd0,0x52]
vmpopc.m ra, v13
# Encoding: |010100|0|01110|00000|010|00011|1010111|
# CHECK: vmpopc.m gp, v14, v0.t
# CHECK-SAME: [0xd7,0x21,0xe0,0x50]
vmpopc.m gp, v14, v0.t

# Encoding: |010101|1|01111|00000|010|00101|1010111|
# CHECK: vmfirst.m t0, v15
# CHECK-SAME: [0xd7,0x22,0xf0,0x56]
vmfirst.m t0, v15
# Encoding: |010101|0|10000|00000|010|00111|1010111|
# CHECK: vmfirst.m t2, v16, v0.t
# CHECK-SAME: [0xd7,0x23,0x00,0x55]
vmfirst.m t2, v16, v0.t

# Encoding: |010111|1|10001|10010|010|10011|1010111|
# CHECK: vcompress.vm v19, v17, v18
# CHECK-SAME: [0xd7,0x29,0x19,0x5f]
vcompress.vm v19, v17, v18

# Encoding: |011000|1|10100|10101|010|10110|1010111|
# CHECK: vmandnot.mm v22, v20, v21
# CHECK-SAME: [0x57,0xab,0x4a,0x63]
vmandnot.mm v22, v20, v21

# Encoding: |011001|1|10111|11000|010|11001|1010111|
# CHECK: vmand.mm v25, v23, v24
# CHECK-SAME: [0xd7,0x2c,0x7c,0x67]
vmand.mm v25, v23, v24

# Encoding: |011010|1|11010|11011|010|11100|1010111|
# CHECK: vmor.mm v28, v26, v27
# CHECK-SAME: [0x57,0xae,0xad,0x6b]
vmor.mm v28, v26, v27

# Encoding: |011011|1|11101|11110|010|11111|1010111|
# CHECK: vmxor.mm v31, v29, v30
# CHECK-SAME: [0xd7,0x2f,0xdf,0x6f]
vmxor.mm v31, v29, v30

# Encoding: |011100|1|00000|00001|010|00010|1010111|
# CHECK: vmornot.mm v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0x72]
vmornot.mm v2, v0, v1

# Encoding: |011101|1|00011|00100|010|00101|1010111|
# CHECK: vmnand.mm v5, v3, v4
# CHECK-SAME: [0xd7,0x22,0x32,0x76]
vmnand.mm v5, v3, v4

# Encoding: |011110|1|00110|00111|010|01000|1010111|
# CHECK: vmnor.mm v8, v6, v7
# CHECK-SAME: [0x57,0xa4,0x63,0x7a]
vmnor.mm v8, v6, v7

# Encoding: |011111|1|01001|01010|010|01011|1010111|
# CHECK: vmxnor.mm v11, v9, v10
# CHECK-SAME: [0xd7,0x25,0x95,0x7e]
vmxnor.mm v11, v9, v10

# Encoding: |100000|1|01100|01101|010|01110|1010111|
# CHECK: vdivu.vv v14, v12, v13
# CHECK-SAME: [0x57,0xa7,0xc6,0x82]
vdivu.vv v14, v12, v13
# Encoding: |100000|0|01111|10000|010|10001|1010111|
# CHECK: vdivu.vv v17, v15, v16, v0.t
# CHECK-SAME: [0xd7,0x28,0xf8,0x80]
vdivu.vv v17, v15, v16, v0.t

# Encoding: |100000|1|10010|01001|110|10011|1010111|
# CHECK: vdivu.vx v19, v18, s1
# CHECK-SAME: [0xd7,0xe9,0x24,0x83]
vdivu.vx v19, v18, s1
# Encoding: |100000|0|10100|01011|110|10101|1010111|
# CHECK: vdivu.vx v21, v20, a1, v0.t
# CHECK-SAME: [0xd7,0xea,0x45,0x81]
vdivu.vx v21, v20, a1, v0.t

# Encoding: |100001|1|10110|10111|010|11000|1010111|
# CHECK: vdiv.vv v24, v22, v23
# CHECK-SAME: [0x57,0xac,0x6b,0x87]
vdiv.vv v24, v22, v23
# Encoding: |100001|0|11001|11010|010|11011|1010111|
# CHECK: vdiv.vv v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x2d,0x9d,0x85]
vdiv.vv v27, v25, v26, v0.t

# Encoding: |100001|1|11100|01101|110|11101|1010111|
# CHECK: vdiv.vx v29, v28, a3
# CHECK-SAME: [0xd7,0xee,0xc6,0x87]
vdiv.vx v29, v28, a3
# Encoding: |100001|0|11110|01111|110|11111|1010111|
# CHECK: vdiv.vx v31, v30, a5, v0.t
# CHECK-SAME: [0xd7,0xef,0xe7,0x85]
vdiv.vx v31, v30, a5, v0.t

# Encoding: |100010|1|00000|00001|010|00010|1010111|
# CHECK: vremu.vv v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0x8a]
vremu.vv v2, v0, v1
# Encoding: |100010|0|00011|00100|010|00101|1010111|
# CHECK: vremu.vv v5, v3, v4, v0.t
# CHECK-SAME: [0xd7,0x22,0x32,0x88]
vremu.vv v5, v3, v4, v0.t

# Encoding: |100010|1|00110|10001|110|00111|1010111|
# CHECK: vremu.vx v7, v6, a7
# CHECK-SAME: [0xd7,0xe3,0x68,0x8a]
vremu.vx v7, v6, a7
# Encoding: |100010|0|01000|10011|110|01001|1010111|
# CHECK: vremu.vx v9, v8, s3, v0.t
# CHECK-SAME: [0xd7,0xe4,0x89,0x88]
vremu.vx v9, v8, s3, v0.t

# Encoding: |100011|1|01010|01011|010|01100|1010111|
# CHECK: vrem.vv v12, v10, v11
# CHECK-SAME: [0x57,0xa6,0xa5,0x8e]
vrem.vv v12, v10, v11
# Encoding: |100011|0|01101|01110|010|01111|1010111|
# CHECK: vrem.vv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x27,0xd7,0x8c]
vrem.vv v15, v13, v14, v0.t

# Encoding: |100011|1|10000|10101|110|10001|1010111|
# CHECK: vrem.vx v17, v16, s5
# CHECK-SAME: [0xd7,0xe8,0x0a,0x8f]
vrem.vx v17, v16, s5
# Encoding: |100011|0|10010|10111|110|10011|1010111|
# CHECK: vrem.vx v19, v18, s7, v0.t
# CHECK-SAME: [0xd7,0xe9,0x2b,0x8d]
vrem.vx v19, v18, s7, v0.t

# Encoding: |100100|1|10100|10101|010|10110|1010111|
# CHECK: vmulhu.vv v22, v20, v21
# CHECK-SAME: [0x57,0xab,0x4a,0x93]
vmulhu.vv v22, v20, v21
# Encoding: |100100|0|10111|11000|010|11001|1010111|
# CHECK: vmulhu.vv v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x2c,0x7c,0x91]
vmulhu.vv v25, v23, v24, v0.t

# Encoding: |100100|1|11010|11001|110|11011|1010111|
# CHECK: vmulhu.vx v27, v26, s9
# CHECK-SAME: [0xd7,0xed,0xac,0x93]
vmulhu.vx v27, v26, s9
# Encoding: |100100|0|11100|11011|110|11101|1010111|
# CHECK: vmulhu.vx v29, v28, s11, v0.t
# CHECK-SAME: [0xd7,0xee,0xcd,0x91]
vmulhu.vx v29, v28, s11, v0.t

# Encoding: |100101|1|11110|11111|010|00000|1010111|
# CHECK: vmul.vv v0, v30, v31
# CHECK-SAME: [0x57,0xa0,0xef,0x97]
vmul.vv v0, v30, v31
# Encoding: |100101|0|00001|00010|010|00011|1010111|
# CHECK: vmul.vv v3, v1, v2, v0.t
# CHECK-SAME: [0xd7,0x21,0x11,0x94]
vmul.vv v3, v1, v2, v0.t

# Encoding: |100101|1|00100|11101|110|00101|1010111|
# CHECK: vmul.vx v5, v4, t4
# CHECK-SAME: [0xd7,0xe2,0x4e,0x96]
vmul.vx v5, v4, t4
# Encoding: |100101|0|00110|11111|110|00111|1010111|
# CHECK: vmul.vx v7, v6, t6, v0.t
# CHECK-SAME: [0xd7,0xe3,0x6f,0x94]
vmul.vx v7, v6, t6, v0.t

# Encoding: |100110|1|01000|01001|010|01010|1010111|
# CHECK: vmulhsu.vv v10, v8, v9
# CHECK-SAME: [0x57,0xa5,0x84,0x9a]
vmulhsu.vv v10, v8, v9
# Encoding: |100110|0|01011|01100|010|01101|1010111|
# CHECK: vmulhsu.vv v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x26,0xb6,0x98]
vmulhsu.vv v13, v11, v12, v0.t

# Encoding: |100110|1|01110|00010|110|01111|1010111|
# CHECK: vmulhsu.vx v15, v14, sp
# CHECK-SAME: [0xd7,0x67,0xe1,0x9a]
vmulhsu.vx v15, v14, sp
# Encoding: |100110|0|10000|00100|110|10001|1010111|
# CHECK: vmulhsu.vx v17, v16, tp, v0.t
# CHECK-SAME: [0xd7,0x68,0x02,0x99]
vmulhsu.vx v17, v16, tp, v0.t

# Encoding: |100111|1|10010|10011|010|10100|1010111|
# CHECK: vmulh.vv v20, v18, v19
# CHECK-SAME: [0x57,0xaa,0x29,0x9f]
vmulh.vv v20, v18, v19
# Encoding: |100111|0|10101|10110|010|10111|1010111|
# CHECK: vmulh.vv v23, v21, v22, v0.t
# CHECK-SAME: [0xd7,0x2b,0x5b,0x9d]
vmulh.vv v23, v21, v22, v0.t

# Encoding: |100111|1|11000|00110|110|11001|1010111|
# CHECK: vmulh.vx v25, v24, t1
# CHECK-SAME: [0xd7,0x6c,0x83,0x9f]
vmulh.vx v25, v24, t1
# Encoding: |100111|0|11010|01000|110|11011|1010111|
# CHECK: vmulh.vx v27, v26, s0, v0.t
# CHECK-SAME: [0xd7,0x6d,0xa4,0x9d]
vmulh.vx v27, v26, s0, v0.t

# Encoding: |101001|1|11100|11101|010|11110|1010111|
# CHECK: vmadd.vv v30, v29, v28
# CHECK-SAME: [0x57,0xaf,0xce,0xa7]
vmadd.vv v30, v29, v28
# Encoding: |101001|0|11111|00000|010|00001|1010111|
# CHECK: vmadd.vv v1, v0, v31, v0.t
# CHECK-SAME: [0xd7,0x20,0xf0,0xa5]
vmadd.vv v1, v0, v31, v0.t

# Encoding: |101001|1|00010|01010|110|00011|1010111|
# CHECK: vmadd.vx v3, a0, v2
# CHECK-SAME: [0xd7,0x61,0x25,0xa6]
vmadd.vx v3, a0, v2
# Encoding: |101001|0|00100|01100|110|00101|1010111|
# CHECK: vmadd.vx v5, a2, v4, v0.t
# CHECK-SAME: [0xd7,0x62,0x46,0xa4]
vmadd.vx v5, a2, v4, v0.t

# Encoding: |101011|1|00110|00111|010|01000|1010111|
# CHECK: vmsub.vv v8, v7, v6
# CHECK-SAME: [0x57,0xa4,0x63,0xae]
vmsub.vv v8, v7, v6
# Encoding: |101011|0|01001|01010|010|01011|1010111|
# CHECK: vmsub.vv v11, v10, v9, v0.t
# CHECK-SAME: [0xd7,0x25,0x95,0xac]
vmsub.vv v11, v10, v9, v0.t

# Encoding: |101011|1|01100|01110|110|01101|1010111|
# CHECK: vmsub.vx v13, a4, v12
# CHECK-SAME: [0xd7,0x66,0xc7,0xae]
vmsub.vx v13, a4, v12
# Encoding: |101011|0|01110|10000|110|01111|1010111|
# CHECK: vmsub.vx v15, a6, v14, v0.t
# CHECK-SAME: [0xd7,0x67,0xe8,0xac]
vmsub.vx v15, a6, v14, v0.t

# Encoding: |101101|1|10000|10001|010|10010|1010111|
# CHECK: vmacc.vv v18, v17, v16
# CHECK-SAME: [0x57,0xa9,0x08,0xb7]
vmacc.vv v18, v17, v16
# Encoding: |101101|0|10011|10100|010|10101|1010111|
# CHECK: vmacc.vv v21, v20, v19, v0.t
# CHECK-SAME: [0xd7,0x2a,0x3a,0xb5]
vmacc.vv v21, v20, v19, v0.t

# Encoding: |101101|1|10110|10010|110|10111|1010111|
# CHECK: vmacc.vx v23, s2, v22
# CHECK-SAME: [0xd7,0x6b,0x69,0xb7]
vmacc.vx v23, s2, v22
# Encoding: |101101|0|11000|10100|110|11001|1010111|
# CHECK: vmacc.vx v25, s4, v24, v0.t
# CHECK-SAME: [0xd7,0x6c,0x8a,0xb5]
vmacc.vx v25, s4, v24, v0.t

# Encoding: |101111|1|11010|11011|010|11100|1010111|
# CHECK: vmsac.vv v28, v27, v26
# CHECK-SAME: [0x57,0xae,0xad,0xbf]
vmsac.vv v28, v27, v26
# Encoding: |101111|0|11101|11110|010|11111|1010111|
# CHECK: vmsac.vv v31, v30, v29, v0.t
# CHECK-SAME: [0xd7,0x2f,0xdf,0xbd]
vmsac.vv v31, v30, v29, v0.t

# Encoding: |101111|1|00000|10110|110|00001|1010111|
# CHECK: vmsac.vx v1, s6, v0
# CHECK-SAME: [0xd7,0x60,0x0b,0xbe]
vmsac.vx v1, s6, v0
# Encoding: |101111|0|00010|11000|110|00011|1010111|
# CHECK: vmsac.vx v3, s8, v2, v0.t
# CHECK-SAME: [0xd7,0x61,0x2c,0xbc]
vmsac.vx v3, s8, v2, v0.t

# Encoding: |110000|1|00100|00101|010|00110|1010111|
# CHECK: vwaddu.vv v6, v4, v5
# CHECK-SAME: [0x57,0xa3,0x42,0xc2]
vwaddu.vv v6, v4, v5
# Encoding: |110000|0|00111|01000|010|01001|1010111|
# CHECK: vwaddu.vv v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x24,0x74,0xc0]
vwaddu.vv v9, v7, v8, v0.t

# Encoding: |110000|1|01010|11010|110|01011|1010111|
# CHECK: vwaddu.vx v11, v10, s10
# CHECK-SAME: [0xd7,0x65,0xad,0xc2]
vwaddu.vx v11, v10, s10
# Encoding: |110000|0|01100|11100|110|01101|1010111|
# CHECK: vwaddu.vx v13, v12, t3, v0.t
# CHECK-SAME: [0xd7,0x66,0xce,0xc0]
vwaddu.vx v13, v12, t3, v0.t

# Encoding: |110001|1|01110|01111|010|10000|1010111|
# CHECK: vwadd.vv v16, v14, v15
# CHECK-SAME: [0x57,0xa8,0xe7,0xc6]
vwadd.vv v16, v14, v15
# Encoding: |110001|0|10001|10010|010|10011|1010111|
# CHECK: vwadd.vv v19, v17, v18, v0.t
# CHECK-SAME: [0xd7,0x29,0x19,0xc5]
vwadd.vv v19, v17, v18, v0.t

# Encoding: |110001|1|10100|11110|110|10101|1010111|
# CHECK: vwadd.vx v21, v20, t5
# CHECK-SAME: [0xd7,0x6a,0x4f,0xc7]
vwadd.vx v21, v20, t5
# Encoding: |110001|0|10110|00001|110|10111|1010111|
# CHECK: vwadd.vx v23, v22, ra, v0.t
# CHECK-SAME: [0xd7,0xeb,0x60,0xc5]
vwadd.vx v23, v22, ra, v0.t

# Encoding: |110010|1|11000|11001|010|11010|1010111|
# CHECK: vwsubu.vv v26, v24, v25
# CHECK-SAME: [0x57,0xad,0x8c,0xcb]
vwsubu.vv v26, v24, v25
# Encoding: |110010|0|11011|11100|010|11101|1010111|
# CHECK: vwsubu.vv v29, v27, v28, v0.t
# CHECK-SAME: [0xd7,0x2e,0xbe,0xc9]
vwsubu.vv v29, v27, v28, v0.t

# Encoding: |110010|1|11110|00011|110|11111|1010111|
# CHECK: vwsubu.vx v31, v30, gp
# CHECK-SAME: [0xd7,0xef,0xe1,0xcb]
vwsubu.vx v31, v30, gp
# Encoding: |110010|0|00000|00101|110|00001|1010111|
# CHECK: vwsubu.vx v1, v0, t0, v0.t
# CHECK-SAME: [0xd7,0xe0,0x02,0xc8]
vwsubu.vx v1, v0, t0, v0.t

# Encoding: |110011|1|00010|00011|010|00100|1010111|
# CHECK: vwsub.vv v4, v2, v3
# CHECK-SAME: [0x57,0xa2,0x21,0xce]
vwsub.vv v4, v2, v3
# Encoding: |110011|0|00101|00110|010|00111|1010111|
# CHECK: vwsub.vv v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x23,0x53,0xcc]
vwsub.vv v7, v5, v6, v0.t

# Encoding: |110011|1|01000|00111|110|01001|1010111|
# CHECK: vwsub.vx v9, v8, t2
# CHECK-SAME: [0xd7,0xe4,0x83,0xce]
vwsub.vx v9, v8, t2
# Encoding: |110011|0|01010|01001|110|01011|1010111|
# CHECK: vwsub.vx v11, v10, s1, v0.t
# CHECK-SAME: [0xd7,0xe5,0xa4,0xcc]
vwsub.vx v11, v10, s1, v0.t

# Encoding: |110100|1|01100|01101|010|01110|1010111|
# CHECK: vwaddu.wv v14, v12, v13
# CHECK-SAME: [0x57,0xa7,0xc6,0xd2]
vwaddu.wv v14, v12, v13
# Encoding: |110100|0|01111|10000|010|10001|1010111|
# CHECK: vwaddu.wv v17, v15, v16, v0.t
# CHECK-SAME: [0xd7,0x28,0xf8,0xd0]
vwaddu.wv v17, v15, v16, v0.t

# Encoding: |110100|1|10010|01011|110|10011|1010111|
# CHECK: vwaddu.wx v19, v18, a1
# CHECK-SAME: [0xd7,0xe9,0x25,0xd3]
vwaddu.wx v19, v18, a1
# Encoding: |110100|0|10100|01101|110|10101|1010111|
# CHECK: vwaddu.wx v21, v20, a3, v0.t
# CHECK-SAME: [0xd7,0xea,0x46,0xd1]
vwaddu.wx v21, v20, a3, v0.t

# Encoding: |110101|1|10110|10111|010|11000|1010111|
# CHECK: vwadd.wv v24, v22, v23
# CHECK-SAME: [0x57,0xac,0x6b,0xd7]
vwadd.wv v24, v22, v23
# Encoding: |110101|0|11001|11010|010|11011|1010111|
# CHECK: vwadd.wv v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x2d,0x9d,0xd5]
vwadd.wv v27, v25, v26, v0.t

# Encoding: |110101|1|11100|01111|110|11101|1010111|
# CHECK: vwadd.wx v29, v28, a5
# CHECK-SAME: [0xd7,0xee,0xc7,0xd7]
vwadd.wx v29, v28, a5
# Encoding: |110101|0|11110|10001|110|11111|1010111|
# CHECK: vwadd.wx v31, v30, a7, v0.t
# CHECK-SAME: [0xd7,0xef,0xe8,0xd5]
vwadd.wx v31, v30, a7, v0.t

# Encoding: |110110|1|00000|00001|010|00010|1010111|
# CHECK: vwsubu.wv v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0xda]
vwsubu.wv v2, v0, v1
# Encoding: |110110|0|00011|00100|010|00101|1010111|
# CHECK: vwsubu.wv v5, v3, v4, v0.t
# CHECK-SAME: [0xd7,0x22,0x32,0xd8]
vwsubu.wv v5, v3, v4, v0.t

# Encoding: |110110|1|00110|10011|110|00111|1010111|
# CHECK: vwsubu.wx v7, v6, s3
# CHECK-SAME: [0xd7,0xe3,0x69,0xda]
vwsubu.wx v7, v6, s3
# Encoding: |110110|0|01000|10101|110|01001|1010111|
# CHECK: vwsubu.wx v9, v8, s5, v0.t
# CHECK-SAME: [0xd7,0xe4,0x8a,0xd8]
vwsubu.wx v9, v8, s5, v0.t

# Encoding: |110111|1|01010|01011|010|01100|1010111|
# CHECK: vwsub.wv v12, v10, v11
# CHECK-SAME: [0x57,0xa6,0xa5,0xde]
vwsub.wv v12, v10, v11
# Encoding: |110111|0|01101|01110|010|01111|1010111|
# CHECK: vwsub.wv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x27,0xd7,0xdc]
vwsub.wv v15, v13, v14, v0.t

# Encoding: |110111|1|10000|10111|110|10001|1010111|
# CHECK: vwsub.wx v17, v16, s7
# CHECK-SAME: [0xd7,0xe8,0x0b,0xdf]
vwsub.wx v17, v16, s7
# Encoding: |110111|0|10010|11001|110|10011|1010111|
# CHECK: vwsub.wx v19, v18, s9, v0.t
# CHECK-SAME: [0xd7,0xe9,0x2c,0xdd]
vwsub.wx v19, v18, s9, v0.t

# Encoding: |111000|1|10100|10101|010|10110|1010111|
# CHECK: vwmulu.vv v22, v20, v21
# CHECK-SAME: [0x57,0xab,0x4a,0xe3]
vwmulu.vv v22, v20, v21
# Encoding: |111000|0|10111|11000|010|11001|1010111|
# CHECK: vwmulu.vv v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x2c,0x7c,0xe1]
vwmulu.vv v25, v23, v24, v0.t

# Encoding: |111000|1|11010|11011|110|11011|1010111|
# CHECK: vwmulu.vx v27, v26, s11
# CHECK-SAME: [0xd7,0xed,0xad,0xe3]
vwmulu.vx v27, v26, s11
# Encoding: |111000|0|11100|11101|110|11101|1010111|
# CHECK: vwmulu.vx v29, v28, t4, v0.t
# CHECK-SAME: [0xd7,0xee,0xce,0xe1]
vwmulu.vx v29, v28, t4, v0.t

# Encoding: |111010|1|11110|11111|010|00000|1010111|
# CHECK: vwmulsu.vv v0, v30, v31
# CHECK-SAME: [0x57,0xa0,0xef,0xeb]
vwmulsu.vv v0, v30, v31
# Encoding: |111010|0|00001|00010|010|00011|1010111|
# CHECK: vwmulsu.vv v3, v1, v2, v0.t
# CHECK-SAME: [0xd7,0x21,0x11,0xe8]
vwmulsu.vv v3, v1, v2, v0.t

# Encoding: |111010|1|00100|11111|110|00101|1010111|
# CHECK: vwmulsu.vx v5, v4, t6
# CHECK-SAME: [0xd7,0xe2,0x4f,0xea]
vwmulsu.vx v5, v4, t6
# Encoding: |111010|0|00110|00010|110|00111|1010111|
# CHECK: vwmulsu.vx v7, v6, sp, v0.t
# CHECK-SAME: [0xd7,0x63,0x61,0xe8]
vwmulsu.vx v7, v6, sp, v0.t

# Encoding: |111011|1|01000|01001|010|01010|1010111|
# CHECK: vwmul.vv v10, v8, v9
# CHECK-SAME: [0x57,0xa5,0x84,0xee]
vwmul.vv v10, v8, v9
# Encoding: |111011|0|01011|01100|010|01101|1010111|
# CHECK: vwmul.vv v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x26,0xb6,0xec]
vwmul.vv v13, v11, v12, v0.t

# Encoding: |111011|1|01110|00100|110|01111|1010111|
# CHECK: vwmul.vx v15, v14, tp
# CHECK-SAME: [0xd7,0x67,0xe2,0xee]
vwmul.vx v15, v14, tp
# Encoding: |111011|0|10000|00110|110|10001|1010111|
# CHECK: vwmul.vx v17, v16, t1, v0.t
# CHECK-SAME: [0xd7,0x68,0x03,0xed]
vwmul.vx v17, v16, t1, v0.t

# Encoding: |111100|1|10010|10011|010|10100|1010111|
# CHECK: vwmaccu.vv v20, v18, v19
# CHECK-SAME: [0x57,0xaa,0x29,0xf3]
vwmaccu.vv v20, v18, v19
# Encoding: |111100|0|10101|10110|010|10111|1010111|
# CHECK: vwmaccu.vv v23, v21, v22, v0.t
# CHECK-SAME: [0xd7,0x2b,0x5b,0xf1]
vwmaccu.vv v23, v21, v22, v0.t

# Encoding: |111100|1|11000|01000|110|11001|1010111|
# CHECK: vwmaccu.vx v25, v24, s0
# CHECK-SAME: [0xd7,0x6c,0x84,0xf3]
vwmaccu.vx v25, v24, s0
# Encoding: |111100|0|11010|01010|110|11011|1010111|
# CHECK: vwmaccu.vx v27, v26, a0, v0.t
# CHECK-SAME: [0xd7,0x6d,0xa5,0xf1]
vwmaccu.vx v27, v26, a0, v0.t

# Encoding: |111101|1|11100|11101|010|11110|1010111|
# CHECK: vwmacc.vv v30, v28, v29
# CHECK-SAME: [0x57,0xaf,0xce,0xf7]
vwmacc.vv v30, v28, v29
# Encoding: |111101|0|11111|00000|010|00001|1010111|
# CHECK: vwmacc.vv v1, v31, v0, v0.t
# CHECK-SAME: [0xd7,0x20,0xf0,0xf5]
vwmacc.vv v1, v31, v0, v0.t

# Encoding: |111101|1|00010|01100|110|00011|1010111|
# CHECK: vwmacc.vx v3, v2, a2
# CHECK-SAME: [0xd7,0x61,0x26,0xf6]
vwmacc.vx v3, v2, a2
# Encoding: |111101|0|00100|01110|110|00101|1010111|
# CHECK: vwmacc.vx v5, v4, a4, v0.t
# CHECK-SAME: [0xd7,0x62,0x47,0xf4]
vwmacc.vx v5, v4, a4, v0.t

# Encoding: |111110|1|00110|00111|010|01000|1010111|
# CHECK: vwmsacu.vv v8, v6, v7
# CHECK-SAME: [0x57,0xa4,0x63,0xfa]
vwmsacu.vv v8, v6, v7
# Encoding: |111110|0|01001|01010|010|01011|1010111|
# CHECK: vwmsacu.vv v11, v9, v10, v0.t
# CHECK-SAME: [0xd7,0x25,0x95,0xf8]
vwmsacu.vv v11, v9, v10, v0.t

# Encoding: |111110|1|01100|10000|110|01101|1010111|
# CHECK: vwmsacu.vx v13, v12, a6
# CHECK-SAME: [0xd7,0x66,0xc8,0xfa]
vwmsacu.vx v13, v12, a6
# Encoding: |111110|0|01110|10010|110|01111|1010111|
# CHECK: vwmsacu.vx v15, v14, s2, v0.t
# CHECK-SAME: [0xd7,0x67,0xe9,0xf8]
vwmsacu.vx v15, v14, s2, v0.t

# Encoding: |111111|1|10000|10001|010|10010|1010111|
# CHECK: vwmsac.vv v18, v16, v17
# CHECK-SAME: [0x57,0xa9,0x08,0xff]
vwmsac.vv v18, v16, v17
# Encoding: |111111|0|10011|10100|010|10101|1010111|
# CHECK: vwmsac.vv v21, v19, v20, v0.t
# CHECK-SAME: [0xd7,0x2a,0x3a,0xfd]
vwmsac.vv v21, v19, v20, v0.t

# Encoding: |111111|1|10110|10100|110|10111|1010111|
# CHECK: vwmsac.vx v23, v22, s4
# CHECK-SAME: [0xd7,0x6b,0x6a,0xff]
vwmsac.vx v23, v22, s4
# Encoding: |111111|0|11000|10110|110|11001|1010111|
# CHECK: vwmsac.vx v25, v24, s6, v0.t
# CHECK-SAME: [0xd7,0x6c,0x8b,0xfd]
vwmsac.vx v25, v24, s6, v0.t

# Encoding: |000000|1|11010|11011|001|11100|1010111|
# CHECK: vfadd.vv v28, v26, v27
# CHECK-SAME: [0x57,0x9e,0xad,0x03]
vfadd.vv v28, v26, v27
# Encoding: |000000|0|11101|11110|001|11111|1010111|
# CHECK: vfadd.vv v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x1f,0xdf,0x01]
vfadd.vv v31, v29, v30, v0.t

# Encoding: |000000|1|00000|00000|101|00001|1010111|
# CHECK: vfadd.vf v1, v0, ft0
# CHECK-SAME: [0xd7,0x50,0x00,0x02]
vfadd.vf v1, v0, ft0
# Encoding: |000000|0|00010|00001|101|00011|1010111|
# CHECK: vfadd.vf v3, v2, ft1, v0.t
# CHECK-SAME: [0xd7,0xd1,0x20,0x00]
vfadd.vf v3, v2, ft1, v0.t

# Encoding: |000001|1|00100|00101|001|00110|1010111|
# CHECK: vfredsum.vs v6, v4, v5
# CHECK-SAME: [0x57,0x93,0x42,0x06]
vfredsum.vs v6, v4, v5
# Encoding: |000001|0|00111|01000|001|01001|1010111|
# CHECK: vfredsum.vs v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x14,0x74,0x04]
vfredsum.vs v9, v7, v8, v0.t

# Encoding: |000010|1|01010|01011|001|01100|1010111|
# CHECK: vfsub.vv v12, v10, v11
# CHECK-SAME: [0x57,0x96,0xa5,0x0a]
vfsub.vv v12, v10, v11
# Encoding: |000010|0|01101|01110|001|01111|1010111|
# CHECK: vfsub.vv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x17,0xd7,0x08]
vfsub.vv v15, v13, v14, v0.t

# Encoding: |000010|1|10000|00010|101|10001|1010111|
# CHECK: vfsub.vf v17, v16, ft2
# CHECK-SAME: [0xd7,0x58,0x01,0x0b]
vfsub.vf v17, v16, ft2
# Encoding: |000010|0|10010|00011|101|10011|1010111|
# CHECK: vfsub.vf v19, v18, ft3, v0.t
# CHECK-SAME: [0xd7,0xd9,0x21,0x09]
vfsub.vf v19, v18, ft3, v0.t

# Encoding: |000011|1|10100|10101|001|10110|1010111|
# CHECK: vfredosum.vs v22, v20, v21
# CHECK-SAME: [0x57,0x9b,0x4a,0x0f]
vfredosum.vs v22, v20, v21
# Encoding: |000011|0|10111|11000|001|11001|1010111|
# CHECK: vfredosum.vs v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x1c,0x7c,0x0d]
vfredosum.vs v25, v23, v24, v0.t

# Encoding: |000100|1|11010|11011|001|11100|1010111|
# CHECK: vfmin.vv v28, v26, v27
# CHECK-SAME: [0x57,0x9e,0xad,0x13]
vfmin.vv v28, v26, v27
# Encoding: |000100|0|11101|11110|001|11111|1010111|
# CHECK: vfmin.vv v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x1f,0xdf,0x11]
vfmin.vv v31, v29, v30, v0.t

# Encoding: |000100|1|00000|00100|101|00001|1010111|
# CHECK: vfmin.vf v1, v0, ft4
# CHECK-SAME: [0xd7,0x50,0x02,0x12]
vfmin.vf v1, v0, ft4
# Encoding: |000100|0|00010|00101|101|00011|1010111|
# CHECK: vfmin.vf v3, v2, ft5, v0.t
# CHECK-SAME: [0xd7,0xd1,0x22,0x10]
vfmin.vf v3, v2, ft5, v0.t

# Encoding: |000101|1|00100|00101|001|00110|1010111|
# CHECK: vfredmin.vs v6, v4, v5
# CHECK-SAME: [0x57,0x93,0x42,0x16]
vfredmin.vs v6, v4, v5
# Encoding: |000101|0|00111|01000|001|01001|1010111|
# CHECK: vfredmin.vs v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x14,0x74,0x14]
vfredmin.vs v9, v7, v8, v0.t

# Encoding: |000110|1|01010|01011|001|01100|1010111|
# CHECK: vfmax.vv v12, v10, v11
# CHECK-SAME: [0x57,0x96,0xa5,0x1a]
vfmax.vv v12, v10, v11
# Encoding: |000110|0|01101|01110|001|01111|1010111|
# CHECK: vfmax.vv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x17,0xd7,0x18]
vfmax.vv v15, v13, v14, v0.t

# Encoding: |000110|1|10000|00110|101|10001|1010111|
# CHECK: vfmax.vf v17, v16, ft6
# CHECK-SAME: [0xd7,0x58,0x03,0x1b]
vfmax.vf v17, v16, ft6
# Encoding: |000110|0|10010|00111|101|10011|1010111|
# CHECK: vfmax.vf v19, v18, ft7, v0.t
# CHECK-SAME: [0xd7,0xd9,0x23,0x19]
vfmax.vf v19, v18, ft7, v0.t

# Encoding: |000111|1|10100|10101|001|10110|1010111|
# CHECK: vfredmax.vs v22, v20, v21
# CHECK-SAME: [0x57,0x9b,0x4a,0x1f]
vfredmax.vs v22, v20, v21
# Encoding: |000111|0|10111|11000|001|11001|1010111|
# CHECK: vfredmax.vs v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x1c,0x7c,0x1d]
vfredmax.vs v25, v23, v24, v0.t

# Encoding: |001000|1|11010|11011|001|11100|1010111|
# CHECK: vfsgnj.vv v28, v26, v27
# CHECK-SAME: [0x57,0x9e,0xad,0x23]
vfsgnj.vv v28, v26, v27
# Encoding: |001000|0|11101|11110|001|11111|1010111|
# CHECK: vfsgnj.vv v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x1f,0xdf,0x21]
vfsgnj.vv v31, v29, v30, v0.t

# Encoding: |001000|1|00000|01000|101|00001|1010111|
# CHECK: vfsgnj.vf v1, v0, fs0
# CHECK-SAME: [0xd7,0x50,0x04,0x22]
vfsgnj.vf v1, v0, fs0
# Encoding: |001000|0|00010|01001|101|00011|1010111|
# CHECK: vfsgnj.vf v3, v2, fs1, v0.t
# CHECK-SAME: [0xd7,0xd1,0x24,0x20]
vfsgnj.vf v3, v2, fs1, v0.t

# Encoding: |001001|1|00100|00101|001|00110|1010111|
# CHECK: vfsgnjn.vv v6, v4, v5
# CHECK-SAME: [0x57,0x93,0x42,0x26]
vfsgnjn.vv v6, v4, v5
# Encoding: |001001|0|00111|01000|001|01001|1010111|
# CHECK: vfsgnjn.vv v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x14,0x74,0x24]
vfsgnjn.vv v9, v7, v8, v0.t

# Encoding: |001001|1|01010|01010|101|01011|1010111|
# CHECK: vfsgnjn.vf v11, v10, fa0
# CHECK-SAME: [0xd7,0x55,0xa5,0x26]
vfsgnjn.vf v11, v10, fa0
# Encoding: |001001|0|01100|01011|101|01101|1010111|
# CHECK: vfsgnjn.vf v13, v12, fa1, v0.t
# CHECK-SAME: [0xd7,0xd6,0xc5,0x24]
vfsgnjn.vf v13, v12, fa1, v0.t

# Encoding: |001010|1|01110|01111|001|10000|1010111|
# CHECK: vfsgnjx.vv v16, v14, v15
# CHECK-SAME: [0x57,0x98,0xe7,0x2a]
vfsgnjx.vv v16, v14, v15
# Encoding: |001010|0|10001|10010|001|10011|1010111|
# CHECK: vfsgnjx.vv v19, v17, v18, v0.t
# CHECK-SAME: [0xd7,0x19,0x19,0x29]
vfsgnjx.vv v19, v17, v18, v0.t

# Encoding: |001010|1|10100|01100|101|10101|1010111|
# CHECK: vfsgnjx.vf v21, v20, fa2
# CHECK-SAME: [0xd7,0x5a,0x46,0x2b]
vfsgnjx.vf v21, v20, fa2
# Encoding: |001010|0|10110|01101|101|10111|1010111|
# CHECK: vfsgnjx.vf v23, v22, fa3, v0.t
# CHECK-SAME: [0xd7,0xdb,0x66,0x29]
vfsgnjx.vf v23, v22, fa3, v0.t

# Encoding: |001100|1|11000|00000|001|01110|1010111|
# CHECK: vfmv.f.s fa4, v24
# CHECK-SAME: [0x57,0x17,0x80,0x33]
vfmv.f.s fa4, v24

# Encoding: |001101|1|00000|01111|101|11001|1010111|
# CHECK: vfmv.s.f v25, fa5
# CHECK-SAME: [0xd7,0xdc,0x07,0x36]
vfmv.s.f v25, fa5

# Encoding: |010111|1|00000|10000|101|11010|1010111|
# CHECK: vfmv.v.f v26, fa6
# CHECK-SAME: [0x57,0x5d,0x08,0x5e]
vfmv.v.f v26, fa6

# Encoding: |010111|0|11011|10001|101|11100|1010111|
# CHECK: vfmerge.vfm v28, v27, fa7, v0
# CHECK-SAME: [0x57,0xde,0xb8,0x5d]
vfmerge.vfm v28, v27, fa7, v0

# Encoding: |011000|1|11101|11110|001|11111|1010111|
# CHECK: vfeq.vv v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0x63]
vfeq.vv v31, v29, v30
# Encoding: |011000|0|00000|00001|001|00010|1010111|
# CHECK: vfeq.vv v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0x60]
vfeq.vv v2, v0, v1, v0.t

# Encoding: |011000|1|00011|10010|101|00100|1010111|
# CHECK: vfeq.vf v4, v3, fs2
# CHECK-SAME: [0x57,0x52,0x39,0x62]
vfeq.vf v4, v3, fs2
# Encoding: |011000|0|00101|10011|101|00110|1010111|
# CHECK: vfeq.vf v6, v5, fs3, v0.t
# CHECK-SAME: [0x57,0xd3,0x59,0x60]
vfeq.vf v6, v5, fs3, v0.t

# Encoding: |011001|1|00111|01000|001|01001|1010111|
# CHECK: vfle.vv v9, v7, v8
# CHECK-SAME: [0xd7,0x14,0x74,0x66]
vfle.vv v9, v7, v8
# Encoding: |011001|0|01010|01011|001|01100|1010111|
# CHECK: vfle.vv v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0x96,0xa5,0x64]
vfle.vv v12, v10, v11, v0.t

# Encoding: |011001|1|01101|10100|101|01110|1010111|
# CHECK: vfle.vf v14, v13, fs4
# CHECK-SAME: [0x57,0x57,0xda,0x66]
vfle.vf v14, v13, fs4
# Encoding: |011001|0|01111|10101|101|10000|1010111|
# CHECK: vfle.vf v16, v15, fs5, v0.t
# CHECK-SAME: [0x57,0xd8,0xfa,0x64]
vfle.vf v16, v15, fs5, v0.t

# Encoding: |011010|1|10001|10010|001|10011|1010111|
# CHECK: vford.vv v19, v17, v18
# CHECK-SAME: [0xd7,0x19,0x19,0x6b]
vford.vv v19, v17, v18
# Encoding: |011010|0|10100|10101|001|10110|1010111|
# CHECK: vford.vv v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0x9b,0x4a,0x69]
vford.vv v22, v20, v21, v0.t

# Encoding: |011010|1|10111|10110|101|11000|1010111|
# CHECK: vford.vf v24, v23, fs6
# CHECK-SAME: [0x57,0x5c,0x7b,0x6b]
vford.vf v24, v23, fs6
# Encoding: |011010|0|11001|10111|101|11010|1010111|
# CHECK: vford.vf v26, v25, fs7, v0.t
# CHECK-SAME: [0x57,0xdd,0x9b,0x69]
vford.vf v26, v25, fs7, v0.t

# Encoding: |011011|1|11011|11100|001|11101|1010111|
# CHECK: vflt.vv v29, v27, v28
# CHECK-SAME: [0xd7,0x1e,0xbe,0x6f]
vflt.vv v29, v27, v28
# Encoding: |011011|0|11110|11111|001|00000|1010111|
# CHECK: vflt.vv v0, v30, v31, v0.t
# CHECK-SAME: [0x57,0x90,0xef,0x6d]
vflt.vv v0, v30, v31, v0.t

# Encoding: |011011|1|00001|11000|101|00010|1010111|
# CHECK: vflt.vf v2, v1, fs8
# CHECK-SAME: [0x57,0x51,0x1c,0x6e]
vflt.vf v2, v1, fs8
# Encoding: |011011|0|00011|11001|101|00100|1010111|
# CHECK: vflt.vf v4, v3, fs9, v0.t
# CHECK-SAME: [0x57,0xd2,0x3c,0x6c]
vflt.vf v4, v3, fs9, v0.t

# Encoding: |011100|1|00101|00110|001|00111|1010111|
# CHECK: vfne.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x13,0x53,0x72]
vfne.vv v7, v5, v6
# Encoding: |011100|0|01000|01001|001|01010|1010111|
# CHECK: vfne.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x95,0x84,0x70]
vfne.vv v10, v8, v9, v0.t

# Encoding: |011100|1|01011|11010|101|01100|1010111|
# CHECK: vfne.vf v12, v11, fs10
# CHECK-SAME: [0x57,0x56,0xbd,0x72]
vfne.vf v12, v11, fs10
# Encoding: |011100|0|01101|11011|101|01110|1010111|
# CHECK: vfne.vf v14, v13, fs11, v0.t
# CHECK-SAME: [0x57,0xd7,0xdd,0x70]
vfne.vf v14, v13, fs11, v0.t

# Encoding: |011101|1|01111|11100|101|10000|1010111|
# CHECK: vfgt.vf v16, v15, ft8
# CHECK-SAME: [0x57,0x58,0xfe,0x76]
vfgt.vf v16, v15, ft8
# Encoding: |011101|0|10001|11101|101|10010|1010111|
# CHECK: vfgt.vf v18, v17, ft9, v0.t
# CHECK-SAME: [0x57,0xd9,0x1e,0x75]
vfgt.vf v18, v17, ft9, v0.t

# Encoding: |011111|1|10011|11110|101|10100|1010111|
# CHECK: vfge.vf v20, v19, ft10
# CHECK-SAME: [0x57,0x5a,0x3f,0x7f]
vfge.vf v20, v19, ft10
# Encoding: |011111|0|10101|00000|101|10110|1010111|
# CHECK: vfge.vf v22, v21, ft0, v0.t
# CHECK-SAME: [0x57,0x5b,0x50,0x7d]
vfge.vf v22, v21, ft0, v0.t

# Encoding: |100000|1|10111|11000|001|11001|1010111|
# CHECK: vfdiv.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x1c,0x7c,0x83]
vfdiv.vv v25, v23, v24
# Encoding: |100000|0|11010|11011|001|11100|1010111|
# CHECK: vfdiv.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0x81]
vfdiv.vv v28, v26, v27, v0.t

# Encoding: |100000|1|11101|00001|101|11110|1010111|
# CHECK: vfdiv.vf v30, v29, ft1
# CHECK-SAME: [0x57,0xdf,0xd0,0x83]
vfdiv.vf v30, v29, ft1
# Encoding: |100000|0|11111|00010|101|00000|1010111|
# CHECK: vfdiv.vf v0, v31, ft2, v0.t
# CHECK-SAME: [0x57,0x50,0xf1,0x81]
vfdiv.vf v0, v31, ft2, v0.t

# Encoding: |100001|1|00001|00011|101|00010|1010111|
# CHECK: vfrdiv.vf v2, v1, ft3
# CHECK-SAME: [0x57,0xd1,0x11,0x86]
vfrdiv.vf v2, v1, ft3
# Encoding: |100001|0|00011|00100|101|00100|1010111|
# CHECK: vfrdiv.vf v4, v3, ft4, v0.t
# CHECK-SAME: [0x57,0x52,0x32,0x84]
vfrdiv.vf v4, v3, ft4, v0.t

# Encoding: |100100|1|00101|00110|001|00111|1010111|
# CHECK: vfmul.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x13,0x53,0x92]
vfmul.vv v7, v5, v6
# Encoding: |100100|0|01000|01001|001|01010|1010111|
# CHECK: vfmul.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0x95,0x84,0x90]
vfmul.vv v10, v8, v9, v0.t

# Encoding: |100100|1|01011|00101|101|01100|1010111|
# CHECK: vfmul.vf v12, v11, ft5
# CHECK-SAME: [0x57,0xd6,0xb2,0x92]
vfmul.vf v12, v11, ft5
# Encoding: |100100|0|01101|00110|101|01110|1010111|
# CHECK: vfmul.vf v14, v13, ft6, v0.t
# CHECK-SAME: [0x57,0x57,0xd3,0x90]
vfmul.vf v14, v13, ft6, v0.t

# Encoding: |101000|1|01111|10000|001|10001|1010111|
# CHECK: vfmadd.vv v17, v16, v15
# CHECK-SAME: [0xd7,0x18,0xf8,0xa2]
vfmadd.vv v17, v16, v15
# Encoding: |101000|0|10010|10011|001|10100|1010111|
# CHECK: vfmadd.vv v20, v19, v18, v0.t
# CHECK-SAME: [0x57,0x9a,0x29,0xa1]
vfmadd.vv v20, v19, v18, v0.t

# Encoding: |101000|1|10101|00111|101|10110|1010111|
# CHECK: vfmadd.vf v22, ft7, v21
# CHECK-SAME: [0x57,0xdb,0x53,0xa3]
vfmadd.vf v22, ft7, v21
# Encoding: |101000|0|10111|01000|101|11000|1010111|
# CHECK: vfmadd.vf v24, fs0, v23, v0.t
# CHECK-SAME: [0x57,0x5c,0x74,0xa1]
vfmadd.vf v24, fs0, v23, v0.t

# Encoding: |101001|1|11001|11010|001|11011|1010111|
# CHECK: vfnmadd.vv v27, v26, v25
# CHECK-SAME: [0xd7,0x1d,0x9d,0xa7]
vfnmadd.vv v27, v26, v25
# Encoding: |101001|0|11100|11101|001|11110|1010111|
# CHECK: vfnmadd.vv v30, v29, v28, v0.t
# CHECK-SAME: [0x57,0x9f,0xce,0xa5]
vfnmadd.vv v30, v29, v28, v0.t

# Encoding: |101001|1|11111|01001|101|00000|1010111|
# CHECK: vfnmadd.vf v0, fs1, v31
# CHECK-SAME: [0x57,0xd0,0xf4,0xa7]
vfnmadd.vf v0, fs1, v31
# Encoding: |101001|0|00001|01010|101|00010|1010111|
# CHECK: vfnmadd.vf v2, fa0, v1, v0.t
# CHECK-SAME: [0x57,0x51,0x15,0xa4]
vfnmadd.vf v2, fa0, v1, v0.t

# Encoding: |101010|1|00011|00100|001|00101|1010111|
# CHECK: vfmsub.vv v5, v4, v3
# CHECK-SAME: [0xd7,0x12,0x32,0xaa]
vfmsub.vv v5, v4, v3
# Encoding: |101010|0|00110|00111|001|01000|1010111|
# CHECK: vfmsub.vv v8, v7, v6, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0xa8]
vfmsub.vv v8, v7, v6, v0.t

# Encoding: |101010|1|01001|01011|101|01010|1010111|
# CHECK: vfmsub.vf v10, fa1, v9
# CHECK-SAME: [0x57,0xd5,0x95,0xaa]
vfmsub.vf v10, fa1, v9
# Encoding: |101010|0|01011|01100|101|01100|1010111|
# CHECK: vfmsub.vf v12, fa2, v11, v0.t
# CHECK-SAME: [0x57,0x56,0xb6,0xa8]
vfmsub.vf v12, fa2, v11, v0.t

# Encoding: |101011|1|01101|01110|001|01111|1010111|
# CHECK: vfnmsub.vv v15, v14, v13
# CHECK-SAME: [0xd7,0x17,0xd7,0xae]
vfnmsub.vv v15, v14, v13
# Encoding: |101011|0|10000|10001|001|10010|1010111|
# CHECK: vfnmsub.vv v18, v17, v16, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0xad]
vfnmsub.vv v18, v17, v16, v0.t

# Encoding: |101011|1|10011|01101|101|10100|1010111|
# CHECK: vfnmsub.vf v20, fa3, v19
# CHECK-SAME: [0x57,0xda,0x36,0xaf]
vfnmsub.vf v20, fa3, v19
# Encoding: |101011|0|10101|01110|101|10110|1010111|
# CHECK: vfnmsub.vf v22, fa4, v21, v0.t
# CHECK-SAME: [0x57,0x5b,0x57,0xad]
vfnmsub.vf v22, fa4, v21, v0.t

# Encoding: |101100|1|10111|11000|001|11001|1010111|
# CHECK: vfmacc.vv v25, v24, v23
# CHECK-SAME: [0xd7,0x1c,0x7c,0xb3]
vfmacc.vv v25, v24, v23
# Encoding: |101100|0|11010|11011|001|11100|1010111|
# CHECK: vfmacc.vv v28, v27, v26, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0xb1]
vfmacc.vv v28, v27, v26, v0.t

# Encoding: |101100|1|11101|01111|101|11110|1010111|
# CHECK: vfmacc.vf v30, fa5, v29
# CHECK-SAME: [0x57,0xdf,0xd7,0xb3]
vfmacc.vf v30, fa5, v29
# Encoding: |101100|0|11111|10000|101|00000|1010111|
# CHECK: vfmacc.vf v0, fa6, v31, v0.t
# CHECK-SAME: [0x57,0x50,0xf8,0xb1]
vfmacc.vf v0, fa6, v31, v0.t

# Encoding: |101101|1|00001|00010|001|00011|1010111|
# CHECK: vfnmacc.vv v3, v2, v1
# CHECK-SAME: [0xd7,0x11,0x11,0xb6]
vfnmacc.vv v3, v2, v1
# Encoding: |101101|0|00100|00101|001|00110|1010111|
# CHECK: vfnmacc.vv v6, v5, v4, v0.t
# CHECK-SAME: [0x57,0x93,0x42,0xb4]
vfnmacc.vv v6, v5, v4, v0.t

# Encoding: |101101|1|00111|10001|101|01000|1010111|
# CHECK: vfnmacc.vf v8, fa7, v7
# CHECK-SAME: [0x57,0xd4,0x78,0xb6]
vfnmacc.vf v8, fa7, v7
# Encoding: |101101|0|01001|10010|101|01010|1010111|
# CHECK: vfnmacc.vf v10, fs2, v9, v0.t
# CHECK-SAME: [0x57,0x55,0x99,0xb4]
vfnmacc.vf v10, fs2, v9, v0.t

# Encoding: |101110|1|01011|01100|001|01101|1010111|
# CHECK: vfmsac.vv v13, v12, v11
# CHECK-SAME: [0xd7,0x16,0xb6,0xba]
vfmsac.vv v13, v12, v11
# Encoding: |101110|0|01110|01111|001|10000|1010111|
# CHECK: vfmsac.vv v16, v15, v14, v0.t
# CHECK-SAME: [0x57,0x98,0xe7,0xb8]
vfmsac.vv v16, v15, v14, v0.t

# Encoding: |101110|1|10001|10011|101|10010|1010111|
# CHECK: vfmsac.vf v18, fs3, v17
# CHECK-SAME: [0x57,0xd9,0x19,0xbb]
vfmsac.vf v18, fs3, v17
# Encoding: |101110|0|10011|10100|101|10100|1010111|
# CHECK: vfmsac.vf v20, fs4, v19, v0.t
# CHECK-SAME: [0x57,0x5a,0x3a,0xb9]
vfmsac.vf v20, fs4, v19, v0.t

# Encoding: |101111|1|10101|10110|001|10111|1010111|
# CHECK: vfnmsac.vv v23, v22, v21
# CHECK-SAME: [0xd7,0x1b,0x5b,0xbf]
vfnmsac.vv v23, v22, v21
# Encoding: |101111|0|11000|11001|001|11010|1010111|
# CHECK: vfnmsac.vv v26, v25, v24, v0.t
# CHECK-SAME: [0x57,0x9d,0x8c,0xbd]
vfnmsac.vv v26, v25, v24, v0.t

# Encoding: |101111|1|11011|10101|101|11100|1010111|
# CHECK: vfnmsac.vf v28, fs5, v27
# CHECK-SAME: [0x57,0xde,0xba,0xbf]
vfnmsac.vf v28, fs5, v27
# Encoding: |101111|0|11101|10110|101|11110|1010111|
# CHECK: vfnmsac.vf v30, fs6, v29, v0.t
# CHECK-SAME: [0x57,0x5f,0xdb,0xbd]
vfnmsac.vf v30, fs6, v29, v0.t

# Encoding: |110000|1|11111|00000|001|00001|1010111|
# CHECK: vfwadd.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x10,0xf0,0xc3]
vfwadd.vv v1, v31, v0
# Encoding: |110000|0|00010|00011|001|00100|1010111|
# CHECK: vfwadd.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x92,0x21,0xc0]
vfwadd.vv v4, v2, v3, v0.t

# Encoding: |110000|1|00101|10111|101|00110|1010111|
# CHECK: vfwadd.vf v6, v5, fs7
# CHECK-SAME: [0x57,0xd3,0x5b,0xc2]
vfwadd.vf v6, v5, fs7
# Encoding: |110000|0|00111|11000|101|01000|1010111|
# CHECK: vfwadd.vf v8, v7, fs8, v0.t
# CHECK-SAME: [0x57,0x54,0x7c,0xc0]
vfwadd.vf v8, v7, fs8, v0.t

# Encoding: |110001|1|01001|01010|001|01011|1010111|
# CHECK: vfwredsum.vs v11, v9, v10
# CHECK-SAME: [0xd7,0x15,0x95,0xc6]
vfwredsum.vs v11, v9, v10
# Encoding: |110001|0|01100|01101|001|01110|1010111|
# CHECK: vfwredsum.vs v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x97,0xc6,0xc4]
vfwredsum.vs v14, v12, v13, v0.t

# Encoding: |110010|1|01111|10000|001|10001|1010111|
# CHECK: vfwsub.vv v17, v15, v16
# CHECK-SAME: [0xd7,0x18,0xf8,0xca]
vfwsub.vv v17, v15, v16
# Encoding: |110010|0|10010|10011|001|10100|1010111|
# CHECK: vfwsub.vv v20, v18, v19, v0.t
# CHECK-SAME: [0x57,0x9a,0x29,0xc9]
vfwsub.vv v20, v18, v19, v0.t

# Encoding: |110010|1|10101|11001|101|10110|1010111|
# CHECK: vfwsub.vf v22, v21, fs9
# CHECK-SAME: [0x57,0xdb,0x5c,0xcb]
vfwsub.vf v22, v21, fs9
# Encoding: |110010|0|10111|11010|101|11000|1010111|
# CHECK: vfwsub.vf v24, v23, fs10, v0.t
# CHECK-SAME: [0x57,0x5c,0x7d,0xc9]
vfwsub.vf v24, v23, fs10, v0.t

# Encoding: |110011|1|11001|11010|001|11011|1010111|
# CHECK: vfwredosum.vs v27, v25, v26
# CHECK-SAME: [0xd7,0x1d,0x9d,0xcf]
vfwredosum.vs v27, v25, v26
# Encoding: |110011|0|11100|11101|001|11110|1010111|
# CHECK: vfwredosum.vs v30, v28, v29, v0.t
# CHECK-SAME: [0x57,0x9f,0xce,0xcd]
vfwredosum.vs v30, v28, v29, v0.t

# Encoding: |110100|1|11111|00000|001|00001|1010111|
# CHECK: vfwadd.wv v1, v31, v0
# CHECK-SAME: [0xd7,0x10,0xf0,0xd3]
vfwadd.wv v1, v31, v0
# Encoding: |110100|0|00010|00011|001|00100|1010111|
# CHECK: vfwadd.wv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x92,0x21,0xd0]
vfwadd.wv v4, v2, v3, v0.t

# Encoding: |110100|1|00101|11011|101|00110|1010111|
# CHECK: vfwadd.wf v6, v5, fs11
# CHECK-SAME: [0x57,0xd3,0x5d,0xd2]
vfwadd.wf v6, v5, fs11
# Encoding: |110100|0|00111|11100|101|01000|1010111|
# CHECK: vfwadd.wf v8, v7, ft8, v0.t
# CHECK-SAME: [0x57,0x54,0x7e,0xd0]
vfwadd.wf v8, v7, ft8, v0.t

# Encoding: |110110|1|01001|01010|001|01011|1010111|
# CHECK: vfwsub.wv v11, v9, v10
# CHECK-SAME: [0xd7,0x15,0x95,0xda]
vfwsub.wv v11, v9, v10
# Encoding: |110110|0|01100|01101|001|01110|1010111|
# CHECK: vfwsub.wv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x97,0xc6,0xd8]
vfwsub.wv v14, v12, v13, v0.t

# Encoding: |110110|1|01111|11101|101|10000|1010111|
# CHECK: vfwsub.wf v16, v15, ft9
# CHECK-SAME: [0x57,0xd8,0xfe,0xda]
vfwsub.wf v16, v15, ft9
# Encoding: |110110|0|10001|11110|101|10010|1010111|
# CHECK: vfwsub.wf v18, v17, ft10, v0.t
# CHECK-SAME: [0x57,0x59,0x1f,0xd9]
vfwsub.wf v18, v17, ft10, v0.t

# Encoding: |111000|1|10011|10100|001|10101|1010111|
# CHECK: vfwmul.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x1a,0x3a,0xe3]
vfwmul.vv v21, v19, v20
# Encoding: |111000|0|10110|10111|001|11000|1010111|
# CHECK: vfwmul.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x9c,0x6b,0xe1]
vfwmul.vv v24, v22, v23, v0.t

# Encoding: |111000|1|11001|00000|101|11010|1010111|
# CHECK: vfwmul.vf v26, v25, ft0
# CHECK-SAME: [0x57,0x5d,0x90,0xe3]
vfwmul.vf v26, v25, ft0
# Encoding: |111000|0|11011|00001|101|11100|1010111|
# CHECK: vfwmul.vf v28, v27, ft1, v0.t
# CHECK-SAME: [0x57,0xde,0xb0,0xe1]
vfwmul.vf v28, v27, ft1, v0.t

# Encoding: |111001|1|11101|11110|001|11111|1010111|
# CHECK: vfdot.vv v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0xe7]
vfdot.vv v31, v29, v30
# Encoding: |111001|0|00000|00001|001|00010|1010111|
# CHECK: vfdot.vv v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0xe4]
vfdot.vv v2, v0, v1, v0.t

# Encoding: |111100|1|00011|00100|001|00101|1010111|
# CHECK: vfwmacc.vv v5, v4, v3
# CHECK-SAME: [0xd7,0x12,0x32,0xf2]
vfwmacc.vv v5, v4, v3
# Encoding: |111100|0|00110|00111|001|01000|1010111|
# CHECK: vfwmacc.vv v8, v7, v6, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0xf0]
vfwmacc.vv v8, v7, v6, v0.t

# Encoding: |111100|1|01001|00010|101|01010|1010111|
# CHECK: vfwmacc.vf v10, ft2, v9
# CHECK-SAME: [0x57,0x55,0x91,0xf2]
vfwmacc.vf v10, ft2, v9
# Encoding: |111100|0|01011|00011|101|01100|1010111|
# CHECK: vfwmacc.vf v12, ft3, v11, v0.t
# CHECK-SAME: [0x57,0xd6,0xb1,0xf0]
vfwmacc.vf v12, ft3, v11, v0.t

# Encoding: |111101|1|01101|01110|001|01111|1010111|
# CHECK: vfwnmacc.vv v15, v14, v13
# CHECK-SAME: [0xd7,0x17,0xd7,0xf6]
vfwnmacc.vv v15, v14, v13
# Encoding: |111101|0|10000|10001|001|10010|1010111|
# CHECK: vfwnmacc.vv v18, v17, v16, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0xf5]
vfwnmacc.vv v18, v17, v16, v0.t

# Encoding: |111101|1|10011|00100|101|10100|1010111|
# CHECK: vfwnmacc.vf v20, ft4, v19
# CHECK-SAME: [0x57,0x5a,0x32,0xf7]
vfwnmacc.vf v20, ft4, v19
# Encoding: |111101|0|10101|00101|101|10110|1010111|
# CHECK: vfwnmacc.vf v22, ft5, v21, v0.t
# CHECK-SAME: [0x57,0xdb,0x52,0xf5]
vfwnmacc.vf v22, ft5, v21, v0.t

# Encoding: |111110|1|10111|11000|001|11001|1010111|
# CHECK: vfwmsac.vv v25, v24, v23
# CHECK-SAME: [0xd7,0x1c,0x7c,0xfb]
vfwmsac.vv v25, v24, v23
# Encoding: |111110|0|11010|11011|001|11100|1010111|
# CHECK: vfwmsac.vv v28, v27, v26, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0xf9]
vfwmsac.vv v28, v27, v26, v0.t

# Encoding: |111110|1|11101|00110|101|11110|1010111|
# CHECK: vfwmsac.vf v30, ft6, v29
# CHECK-SAME: [0x57,0x5f,0xd3,0xfb]
vfwmsac.vf v30, ft6, v29
# Encoding: |111110|0|11111|00111|101|00000|1010111|
# CHECK: vfwmsac.vf v0, ft7, v31, v0.t
# CHECK-SAME: [0x57,0xd0,0xf3,0xf9]
vfwmsac.vf v0, ft7, v31, v0.t

# Encoding: |111111|1|00001|00010|001|00011|1010111|
# CHECK: vfwnmsac.vv v3, v2, v1
# CHECK-SAME: [0xd7,0x11,0x11,0xfe]
vfwnmsac.vv v3, v2, v1
# Encoding: |111111|0|00100|00101|001|00110|1010111|
# CHECK: vfwnmsac.vv v6, v5, v4, v0.t
# CHECK-SAME: [0x57,0x93,0x42,0xfc]
vfwnmsac.vv v6, v5, v4, v0.t

# Encoding: |111111|1|00111|01000|101|01000|1010111|
# CHECK: vfwnmsac.vf v8, fs0, v7
# CHECK-SAME: [0x57,0x54,0x74,0xfe]
vfwnmsac.vf v8, fs0, v7
# Encoding: |111111|0|01001|01001|101|01010|1010111|
# CHECK: vfwnmsac.vf v10, fs1, v9, v0.t
# CHECK-SAME: [0x57,0xd5,0x94,0xfc]
vfwnmsac.vf v10, fs1, v9, v0.t

# Encoding: |100011|1|01011|00000|001|01100|1010111|
# CHECK: vfsqrt.v v12, v11
# CHECK-SAME: [0x57,0x16,0xb0,0x8e]
vfsqrt.v v12, v11
# Encoding: |100011|0|01101|00000|001|01110|1010111|
# CHECK: vfsqrt.v v14, v13, v0.t
# CHECK-SAME: [0x57,0x17,0xd0,0x8c]
vfsqrt.v v14, v13, v0.t

# Encoding: |100011|1|01111|10000|001|10000|1010111|
# CHECK: vfclass.v v16, v15
# CHECK-SAME: [0x57,0x18,0xf8,0x8e]
vfclass.v v16, v15
# Encoding: |100011|0|10001|10000|001|10010|1010111|
# CHECK: vfclass.v v18, v17, v0.t
# CHECK-SAME: [0x57,0x19,0x18,0x8d]
vfclass.v v18, v17, v0.t

# Encoding: |100010|1|10011|00000|001|10100|1010111|
# CHECK: vfcvt.xu.f.v v20, v19
# CHECK-SAME: [0x57,0x1a,0x30,0x8b]
vfcvt.xu.f.v v20, v19
# Encoding: |100010|0|10101|00000|001|10110|1010111|
# CHECK: vfcvt.xu.f.v v22, v21, v0.t
# CHECK-SAME: [0x57,0x1b,0x50,0x89]
vfcvt.xu.f.v v22, v21, v0.t

# Encoding: |100010|1|10111|00001|001|11000|1010111|
# CHECK: vfcvt.x.f.v v24, v23
# CHECK-SAME: [0x57,0x9c,0x70,0x8b]
vfcvt.x.f.v v24, v23
# Encoding: |100010|0|11001|00001|001|11010|1010111|
# CHECK: vfcvt.x.f.v v26, v25, v0.t
# CHECK-SAME: [0x57,0x9d,0x90,0x89]
vfcvt.x.f.v v26, v25, v0.t

# Encoding: |100010|1|11011|00010|001|11100|1010111|
# CHECK: vfcvt.f.xu.v v28, v27
# CHECK-SAME: [0x57,0x1e,0xb1,0x8b]
vfcvt.f.xu.v v28, v27
# Encoding: |100010|0|11101|00010|001|11110|1010111|
# CHECK: vfcvt.f.xu.v v30, v29, v0.t
# CHECK-SAME: [0x57,0x1f,0xd1,0x89]
vfcvt.f.xu.v v30, v29, v0.t

# Encoding: |100010|1|11111|00011|001|00000|1010111|
# CHECK: vfcvt.f.x.v v0, v31
# CHECK-SAME: [0x57,0x90,0xf1,0x8b]
vfcvt.f.x.v v0, v31
# Encoding: |100010|0|00001|00011|001|00010|1010111|
# CHECK: vfcvt.f.x.v v2, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x11,0x88]
vfcvt.f.x.v v2, v1, v0.t

# Encoding: |100010|1|00011|01000|001|00100|1010111|
# CHECK: vfwcvt.xu.f.v v4, v3
# CHECK-SAME: [0x57,0x12,0x34,0x8a]
vfwcvt.xu.f.v v4, v3
# Encoding: |100010|0|00101|01000|001|00110|1010111|
# CHECK: vfwcvt.xu.f.v v6, v5, v0.t
# CHECK-SAME: [0x57,0x13,0x54,0x88]
vfwcvt.xu.f.v v6, v5, v0.t

# Encoding: |100010|1|00111|01001|001|01000|1010111|
# CHECK: vfwcvt.x.f.v v8, v7
# CHECK-SAME: [0x57,0x94,0x74,0x8a]
vfwcvt.x.f.v v8, v7
# Encoding: |100010|0|01001|01001|001|01010|1010111|
# CHECK: vfwcvt.x.f.v v10, v9, v0.t
# CHECK-SAME: [0x57,0x95,0x94,0x88]
vfwcvt.x.f.v v10, v9, v0.t

# Encoding: |100010|1|01011|01010|001|01100|1010111|
# CHECK: vfwcvt.f.xu.v v12, v11
# CHECK-SAME: [0x57,0x16,0xb5,0x8a]
vfwcvt.f.xu.v v12, v11
# Encoding: |100010|0|01101|01010|001|01110|1010111|
# CHECK: vfwcvt.f.xu.v v14, v13, v0.t
# CHECK-SAME: [0x57,0x17,0xd5,0x88]
vfwcvt.f.xu.v v14, v13, v0.t

# Encoding: |100010|1|01111|01011|001|10000|1010111|
# CHECK: vfwcvt.f.x.v v16, v15
# CHECK-SAME: [0x57,0x98,0xf5,0x8a]
vfwcvt.f.x.v v16, v15
# Encoding: |100010|0|10001|01011|001|10010|1010111|
# CHECK: vfwcvt.f.x.v v18, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x15,0x89]
vfwcvt.f.x.v v18, v17, v0.t

# Encoding: |100010|1|10011|01100|001|10100|1010111|
# CHECK: vfwcvt.f.f.v v20, v19
# CHECK-SAME: [0x57,0x1a,0x36,0x8b]
vfwcvt.f.f.v v20, v19
# Encoding: |100010|0|10101|01100|001|10110|1010111|
# CHECK: vfwcvt.f.f.v v22, v21, v0.t
# CHECK-SAME: [0x57,0x1b,0x56,0x89]
vfwcvt.f.f.v v22, v21, v0.t

# Encoding: |100010|1|10111|10000|001|11000|1010111|
# CHECK: vfncvt.xu.f.v v24, v23
# CHECK-SAME: [0x57,0x1c,0x78,0x8b]
vfncvt.xu.f.v v24, v23
# Encoding: |100010|0|11001|10000|001|11010|1010111|
# CHECK: vfncvt.xu.f.v v26, v25, v0.t
# CHECK-SAME: [0x57,0x1d,0x98,0x89]
vfncvt.xu.f.v v26, v25, v0.t

# Encoding: |100010|1|11011|10001|001|11100|1010111|
# CHECK: vfncvt.x.f.v v28, v27
# CHECK-SAME: [0x57,0x9e,0xb8,0x8b]
vfncvt.x.f.v v28, v27
# Encoding: |100010|0|11101|10001|001|11110|1010111|
# CHECK: vfncvt.x.f.v v30, v29, v0.t
# CHECK-SAME: [0x57,0x9f,0xd8,0x89]
vfncvt.x.f.v v30, v29, v0.t

# Encoding: |100010|1|11111|10010|001|00000|1010111|
# CHECK: vfncvt.f.xu.v v0, v31
# CHECK-SAME: [0x57,0x10,0xf9,0x8b]
vfncvt.f.xu.v v0, v31
# Encoding: |100010|0|00001|10010|001|00010|1010111|
# CHECK: vfncvt.f.xu.v v2, v1, v0.t
# CHECK-SAME: [0x57,0x11,0x19,0x88]
vfncvt.f.xu.v v2, v1, v0.t

# Encoding: |100010|1|00011|10011|001|00100|1010111|
# CHECK: vfncvt.f.x.v v4, v3
# CHECK-SAME: [0x57,0x92,0x39,0x8a]
vfncvt.f.x.v v4, v3
# Encoding: |100010|0|00101|10011|001|00110|1010111|
# CHECK: vfncvt.f.x.v v6, v5, v0.t
# CHECK-SAME: [0x57,0x93,0x59,0x88]
vfncvt.f.x.v v6, v5, v0.t

# Encoding: |100010|1|00111|10100|001|01000|1010111|
# CHECK: vfncvt.f.f.v v8, v7
# CHECK-SAME: [0x57,0x14,0x7a,0x8a]
vfncvt.f.f.v v8, v7
# Encoding: |100010|0|01001|10100|001|01010|1010111|
# CHECK: vfncvt.f.f.v v10, v9, v0.t
# CHECK-SAME: [0x57,0x15,0x9a,0x88]
vfncvt.f.f.v v10, v9, v0.t

# Encoding: |010110|1|01011|00001|010|01100|1010111|
# CHECK: vmsbf.m v12, v11
# CHECK-SAME: [0x57,0xa6,0xb0,0x5a]
vmsbf.m v12, v11
# Encoding: |010110|0|01101|00001|010|01110|1010111|
# CHECK: vmsbf.m v14, v13, v0.t
# CHECK-SAME: [0x57,0xa7,0xd0,0x58]
vmsbf.m v14, v13, v0.t

# Encoding: |010110|1|01111|00010|010|10000|1010111|
# CHECK: vmsof.m v16, v15
# CHECK-SAME: [0x57,0x28,0xf1,0x5a]
vmsof.m v16, v15
# Encoding: |010110|0|10001|00010|010|10010|1010111|
# CHECK: vmsof.m v18, v17, v0.t
# CHECK-SAME: [0x57,0x29,0x11,0x59]
vmsof.m v18, v17, v0.t

# Encoding: |010110|1|10011|00011|010|10100|1010111|
# CHECK: vmsif.m v20, v19
# CHECK-SAME: [0x57,0xaa,0x31,0x5b]
vmsif.m v20, v19
# Encoding: |010110|0|10101|00011|010|10110|1010111|
# CHECK: vmsif.m v22, v21, v0.t
# CHECK-SAME: [0x57,0xab,0x51,0x59]
vmsif.m v22, v21, v0.t

# Encoding: |010110|1|10111|10000|010|11000|1010111|
# CHECK: vmiota.m v24, v23
# CHECK-SAME: [0x57,0x2c,0x78,0x5b]
vmiota.m v24, v23
# Encoding: |010110|0|11001|10000|010|11010|1010111|
# CHECK: vmiota.m v26, v25, v0.t
# CHECK-SAME: [0x57,0x2d,0x98,0x59]
vmiota.m v26, v25, v0.t

# Encoding: |010110|1|00000|10001|010|11011|1010111|
# CHECK: vid.v v27
# CHECK-SAME: [0xd7,0xad,0x08,0x5a]
vid.v v27
# Encoding: |010110|0|00000|10001|010|11100|1010111|
# CHECK: vid.v v28, v0.t
# CHECK-SAME: [0x57,0xae,0x08,0x58]
vid.v v28, v0.t

# Encoding: |000|100|1|00000|11000|000|11101|0000111|
# CHECK: vlb.v v29, (s8)
# CHECK-SAME: [0x87,0x0e,0x0c,0x12]
vlb.v v29, (s8)
# Encoding: |000|100|0|00000|11010|000|11110|0000111|
# CHECK: vlb.v v30, (s10), v0.t
# CHECK-SAME: [0x07,0x0f,0x0d,0x10]
vlb.v v30, (s10), v0.t

# Encoding: |000|100|1|00000|11100|101|11111|0000111|
# CHECK: vlh.v v31, (t3)
# CHECK-SAME: [0x87,0x5f,0x0e,0x12]
vlh.v v31, (t3)
# Encoding: |000|100|0|00000|11110|101|00000|0000111|
# CHECK: vlh.v v0, (t5), v0.t
# CHECK-SAME: [0x07,0x50,0x0f,0x10]
vlh.v v0, (t5), v0.t

# Encoding: |000|100|1|00000|00001|110|00001|0000111|
# CHECK: vlw.v v1, (ra)
# CHECK-SAME: [0x87,0xe0,0x00,0x12]
vlw.v v1, (ra)
# Encoding: |000|100|0|00000|00011|110|00010|0000111|
# CHECK: vlw.v v2, (gp), v0.t
# CHECK-SAME: [0x07,0xe1,0x01,0x10]
vlw.v v2, (gp), v0.t

# Encoding: |000|000|1|00000|00101|000|00011|0000111|
# CHECK: vlbu.v v3, (t0)
# CHECK-SAME: [0x87,0x81,0x02,0x02]
vlbu.v v3, (t0)
# Encoding: |000|000|0|00000|00111|000|00100|0000111|
# CHECK: vlbu.v v4, (t2), v0.t
# CHECK-SAME: [0x07,0x82,0x03,0x00]
vlbu.v v4, (t2), v0.t

# Encoding: |000|000|1|00000|01001|101|00101|0000111|
# CHECK: vlhu.v v5, (s1)
# CHECK-SAME: [0x87,0xd2,0x04,0x02]
vlhu.v v5, (s1)
# Encoding: |000|000|0|00000|01011|101|00110|0000111|
# CHECK: vlhu.v v6, (a1), v0.t
# CHECK-SAME: [0x07,0xd3,0x05,0x00]
vlhu.v v6, (a1), v0.t

# Encoding: |000|000|1|00000|01101|110|00111|0000111|
# CHECK: vlwu.v v7, (a3)
# CHECK-SAME: [0x87,0xe3,0x06,0x02]
vlwu.v v7, (a3)
# Encoding: |000|000|0|00000|01111|110|01000|0000111|
# CHECK: vlwu.v v8, (a5), v0.t
# CHECK-SAME: [0x07,0xe4,0x07,0x00]
vlwu.v v8, (a5), v0.t

# Encoding: |000|000|1|00000|10001|111|01001|0000111|
# CHECK: vle.v v9, (a7)
# CHECK-SAME: [0x87,0xf4,0x08,0x02]
vle.v v9, (a7)
# Encoding: |000|000|0|00000|10011|111|01010|0000111|
# CHECK: vle.v v10, (s3), v0.t
# CHECK-SAME: [0x07,0xf5,0x09,0x00]
vle.v v10, (s3), v0.t

# Encoding: |000|000|1|00000|10101|000|01011|0100111|
# CHECK: vsb.v v11, (s5)
# CHECK-SAME: [0xa7,0x85,0x0a,0x02]
vsb.v v11, (s5)
# Encoding: |000|000|0|00000|10111|000|01100|0100111|
# CHECK: vsb.v v12, (s7), v0.t
# CHECK-SAME: [0x27,0x86,0x0b,0x00]
vsb.v v12, (s7), v0.t

# Encoding: |000|000|1|00000|11001|101|01101|0100111|
# CHECK: vsh.v v13, (s9)
# CHECK-SAME: [0xa7,0xd6,0x0c,0x02]
vsh.v v13, (s9)
# Encoding: |000|000|0|00000|11011|101|01110|0100111|
# CHECK: vsh.v v14, (s11), v0.t
# CHECK-SAME: [0x27,0xd7,0x0d,0x00]
vsh.v v14, (s11), v0.t

# Encoding: |000|000|1|00000|11101|110|01111|0100111|
# CHECK: vsw.v v15, (t4)
# CHECK-SAME: [0xa7,0xe7,0x0e,0x02]
vsw.v v15, (t4)
# Encoding: |000|000|0|00000|11111|110|10000|0100111|
# CHECK: vsw.v v16, (t6), v0.t
# CHECK-SAME: [0x27,0xe8,0x0f,0x00]
vsw.v v16, (t6), v0.t

# Encoding: |000|000|1|00000|00010|111|10001|0100111|
# CHECK: vse.v v17, (sp)
# CHECK-SAME: [0xa7,0x78,0x01,0x02]
vse.v v17, (sp)
# Encoding: |000|000|0|00000|00100|111|10010|0100111|
# CHECK: vse.v v18, (tp), v0.t
# CHECK-SAME: [0x27,0x79,0x02,0x00]
vse.v v18, (tp), v0.t

# Encoding: |000|110|1|00110|01000|000|10011|0000111|
# CHECK: vlsb.v v19, (s0), t1
# CHECK-SAME: [0x87,0x09,0x64,0x1a]
vlsb.v v19, (s0), t1
# Encoding: |000|110|0|01010|01100|000|10100|0000111|
# CHECK: vlsb.v v20, (a2), a0, v0.t
# CHECK-SAME: [0x07,0x0a,0xa6,0x18]
vlsb.v v20, (a2), a0, v0.t

# Encoding: |000|110|1|01110|10000|101|10101|0000111|
# CHECK: vlsh.v v21, (a6), a4
# CHECK-SAME: [0x87,0x5a,0xe8,0x1a]
vlsh.v v21, (a6), a4
# Encoding: |000|110|0|10010|10100|101|10110|0000111|
# CHECK: vlsh.v v22, (s4), s2, v0.t
# CHECK-SAME: [0x07,0x5b,0x2a,0x19]
vlsh.v v22, (s4), s2, v0.t

# Encoding: |000|110|1|10110|11000|110|10111|0000111|
# CHECK: vlsw.v v23, (s8), s6
# CHECK-SAME: [0x87,0x6b,0x6c,0x1b]
vlsw.v v23, (s8), s6
# Encoding: |000|110|0|11010|11100|110|11000|0000111|
# CHECK: vlsw.v v24, (t3), s10, v0.t
# CHECK-SAME: [0x07,0x6c,0xae,0x19]
vlsw.v v24, (t3), s10, v0.t

# Encoding: |000|010|1|11110|00001|000|11001|0000111|
# CHECK: vlsbu.v v25, (ra), t5
# CHECK-SAME: [0x87,0x8c,0xe0,0x0b]
vlsbu.v v25, (ra), t5
# Encoding: |000|010|0|00011|00101|000|11010|0000111|
# CHECK: vlsbu.v v26, (t0), gp, v0.t
# CHECK-SAME: [0x07,0x8d,0x32,0x08]
vlsbu.v v26, (t0), gp, v0.t

# Encoding: |000|010|1|00111|01001|101|11011|0000111|
# CHECK: vlshu.v v27, (s1), t2
# CHECK-SAME: [0x87,0xdd,0x74,0x0a]
vlshu.v v27, (s1), t2
# Encoding: |000|010|0|01011|01101|101|11100|0000111|
# CHECK: vlshu.v v28, (a3), a1, v0.t
# CHECK-SAME: [0x07,0xde,0xb6,0x08]
vlshu.v v28, (a3), a1, v0.t

# Encoding: |000|010|1|01111|10001|110|11101|0000111|
# CHECK: vlswu.v v29, (a7), a5
# CHECK-SAME: [0x87,0xee,0xf8,0x0a]
vlswu.v v29, (a7), a5
# Encoding: |000|010|0|10011|10101|110|11110|0000111|
# CHECK: vlswu.v v30, (s5), s3, v0.t
# CHECK-SAME: [0x07,0xef,0x3a,0x09]
vlswu.v v30, (s5), s3, v0.t

# Encoding: |000|010|1|10111|11001|111|11111|0000111|
# CHECK: vlse.v v31, (s9), s7
# CHECK-SAME: [0x87,0xff,0x7c,0x0b]
vlse.v v31, (s9), s7
# Encoding: |000|010|0|11011|11101|111|00000|0000111|
# CHECK: vlse.v v0, (t4), s11, v0.t
# CHECK-SAME: [0x07,0xf0,0xbe,0x09]
vlse.v v0, (t4), s11, v0.t

# Encoding: |000|010|1|11111|00010|000|00001|0100111|
# CHECK: vssb.v v1, (sp), t6
# CHECK-SAME: [0xa7,0x00,0xf1,0x0b]
vssb.v v1, (sp), t6
# Encoding: |000|010|0|00100|00110|000|00010|0100111|
# CHECK: vssb.v v2, (t1), tp, v0.t
# CHECK-SAME: [0x27,0x01,0x43,0x08]
vssb.v v2, (t1), tp, v0.t

# Encoding: |000|010|1|01000|01010|101|00011|0100111|
# CHECK: vssh.v v3, (a0), s0
# CHECK-SAME: [0xa7,0x51,0x85,0x0a]
vssh.v v3, (a0), s0
# Encoding: |000|010|0|01100|01110|101|00100|0100111|
# CHECK: vssh.v v4, (a4), a2, v0.t
# CHECK-SAME: [0x27,0x52,0xc7,0x08]
vssh.v v4, (a4), a2, v0.t

# Encoding: |000|010|1|10000|10010|110|00101|0100111|
# CHECK: vssw.v v5, (s2), a6
# CHECK-SAME: [0xa7,0x62,0x09,0x0b]
vssw.v v5, (s2), a6
# Encoding: |000|010|0|10100|10110|110|00110|0100111|
# CHECK: vssw.v v6, (s6), s4, v0.t
# CHECK-SAME: [0x27,0x63,0x4b,0x09]
vssw.v v6, (s6), s4, v0.t

# Encoding: |000|010|1|11000|11010|111|00111|0100111|
# CHECK: vsse.v v7, (s10), s8
# CHECK-SAME: [0xa7,0x73,0x8d,0x0b]
vsse.v v7, (s10), s8
# Encoding: |000|010|0|11100|11110|111|01000|0100111|
# CHECK: vsse.v v8, (t5), t3, v0.t
# CHECK-SAME: [0x27,0x74,0xcf,0x09]
vsse.v v8, (t5), t3, v0.t

# Encoding: |000|111|1|01001|00001|000|01010|0000111|
# CHECK: vlxb.v v10, (ra), v9
# CHECK-SAME: [0x07,0x85,0x90,0x1e]
vlxb.v v10, (ra), v9
# Encoding: |000|111|0|01011|00011|000|01100|0000111|
# CHECK: vlxb.v v12, (gp), v11, v0.t
# CHECK-SAME: [0x07,0x86,0xb1,0x1c]
vlxb.v v12, (gp), v11, v0.t

# Encoding: |000|111|1|01101|00101|101|01110|0000111|
# CHECK: vlxh.v v14, (t0), v13
# CHECK-SAME: [0x07,0xd7,0xd2,0x1e]
vlxh.v v14, (t0), v13
# Encoding: |000|111|0|01111|00111|101|10000|0000111|
# CHECK: vlxh.v v16, (t2), v15, v0.t
# CHECK-SAME: [0x07,0xd8,0xf3,0x1c]
vlxh.v v16, (t2), v15, v0.t

# Encoding: |000|111|1|10001|01001|110|10010|0000111|
# CHECK: vlxw.v v18, (s1), v17
# CHECK-SAME: [0x07,0xe9,0x14,0x1f]
vlxw.v v18, (s1), v17
# Encoding: |000|111|0|10011|01011|110|10100|0000111|
# CHECK: vlxw.v v20, (a1), v19, v0.t
# CHECK-SAME: [0x07,0xea,0x35,0x1d]
vlxw.v v20, (a1), v19, v0.t

# Encoding: |000|011|1|10101|01101|000|10110|0000111|
# CHECK: vlxbu.v v22, (a3), v21
# CHECK-SAME: [0x07,0x8b,0x56,0x0f]
vlxbu.v v22, (a3), v21
# Encoding: |000|011|0|10111|01111|000|11000|0000111|
# CHECK: vlxbu.v v24, (a5), v23, v0.t
# CHECK-SAME: [0x07,0x8c,0x77,0x0d]
vlxbu.v v24, (a5), v23, v0.t

# Encoding: |000|011|1|11001|10001|101|11010|0000111|
# CHECK: vlxhu.v v26, (a7), v25
# CHECK-SAME: [0x07,0xdd,0x98,0x0f]
vlxhu.v v26, (a7), v25
# Encoding: |000|011|0|11011|10011|101|11100|0000111|
# CHECK: vlxhu.v v28, (s3), v27, v0.t
# CHECK-SAME: [0x07,0xde,0xb9,0x0d]
vlxhu.v v28, (s3), v27, v0.t

# Encoding: |000|011|1|11101|10101|110|11110|0000111|
# CHECK: vlxwu.v v30, (s5), v29
# CHECK-SAME: [0x07,0xef,0xda,0x0f]
vlxwu.v v30, (s5), v29
# Encoding: |000|011|0|11111|10111|110|00000|0000111|
# CHECK: vlxwu.v v0, (s7), v31, v0.t
# CHECK-SAME: [0x07,0xe0,0xfb,0x0d]
vlxwu.v v0, (s7), v31, v0.t

# Encoding: |000|011|1|00001|11001|111|00010|0000111|
# CHECK: vlxe.v v2, (s9), v1
# CHECK-SAME: [0x07,0xf1,0x1c,0x0e]
vlxe.v v2, (s9), v1
# Encoding: |000|011|0|00011|11011|111|00100|0000111|
# CHECK: vlxe.v v4, (s11), v3, v0.t
# CHECK-SAME: [0x07,0xf2,0x3d,0x0c]
vlxe.v v4, (s11), v3, v0.t

# Encoding: |000|011|1|00101|11101|000|00110|0100111|
# CHECK: vsxb.v v6, (t4), v5
# CHECK-SAME: [0x27,0x83,0x5e,0x0e]
vsxb.v v6, (t4), v5
# Encoding: |000|011|0|00111|11111|000|01000|0100111|
# CHECK: vsxb.v v8, (t6), v7, v0.t
# CHECK-SAME: [0x27,0x84,0x7f,0x0c]
vsxb.v v8, (t6), v7, v0.t

# Encoding: |000|011|1|01001|00010|101|01010|0100111|
# CHECK: vsxh.v v10, (sp), v9
# CHECK-SAME: [0x27,0x55,0x91,0x0e]
vsxh.v v10, (sp), v9
# Encoding: |000|011|0|01011|00100|101|01100|0100111|
# CHECK: vsxh.v v12, (tp), v11, v0.t
# CHECK-SAME: [0x27,0x56,0xb2,0x0c]
vsxh.v v12, (tp), v11, v0.t

# Encoding: |000|011|1|01101|00110|110|01110|0100111|
# CHECK: vsxw.v v14, (t1), v13
# CHECK-SAME: [0x27,0x67,0xd3,0x0e]
vsxw.v v14, (t1), v13
# Encoding: |000|011|0|01111|01000|110|10000|0100111|
# CHECK: vsxw.v v16, (s0), v15, v0.t
# CHECK-SAME: [0x27,0x68,0xf4,0x0c]
vsxw.v v16, (s0), v15, v0.t

# Encoding: |000|011|1|10001|01010|111|10010|0100111|
# CHECK: vsxe.v v18, (a0), v17
# CHECK-SAME: [0x27,0x79,0x15,0x0f]
vsxe.v v18, (a0), v17
# Encoding: |000|011|0|10011|01100|111|10100|0100111|
# CHECK: vsxe.v v20, (a2), v19, v0.t
# CHECK-SAME: [0x27,0x7a,0x36,0x0d]
vsxe.v v20, (a2), v19, v0.t

# Encoding: |000|111|1|10101|01110|000|10110|0100111|
# CHECK: vsuxb.v v22, (a4), v21
# CHECK-SAME: [0x27,0x0b,0x57,0x1f]
vsuxb.v v22, (a4), v21
# Encoding: |000|111|0|10111|10000|000|11000|0100111|
# CHECK: vsuxb.v v24, (a6), v23, v0.t
# CHECK-SAME: [0x27,0x0c,0x78,0x1d]
vsuxb.v v24, (a6), v23, v0.t

# Encoding: |000|111|1|11001|10010|101|11010|0100111|
# CHECK: vsuxh.v v26, (s2), v25
# CHECK-SAME: [0x27,0x5d,0x99,0x1f]
vsuxh.v v26, (s2), v25
# Encoding: |000|111|0|11011|10100|101|11100|0100111|
# CHECK: vsuxh.v v28, (s4), v27, v0.t
# CHECK-SAME: [0x27,0x5e,0xba,0x1d]
vsuxh.v v28, (s4), v27, v0.t

# Encoding: |000|111|1|11101|10110|110|11110|0100111|
# CHECK: vsuxw.v v30, (s6), v29
# CHECK-SAME: [0x27,0x6f,0xdb,0x1f]
vsuxw.v v30, (s6), v29
# Encoding: |000|111|0|11111|11000|110|00000|0100111|
# CHECK: vsuxw.v v0, (s8), v31, v0.t
# CHECK-SAME: [0x27,0x60,0xfc,0x1d]
vsuxw.v v0, (s8), v31, v0.t

# Encoding: |000|111|1|00001|11010|111|00010|0100111|
# CHECK: vsuxe.v v2, (s10), v1
# CHECK-SAME: [0x27,0x71,0x1d,0x1e]
vsuxe.v v2, (s10), v1
# Encoding: |000|111|0|00011|11100|111|00100|0100111|
# CHECK: vsuxe.v v4, (t3), v3, v0.t
# CHECK-SAME: [0x27,0x72,0x3e,0x1c]
vsuxe.v v4, (t3), v3, v0.t

