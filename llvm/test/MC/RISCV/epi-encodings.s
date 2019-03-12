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
# CHECK: vadc.vv v18, v16, v17
# CHECK-SAME: [0x57,0x89,0x08,0x43]
vadc.vv v18, v16, v17

# Encoding: |010000|1|10011|10110|100|10100|1010111|
# CHECK: vadc.vx v20, v19, s6
# CHECK-SAME: [0x57,0x4a,0x3b,0x43]
vadc.vx v20, v19, s6

# Encoding: |010000|1|10101|10000|011|10110|1010111|
# CHECK: vadc.vi v22, v21, -16
# CHECK-SAME: [0x57,0x3b,0x58,0x43]
vadc.vi v22, v21, -16

# Encoding: |010010|1|10111|11000|000|11001|1010111|
# CHECK: vsbc.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x0c,0x7c,0x4b]
vsbc.vv v25, v23, v24

# Encoding: |010010|1|11010|11000|100|11011|1010111|
# CHECK: vsbc.vx v27, v26, s8
# CHECK-SAME: [0xd7,0x4d,0xac,0x4b]
vsbc.vx v27, v26, s8

# Encoding: |010111|1|11100|11101|000|11110|1010111|
# CHECK: vmerge.vv v30, v28, v29
# CHECK-SAME: [0x57,0x8f,0xce,0x5f]
vmerge.vv v30, v28, v29
# Encoding: |010111|0|11111|00000|000|00001|1010111|
# CHECK: vmerge.vv v1, v31, v0, v0.t
# CHECK-SAME: [0xd7,0x00,0xf0,0x5d]
vmerge.vv v1, v31, v0, v0.t

# Encoding: |010111|1|00010|11010|100|00011|1010111|
# CHECK: vmerge.vx v3, v2, s10
# CHECK-SAME: [0xd7,0x41,0x2d,0x5e]
vmerge.vx v3, v2, s10
# Encoding: |010111|0|00100|11100|100|00101|1010111|
# CHECK: vmerge.vx v5, v4, t3, v0.t
# CHECK-SAME: [0xd7,0x42,0x4e,0x5c]
vmerge.vx v5, v4, t3, v0.t

# Encoding: |010111|1|00110|10001|011|00111|1010111|
# CHECK: vmerge.vi v7, v6, -15
# CHECK-SAME: [0xd7,0xb3,0x68,0x5e]
vmerge.vi v7, v6, -15
# Encoding: |010111|0|01000|10010|011|01001|1010111|
# CHECK: vmerge.vi v9, v8, -14, v0.t
# CHECK-SAME: [0xd7,0x34,0x89,0x5c]
vmerge.vi v9, v8, -14, v0.t

# Encoding: |011000|1|01010|01011|000|01100|1010111|
# CHECK: vseq.vv v12, v10, v11
# CHECK-SAME: [0x57,0x86,0xa5,0x62]
vseq.vv v12, v10, v11
# Encoding: |011000|0|01101|01110|000|01111|1010111|
# CHECK: vseq.vv v15, v13, v14, v0.t
# CHECK-SAME: [0xd7,0x07,0xd7,0x60]
vseq.vv v15, v13, v14, v0.t

# Encoding: |011000|1|10000|11110|100|10001|1010111|
# CHECK: vseq.vx v17, v16, t5
# CHECK-SAME: [0xd7,0x48,0x0f,0x63]
vseq.vx v17, v16, t5
# Encoding: |011000|0|10010|00001|100|10011|1010111|
# CHECK: vseq.vx v19, v18, ra, v0.t
# CHECK-SAME: [0xd7,0xc9,0x20,0x61]
vseq.vx v19, v18, ra, v0.t

# Encoding: |011000|1|10100|10011|011|10101|1010111|
# CHECK: vseq.vi v21, v20, -13
# CHECK-SAME: [0xd7,0xba,0x49,0x63]
vseq.vi v21, v20, -13
# Encoding: |011000|0|10110|10100|011|10111|1010111|
# CHECK: vseq.vi v23, v22, -12, v0.t
# CHECK-SAME: [0xd7,0x3b,0x6a,0x61]
vseq.vi v23, v22, -12, v0.t

# Encoding: |011001|1|11000|11001|000|11010|1010111|
# CHECK: vsne.vv v26, v24, v25
# CHECK-SAME: [0x57,0x8d,0x8c,0x67]
vsne.vv v26, v24, v25
# Encoding: |011001|0|11011|11100|000|11101|1010111|
# CHECK: vsne.vv v29, v27, v28, v0.t
# CHECK-SAME: [0xd7,0x0e,0xbe,0x65]
vsne.vv v29, v27, v28, v0.t

# Encoding: |011001|1|11110|00011|100|11111|1010111|
# CHECK: vsne.vx v31, v30, gp
# CHECK-SAME: [0xd7,0xcf,0xe1,0x67]
vsne.vx v31, v30, gp
# Encoding: |011001|0|00000|00101|100|00001|1010111|
# CHECK: vsne.vx v1, v0, t0, v0.t
# CHECK-SAME: [0xd7,0xc0,0x02,0x64]
vsne.vx v1, v0, t0, v0.t

# Encoding: |011001|1|00010|10101|011|00011|1010111|
# CHECK: vsne.vi v3, v2, -11
# CHECK-SAME: [0xd7,0xb1,0x2a,0x66]
vsne.vi v3, v2, -11
# Encoding: |011001|0|00100|10110|011|00101|1010111|
# CHECK: vsne.vi v5, v4, -10, v0.t
# CHECK-SAME: [0xd7,0x32,0x4b,0x64]
vsne.vi v5, v4, -10, v0.t

# Encoding: |011010|1|00110|00111|000|01000|1010111|
# CHECK: vsltu.vv v8, v6, v7
# CHECK-SAME: [0x57,0x84,0x63,0x6a]
vsltu.vv v8, v6, v7
# Encoding: |011010|0|01001|01010|000|01011|1010111|
# CHECK: vsltu.vv v11, v9, v10, v0.t
# CHECK-SAME: [0xd7,0x05,0x95,0x68]
vsltu.vv v11, v9, v10, v0.t

# Encoding: |011010|1|01100|00111|100|01101|1010111|
# CHECK: vsltu.vx v13, v12, t2
# CHECK-SAME: [0xd7,0xc6,0xc3,0x6a]
vsltu.vx v13, v12, t2
# Encoding: |011010|0|01110|01001|100|01111|1010111|
# CHECK: vsltu.vx v15, v14, s1, v0.t
# CHECK-SAME: [0xd7,0xc7,0xe4,0x68]
vsltu.vx v15, v14, s1, v0.t

# Encoding: |011011|1|10000|10001|000|10010|1010111|
# CHECK: vslt.vv v18, v16, v17
# CHECK-SAME: [0x57,0x89,0x08,0x6f]
vslt.vv v18, v16, v17
# Encoding: |011011|0|10011|10100|000|10101|1010111|
# CHECK: vslt.vv v21, v19, v20, v0.t
# CHECK-SAME: [0xd7,0x0a,0x3a,0x6d]
vslt.vv v21, v19, v20, v0.t

# Encoding: |011011|1|10110|01011|100|10111|1010111|
# CHECK: vslt.vx v23, v22, a1
# CHECK-SAME: [0xd7,0xcb,0x65,0x6f]
vslt.vx v23, v22, a1
# Encoding: |011011|0|11000|01101|100|11001|1010111|
# CHECK: vslt.vx v25, v24, a3, v0.t
# CHECK-SAME: [0xd7,0xcc,0x86,0x6d]
vslt.vx v25, v24, a3, v0.t

# Encoding: |011100|1|11010|11011|000|11100|1010111|
# CHECK: vsleu.vv v28, v26, v27
# CHECK-SAME: [0x57,0x8e,0xad,0x73]
vsleu.vv v28, v26, v27
# Encoding: |011100|0|11101|11110|000|11111|1010111|
# CHECK: vsleu.vv v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x0f,0xdf,0x71]
vsleu.vv v31, v29, v30, v0.t

# Encoding: |011100|1|00000|01111|100|00001|1010111|
# CHECK: vsleu.vx v1, v0, a5
# CHECK-SAME: [0xd7,0xc0,0x07,0x72]
vsleu.vx v1, v0, a5
# Encoding: |011100|0|00010|10001|100|00011|1010111|
# CHECK: vsleu.vx v3, v2, a7, v0.t
# CHECK-SAME: [0xd7,0xc1,0x28,0x70]
vsleu.vx v3, v2, a7, v0.t

# Encoding: |011100|1|00100|10111|011|00101|1010111|
# CHECK: vsleu.vi v5, v4, -9
# CHECK-SAME: [0xd7,0xb2,0x4b,0x72]
vsleu.vi v5, v4, -9
# Encoding: |011100|0|00110|11000|011|00111|1010111|
# CHECK: vsleu.vi v7, v6, -8, v0.t
# CHECK-SAME: [0xd7,0x33,0x6c,0x70]
vsleu.vi v7, v6, -8, v0.t

# Encoding: |011101|1|01000|01001|000|01010|1010111|
# CHECK: vsle.vv v10, v8, v9
# CHECK-SAME: [0x57,0x85,0x84,0x76]
vsle.vv v10, v8, v9
# Encoding: |011101|0|01011|01100|000|01101|1010111|
# CHECK: vsle.vv v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x06,0xb6,0x74]
vsle.vv v13, v11, v12, v0.t

# Encoding: |011101|1|01110|10011|100|01111|1010111|
# CHECK: vsle.vx v15, v14, s3
# CHECK-SAME: [0xd7,0xc7,0xe9,0x76]
vsle.vx v15, v14, s3
# Encoding: |011101|0|10000|10101|100|10001|1010111|
# CHECK: vsle.vx v17, v16, s5, v0.t
# CHECK-SAME: [0xd7,0xc8,0x0a,0x75]
vsle.vx v17, v16, s5, v0.t

# Encoding: |011101|1|10010|11001|011|10011|1010111|
# CHECK: vsle.vi v19, v18, -7
# CHECK-SAME: [0xd7,0xb9,0x2c,0x77]
vsle.vi v19, v18, -7
# Encoding: |011101|0|10100|11010|011|10101|1010111|
# CHECK: vsle.vi v21, v20, -6, v0.t
# CHECK-SAME: [0xd7,0x3a,0x4d,0x75]
vsle.vi v21, v20, -6, v0.t

# Encoding: |011110|1|10110|10111|100|10111|1010111|
# CHECK: vsgtu.vx v23, v22, s7
# CHECK-SAME: [0xd7,0xcb,0x6b,0x7b]
vsgtu.vx v23, v22, s7
# Encoding: |011110|0|11000|11001|100|11001|1010111|
# CHECK: vsgtu.vx v25, v24, s9, v0.t
# CHECK-SAME: [0xd7,0xcc,0x8c,0x79]
vsgtu.vx v25, v24, s9, v0.t

# Encoding: |011110|1|11010|11011|011|11011|1010111|
# CHECK: vsgtu.vi v27, v26, -5
# CHECK-SAME: [0xd7,0xbd,0xad,0x7b]
vsgtu.vi v27, v26, -5
# Encoding: |011110|0|11100|11100|011|11101|1010111|
# CHECK: vsgtu.vi v29, v28, -4, v0.t
# CHECK-SAME: [0xd7,0x3e,0xce,0x79]
vsgtu.vi v29, v28, -4, v0.t

# Encoding: |011111|1|11110|11011|100|11111|1010111|
# CHECK: vsgt.vx v31, v30, s11
# CHECK-SAME: [0xd7,0xcf,0xed,0x7f]
vsgt.vx v31, v30, s11
# Encoding: |011111|0|00000|11101|100|00001|1010111|
# CHECK: vsgt.vx v1, v0, t4, v0.t
# CHECK-SAME: [0xd7,0xc0,0x0e,0x7c]
vsgt.vx v1, v0, t4, v0.t

# Encoding: |011111|1|00010|11101|011|00011|1010111|
# CHECK: vsgt.vi v3, v2, -3
# CHECK-SAME: [0xd7,0xb1,0x2e,0x7e]
vsgt.vi v3, v2, -3
# Encoding: |011111|0|00100|11110|011|00101|1010111|
# CHECK: vsgt.vi v5, v4, -2, v0.t
# CHECK-SAME: [0xd7,0x32,0x4f,0x7c]
vsgt.vi v5, v4, -2, v0.t

# Encoding: |100000|1|00110|00111|000|01000|1010111|
# CHECK: vsaddu.vv v8, v6, v7
# CHECK-SAME: [0x57,0x84,0x63,0x82]
vsaddu.vv v8, v6, v7
# Encoding: |100000|0|01001|01010|000|01011|1010111|
# CHECK: vsaddu.vv v11, v9, v10, v0.t
# CHECK-SAME: [0xd7,0x05,0x95,0x80]
vsaddu.vv v11, v9, v10, v0.t

# Encoding: |100000|1|01100|11111|100|01101|1010111|
# CHECK: vsaddu.vx v13, v12, t6
# CHECK-SAME: [0xd7,0xc6,0xcf,0x82]
vsaddu.vx v13, v12, t6
# Encoding: |100000|0|01110|00010|100|01111|1010111|
# CHECK: vsaddu.vx v15, v14, sp, v0.t
# CHECK-SAME: [0xd7,0x47,0xe1,0x80]
vsaddu.vx v15, v14, sp, v0.t

# Encoding: |100000|1|10000|11111|011|10001|1010111|
# CHECK: vsaddu.vi v17, v16, -1
# CHECK-SAME: [0xd7,0xb8,0x0f,0x83]
vsaddu.vi v17, v16, -1
# Encoding: |100000|0|10010|00000|011|10011|1010111|
# CHECK: vsaddu.vi v19, v18, 0, v0.t
# CHECK-SAME: [0xd7,0x39,0x20,0x81]
vsaddu.vi v19, v18, 0, v0.t

# Encoding: |100001|1|10100|10101|000|10110|1010111|
# CHECK: vsadd.vv v22, v20, v21
# CHECK-SAME: [0x57,0x8b,0x4a,0x87]
vsadd.vv v22, v20, v21
# Encoding: |100001|0|10111|11000|000|11001|1010111|
# CHECK: vsadd.vv v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x0c,0x7c,0x85]
vsadd.vv v25, v23, v24, v0.t

# Encoding: |100001|1|11010|00100|100|11011|1010111|
# CHECK: vsadd.vx v27, v26, tp
# CHECK-SAME: [0xd7,0x4d,0xa2,0x87]
vsadd.vx v27, v26, tp
# Encoding: |100001|0|11100|00110|100|11101|1010111|
# CHECK: vsadd.vx v29, v28, t1, v0.t
# CHECK-SAME: [0xd7,0x4e,0xc3,0x85]
vsadd.vx v29, v28, t1, v0.t

# Encoding: |100001|1|11110|00001|011|11111|1010111|
# CHECK: vsadd.vi v31, v30, 1
# CHECK-SAME: [0xd7,0xbf,0xe0,0x87]
vsadd.vi v31, v30, 1
# Encoding: |100001|0|00000|00010|011|00001|1010111|
# CHECK: vsadd.vi v1, v0, 2, v0.t
# CHECK-SAME: [0xd7,0x30,0x01,0x84]
vsadd.vi v1, v0, 2, v0.t

# Encoding: |100010|1|00010|00011|000|00100|1010111|
# CHECK: vssubu.vv v4, v2, v3
# CHECK-SAME: [0x57,0x82,0x21,0x8a]
vssubu.vv v4, v2, v3
# Encoding: |100010|0|00101|00110|000|00111|1010111|
# CHECK: vssubu.vv v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x03,0x53,0x88]
vssubu.vv v7, v5, v6, v0.t

# Encoding: |100010|1|01000|01000|100|01001|1010111|
# CHECK: vssubu.vx v9, v8, s0
# CHECK-SAME: [0xd7,0x44,0x84,0x8a]
vssubu.vx v9, v8, s0
# Encoding: |100010|0|01010|01010|100|01011|1010111|
# CHECK: vssubu.vx v11, v10, a0, v0.t
# CHECK-SAME: [0xd7,0x45,0xa5,0x88]
vssubu.vx v11, v10, a0, v0.t

# Encoding: |100011|1|01100|01101|000|01110|1010111|
# CHECK: vssub.vv v14, v12, v13
# CHECK-SAME: [0x57,0x87,0xc6,0x8e]
vssub.vv v14, v12, v13
# Encoding: |100011|0|01111|10000|000|10001|1010111|
# CHECK: vssub.vv v17, v15, v16, v0.t
# CHECK-SAME: [0xd7,0x08,0xf8,0x8c]
vssub.vv v17, v15, v16, v0.t

# Encoding: |100011|1|10010|01100|100|10011|1010111|
# CHECK: vssub.vx v19, v18, a2
# CHECK-SAME: [0xd7,0x49,0x26,0x8f]
vssub.vx v19, v18, a2
# Encoding: |100011|0|10100|01110|100|10101|1010111|
# CHECK: vssub.vx v21, v20, a4, v0.t
# CHECK-SAME: [0xd7,0x4a,0x47,0x8d]
vssub.vx v21, v20, a4, v0.t

# Encoding: |100100|1|10110|10111|000|11000|1010111|
# CHECK: vaadd.vv v24, v22, v23
# CHECK-SAME: [0x57,0x8c,0x6b,0x93]
vaadd.vv v24, v22, v23
# Encoding: |100100|0|11001|11010|000|11011|1010111|
# CHECK: vaadd.vv v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x0d,0x9d,0x91]
vaadd.vv v27, v25, v26, v0.t

# Encoding: |100100|1|11100|10000|100|11101|1010111|
# CHECK: vaadd.vx v29, v28, a6
# CHECK-SAME: [0xd7,0x4e,0xc8,0x93]
vaadd.vx v29, v28, a6
# Encoding: |100100|0|11110|10010|100|11111|1010111|
# CHECK: vaadd.vx v31, v30, s2, v0.t
# CHECK-SAME: [0xd7,0x4f,0xe9,0x91]
vaadd.vx v31, v30, s2, v0.t

# Encoding: |100100|1|00000|00011|011|00001|1010111|
# CHECK: vaadd.vi v1, v0, 3
# CHECK-SAME: [0xd7,0xb0,0x01,0x92]
vaadd.vi v1, v0, 3
# Encoding: |100100|0|00010|00100|011|00011|1010111|
# CHECK: vaadd.vi v3, v2, 4, v0.t
# CHECK-SAME: [0xd7,0x31,0x22,0x90]
vaadd.vi v3, v2, 4, v0.t

# Encoding: |100101|1|00100|00101|000|00110|1010111|
# CHECK: vsll.vv v6, v4, v5
# CHECK-SAME: [0x57,0x83,0x42,0x96]
vsll.vv v6, v4, v5
# Encoding: |100101|0|00111|01000|000|01001|1010111|
# CHECK: vsll.vv v9, v7, v8, v0.t
# CHECK-SAME: [0xd7,0x04,0x74,0x94]
vsll.vv v9, v7, v8, v0.t

# Encoding: |100101|1|01010|10100|100|01011|1010111|
# CHECK: vsll.vx v11, v10, s4
# CHECK-SAME: [0xd7,0x45,0xaa,0x96]
vsll.vx v11, v10, s4
# Encoding: |100101|0|01100|10110|100|01101|1010111|
# CHECK: vsll.vx v13, v12, s6, v0.t
# CHECK-SAME: [0xd7,0x46,0xcb,0x94]
vsll.vx v13, v12, s6, v0.t

# Encoding: |100101|1|01110|00101|011|01111|1010111|
# CHECK: vsll.vi v15, v14, 5
# CHECK-SAME: [0xd7,0xb7,0xe2,0x96]
vsll.vi v15, v14, 5
# Encoding: |100101|0|10000|00110|011|10001|1010111|
# CHECK: vsll.vi v17, v16, 6, v0.t
# CHECK-SAME: [0xd7,0x38,0x03,0x95]
vsll.vi v17, v16, 6, v0.t

# Encoding: |100110|1|10010|10011|000|10100|1010111|
# CHECK: vasub.vv v20, v18, v19
# CHECK-SAME: [0x57,0x8a,0x29,0x9b]
vasub.vv v20, v18, v19
# Encoding: |100110|0|10101|10110|000|10111|1010111|
# CHECK: vasub.vv v23, v21, v22, v0.t
# CHECK-SAME: [0xd7,0x0b,0x5b,0x99]
vasub.vv v23, v21, v22, v0.t

# Encoding: |100110|1|11000|11000|100|11001|1010111|
# CHECK: vasub.vx v25, v24, s8
# CHECK-SAME: [0xd7,0x4c,0x8c,0x9b]
vasub.vx v25, v24, s8
# Encoding: |100110|0|11010|11010|100|11011|1010111|
# CHECK: vasub.vx v27, v26, s10, v0.t
# CHECK-SAME: [0xd7,0x4d,0xad,0x99]
vasub.vx v27, v26, s10, v0.t

# Encoding: |100111|1|11100|11101|000|11110|1010111|
# CHECK: vsmul.vv v30, v28, v29
# CHECK-SAME: [0x57,0x8f,0xce,0x9f]
vsmul.vv v30, v28, v29
# Encoding: |100111|0|11111|00000|000|00001|1010111|
# CHECK: vsmul.vv v1, v31, v0, v0.t
# CHECK-SAME: [0xd7,0x00,0xf0,0x9d]
vsmul.vv v1, v31, v0, v0.t

# Encoding: |100111|1|00010|11100|100|00011|1010111|
# CHECK: vsmul.vx v3, v2, t3
# CHECK-SAME: [0xd7,0x41,0x2e,0x9e]
vsmul.vx v3, v2, t3
# Encoding: |100111|0|00100|11110|100|00101|1010111|
# CHECK: vsmul.vx v5, v4, t5, v0.t
# CHECK-SAME: [0xd7,0x42,0x4f,0x9c]
vsmul.vx v5, v4, t5, v0.t

# Encoding: |101000|1|00110|00111|000|01000|1010111|
# CHECK: vsrl.vv v8, v6, v7
# CHECK-SAME: [0x57,0x84,0x63,0xa2]
vsrl.vv v8, v6, v7
# Encoding: |101000|0|01001|01010|000|01011|1010111|
# CHECK: vsrl.vv v11, v9, v10, v0.t
# CHECK-SAME: [0xd7,0x05,0x95,0xa0]
vsrl.vv v11, v9, v10, v0.t

# Encoding: |101000|1|01100|00001|100|01101|1010111|
# CHECK: vsrl.vx v13, v12, ra
# CHECK-SAME: [0xd7,0xc6,0xc0,0xa2]
vsrl.vx v13, v12, ra
# Encoding: |101000|0|01110|00011|100|01111|1010111|
# CHECK: vsrl.vx v15, v14, gp, v0.t
# CHECK-SAME: [0xd7,0xc7,0xe1,0xa0]
vsrl.vx v15, v14, gp, v0.t

# Encoding: |101000|1|10000|00111|011|10001|1010111|
# CHECK: vsrl.vi v17, v16, 7
# CHECK-SAME: [0xd7,0xb8,0x03,0xa3]
vsrl.vi v17, v16, 7
# Encoding: |101000|0|10010|01000|011|10011|1010111|
# CHECK: vsrl.vi v19, v18, 8, v0.t
# CHECK-SAME: [0xd7,0x39,0x24,0xa1]
vsrl.vi v19, v18, 8, v0.t

# Encoding: |101001|1|10100|10101|000|10110|1010111|
# CHECK: vsra.vv v22, v20, v21
# CHECK-SAME: [0x57,0x8b,0x4a,0xa7]
vsra.vv v22, v20, v21
# Encoding: |101001|0|10111|11000|000|11001|1010111|
# CHECK: vsra.vv v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x0c,0x7c,0xa5]
vsra.vv v25, v23, v24, v0.t

# Encoding: |101001|1|11010|00101|100|11011|1010111|
# CHECK: vsra.vx v27, v26, t0
# CHECK-SAME: [0xd7,0xcd,0xa2,0xa7]
vsra.vx v27, v26, t0
# Encoding: |101001|0|11100|00111|100|11101|1010111|
# CHECK: vsra.vx v29, v28, t2, v0.t
# CHECK-SAME: [0xd7,0xce,0xc3,0xa5]
vsra.vx v29, v28, t2, v0.t

# Encoding: |101001|1|11110|01001|011|11111|1010111|
# CHECK: vsra.vi v31, v30, 9
# CHECK-SAME: [0xd7,0xbf,0xe4,0xa7]
vsra.vi v31, v30, 9
# Encoding: |101001|0|00000|01010|011|00001|1010111|
# CHECK: vsra.vi v1, v0, 10, v0.t
# CHECK-SAME: [0xd7,0x30,0x05,0xa4]
vsra.vi v1, v0, 10, v0.t

# Encoding: |101010|1|00010|00011|000|00100|1010111|
# CHECK: vssrl.vv v4, v2, v3
# CHECK-SAME: [0x57,0x82,0x21,0xaa]
vssrl.vv v4, v2, v3
# Encoding: |101010|0|00101|00110|000|00111|1010111|
# CHECK: vssrl.vv v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x03,0x53,0xa8]
vssrl.vv v7, v5, v6, v0.t

# Encoding: |101010|1|01000|01001|100|01001|1010111|
# CHECK: vssrl.vx v9, v8, s1
# CHECK-SAME: [0xd7,0xc4,0x84,0xaa]
vssrl.vx v9, v8, s1
# Encoding: |101010|0|01010|01011|100|01011|1010111|
# CHECK: vssrl.vx v11, v10, a1, v0.t
# CHECK-SAME: [0xd7,0xc5,0xa5,0xa8]
vssrl.vx v11, v10, a1, v0.t

# Encoding: |101010|1|01100|01011|011|01101|1010111|
# CHECK: vssrl.vi v13, v12, 11
# CHECK-SAME: [0xd7,0xb6,0xc5,0xaa]
vssrl.vi v13, v12, 11
# Encoding: |101010|0|01110|01100|011|01111|1010111|
# CHECK: vssrl.vi v15, v14, 12, v0.t
# CHECK-SAME: [0xd7,0x37,0xe6,0xa8]
vssrl.vi v15, v14, 12, v0.t

# Encoding: |101011|1|10000|10001|000|10010|1010111|
# CHECK: vssra.vv v18, v16, v17
# CHECK-SAME: [0x57,0x89,0x08,0xaf]
vssra.vv v18, v16, v17
# Encoding: |101011|0|10011|10100|000|10101|1010111|
# CHECK: vssra.vv v21, v19, v20, v0.t
# CHECK-SAME: [0xd7,0x0a,0x3a,0xad]
vssra.vv v21, v19, v20, v0.t

# Encoding: |101011|1|10110|01101|100|10111|1010111|
# CHECK: vssra.vx v23, v22, a3
# CHECK-SAME: [0xd7,0xcb,0x66,0xaf]
vssra.vx v23, v22, a3
# Encoding: |101011|0|11000|01111|100|11001|1010111|
# CHECK: vssra.vx v25, v24, a5, v0.t
# CHECK-SAME: [0xd7,0xcc,0x87,0xad]
vssra.vx v25, v24, a5, v0.t

# Encoding: |101011|1|11010|01101|011|11011|1010111|
# CHECK: vssra.vi v27, v26, 13
# CHECK-SAME: [0xd7,0xbd,0xa6,0xaf]
vssra.vi v27, v26, 13
# Encoding: |101011|0|11100|01110|011|11101|1010111|
# CHECK: vssra.vi v29, v28, 14, v0.t
# CHECK-SAME: [0xd7,0x3e,0xc7,0xad]
vssra.vi v29, v28, 14, v0.t

# Encoding: |101100|1|11110|11111|000|00000|1010111|
# CHECK: vnsrl.vv v0, v30, v31
# CHECK-SAME: [0x57,0x80,0xef,0xb3]
vnsrl.vv v0, v30, v31
# Encoding: |101100|0|00001|00010|000|00011|1010111|
# CHECK: vnsrl.vv v3, v1, v2, v0.t
# CHECK-SAME: [0xd7,0x01,0x11,0xb0]
vnsrl.vv v3, v1, v2, v0.t

# Encoding: |101100|1|00100|10001|100|00101|1010111|
# CHECK: vnsrl.vx v5, v4, a7
# CHECK-SAME: [0xd7,0xc2,0x48,0xb2]
vnsrl.vx v5, v4, a7
# Encoding: |101100|0|00110|10011|100|00111|1010111|
# CHECK: vnsrl.vx v7, v6, s3, v0.t
# CHECK-SAME: [0xd7,0xc3,0x69,0xb0]
vnsrl.vx v7, v6, s3, v0.t

# Encoding: |101100|1|01000|01111|011|01001|1010111|
# CHECK: vnsrl.vi v9, v8, 15
# CHECK-SAME: [0xd7,0xb4,0x87,0xb2]
vnsrl.vi v9, v8, 15
# Encoding: |101100|0|01010|10000|011|01011|1010111|
# CHECK: vnsrl.vi v11, v10, -16, v0.t
# CHECK-SAME: [0xd7,0x35,0xa8,0xb0]
vnsrl.vi v11, v10, -16, v0.t

# Encoding: |101101|1|01100|01101|000|01110|1010111|
# CHECK: vnsra.vv v14, v12, v13
# CHECK-SAME: [0x57,0x87,0xc6,0xb6]
vnsra.vv v14, v12, v13
# Encoding: |101101|0|01111|10000|000|10001|1010111|
# CHECK: vnsra.vv v17, v15, v16, v0.t
# CHECK-SAME: [0xd7,0x08,0xf8,0xb4]
vnsra.vv v17, v15, v16, v0.t

# Encoding: |101101|1|10010|10101|100|10011|1010111|
# CHECK: vnsra.vx v19, v18, s5
# CHECK-SAME: [0xd7,0xc9,0x2a,0xb7]
vnsra.vx v19, v18, s5
# Encoding: |101101|0|10100|10111|100|10101|1010111|
# CHECK: vnsra.vx v21, v20, s7, v0.t
# CHECK-SAME: [0xd7,0xca,0x4b,0xb5]
vnsra.vx v21, v20, s7, v0.t

# Encoding: |101101|1|10110|10001|011|10111|1010111|
# CHECK: vnsra.vi v23, v22, -15
# CHECK-SAME: [0xd7,0xbb,0x68,0xb7]
vnsra.vi v23, v22, -15
# Encoding: |101101|0|11000|10010|011|11001|1010111|
# CHECK: vnsra.vi v25, v24, -14, v0.t
# CHECK-SAME: [0xd7,0x3c,0x89,0xb5]
vnsra.vi v25, v24, -14, v0.t

# Encoding: |101110|1|11010|11011|000|11100|1010111|
# CHECK: vnclipu.vv v28, v26, v27
# CHECK-SAME: [0x57,0x8e,0xad,0xbb]
vnclipu.vv v28, v26, v27
# Encoding: |101110|0|11101|11110|000|11111|1010111|
# CHECK: vnclipu.vv v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x0f,0xdf,0xb9]
vnclipu.vv v31, v29, v30, v0.t

# Encoding: |101110|1|00000|11001|100|00001|1010111|
# CHECK: vnclipu.vx v1, v0, s9
# CHECK-SAME: [0xd7,0xc0,0x0c,0xba]
vnclipu.vx v1, v0, s9
# Encoding: |101110|0|00010|11011|100|00011|1010111|
# CHECK: vnclipu.vx v3, v2, s11, v0.t
# CHECK-SAME: [0xd7,0xc1,0x2d,0xb8]
vnclipu.vx v3, v2, s11, v0.t

# Encoding: |101110|1|00100|10011|011|00101|1010111|
# CHECK: vnclipu.vi v5, v4, -13
# CHECK-SAME: [0xd7,0xb2,0x49,0xba]
vnclipu.vi v5, v4, -13
# Encoding: |101110|0|00110|10100|011|00111|1010111|
# CHECK: vnclipu.vi v7, v6, -12, v0.t
# CHECK-SAME: [0xd7,0x33,0x6a,0xb8]
vnclipu.vi v7, v6, -12, v0.t

# Encoding: |101111|1|01000|01001|000|01010|1010111|
# CHECK: vnclip.vv v10, v8, v9
# CHECK-SAME: [0x57,0x85,0x84,0xbe]
vnclip.vv v10, v8, v9
# Encoding: |101111|0|01011|01100|000|01101|1010111|
# CHECK: vnclip.vv v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x06,0xb6,0xbc]
vnclip.vv v13, v11, v12, v0.t

# Encoding: |101111|1|01110|11101|100|01111|1010111|
# CHECK: vnclip.vx v15, v14, t4
# CHECK-SAME: [0xd7,0xc7,0xee,0xbe]
vnclip.vx v15, v14, t4
# Encoding: |101111|0|10000|11111|100|10001|1010111|
# CHECK: vnclip.vx v17, v16, t6, v0.t
# CHECK-SAME: [0xd7,0xc8,0x0f,0xbd]
vnclip.vx v17, v16, t6, v0.t

# Encoding: |101111|1|10010|10101|011|10011|1010111|
# CHECK: vnclip.vi v19, v18, -11
# CHECK-SAME: [0xd7,0xb9,0x2a,0xbf]
vnclip.vi v19, v18, -11
# Encoding: |101111|0|10100|10110|011|10101|1010111|
# CHECK: vnclip.vi v21, v20, -10, v0.t
# CHECK-SAME: [0xd7,0x3a,0x4b,0xbd]
vnclip.vi v21, v20, -10, v0.t

# Encoding: |110000|1|10110|10111|000|11000|1010111|
# CHECK: vwredsumu.vs v24, v22, v23
# CHECK-SAME: [0x57,0x8c,0x6b,0xc3]
vwredsumu.vs v24, v22, v23
# Encoding: |110000|0|11001|11010|000|11011|1010111|
# CHECK: vwredsumu.vs v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x0d,0x9d,0xc1]
vwredsumu.vs v27, v25, v26, v0.t

# Encoding: |110001|1|11100|11101|000|11110|1010111|
# CHECK: vwredsum.vs v30, v28, v29
# CHECK-SAME: [0x57,0x8f,0xce,0xc7]
vwredsum.vs v30, v28, v29
# Encoding: |110001|0|11111|00000|000|00001|1010111|
# CHECK: vwredsum.vs v1, v31, v0, v0.t
# CHECK-SAME: [0xd7,0x00,0xf0,0xc5]
vwredsum.vs v1, v31, v0, v0.t

# Encoding: |111000|1|00010|00011|000|00100|1010111|
# CHECK: vdotu.vv v4, v2, v3
# CHECK-SAME: [0x57,0x82,0x21,0xe2]
vdotu.vv v4, v2, v3
# Encoding: |111000|0|00101|00110|000|00111|1010111|
# CHECK: vdotu.vv v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x03,0x53,0xe0]
vdotu.vv v7, v5, v6, v0.t

# Encoding: |111001|1|01000|01001|000|01010|1010111|
# CHECK: vdot.vv v10, v8, v9
# CHECK-SAME: [0x57,0x85,0x84,0xe6]
vdot.vv v10, v8, v9
# Encoding: |111001|0|01011|01100|000|01101|1010111|
# CHECK: vdot.vv v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x06,0xb6,0xe4]
vdot.vv v13, v11, v12, v0.t

# Encoding: |111100|1|01110|01111|000|10000|1010111|
# CHECK: vwsmaccu.vv v16, v14, v15
# CHECK-SAME: [0x57,0x88,0xe7,0xf2]
vwsmaccu.vv v16, v14, v15
# Encoding: |111100|0|10001|10010|000|10011|1010111|
# CHECK: vwsmaccu.vv v19, v17, v18, v0.t
# CHECK-SAME: [0xd7,0x09,0x19,0xf1]
vwsmaccu.vv v19, v17, v18, v0.t

# Encoding: |111100|1|10100|00010|100|10101|1010111|
# CHECK: vwsmaccu.vx v21, v20, sp
# CHECK-SAME: [0xd7,0x4a,0x41,0xf3]
vwsmaccu.vx v21, v20, sp
# Encoding: |111100|0|10110|00100|100|10111|1010111|
# CHECK: vwsmaccu.vx v23, v22, tp, v0.t
# CHECK-SAME: [0xd7,0x4b,0x62,0xf1]
vwsmaccu.vx v23, v22, tp, v0.t

# Encoding: |111101|1|11000|11001|000|11010|1010111|
# CHECK: vwsmacc.vv v26, v24, v25
# CHECK-SAME: [0x57,0x8d,0x8c,0xf7]
vwsmacc.vv v26, v24, v25
# Encoding: |111101|0|11011|11100|000|11101|1010111|
# CHECK: vwsmacc.vv v29, v27, v28, v0.t
# CHECK-SAME: [0xd7,0x0e,0xbe,0xf5]
vwsmacc.vv v29, v27, v28, v0.t

# Encoding: |111101|1|11110|00110|100|11111|1010111|
# CHECK: vwsmacc.vx v31, v30, t1
# CHECK-SAME: [0xd7,0x4f,0xe3,0xf7]
vwsmacc.vx v31, v30, t1
# Encoding: |111101|0|00000|01000|100|00001|1010111|
# CHECK: vwsmacc.vx v1, v0, s0, v0.t
# CHECK-SAME: [0xd7,0x40,0x04,0xf4]
vwsmacc.vx v1, v0, s0, v0.t

# Encoding: |111110|1|00010|00011|000|00100|1010111|
# CHECK: vwsmsacu.vv v4, v2, v3
# CHECK-SAME: [0x57,0x82,0x21,0xfa]
vwsmsacu.vv v4, v2, v3
# Encoding: |111110|0|00101|00110|000|00111|1010111|
# CHECK: vwsmsacu.vv v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x03,0x53,0xf8]
vwsmsacu.vv v7, v5, v6, v0.t

# Encoding: |111110|1|01000|01010|100|01001|1010111|
# CHECK: vwsmsacu.vx v9, v8, a0
# CHECK-SAME: [0xd7,0x44,0x85,0xfa]
vwsmsacu.vx v9, v8, a0
# Encoding: |111110|0|01010|01100|100|01011|1010111|
# CHECK: vwsmsacu.vx v11, v10, a2, v0.t
# CHECK-SAME: [0xd7,0x45,0xa6,0xf8]
vwsmsacu.vx v11, v10, a2, v0.t

# Encoding: |111111|1|01100|01101|000|01110|1010111|
# CHECK: vwsmsac.vv v14, v12, v13
# CHECK-SAME: [0x57,0x87,0xc6,0xfe]
vwsmsac.vv v14, v12, v13
# Encoding: |111111|0|01111|10000|000|10001|1010111|
# CHECK: vwsmsac.vv v17, v15, v16, v0.t
# CHECK-SAME: [0xd7,0x08,0xf8,0xfc]
vwsmsac.vv v17, v15, v16, v0.t

# Encoding: |111111|1|10010|01110|100|10011|1010111|
# CHECK: vwsmsac.vx v19, v18, a4
# CHECK-SAME: [0xd7,0x49,0x27,0xff]
vwsmsac.vx v19, v18, a4
# Encoding: |111111|0|10100|10000|100|10101|1010111|
# CHECK: vwsmsac.vx v21, v20, a6, v0.t
# CHECK-SAME: [0xd7,0x4a,0x48,0xfd]
vwsmsac.vx v21, v20, a6, v0.t

# Encoding: |000000|1|10110|10111|010|11000|1010111|
# CHECK: vredsum.vs v24, v22, v23
# CHECK-SAME: [0x57,0xac,0x6b,0x03]
vredsum.vs v24, v22, v23
# Encoding: |000000|0|11001|11010|010|11011|1010111|
# CHECK: vredsum.vs v27, v25, v26, v0.t
# CHECK-SAME: [0xd7,0x2d,0x9d,0x01]
vredsum.vs v27, v25, v26, v0.t

# Encoding: |000001|1|11100|11101|010|11110|1010111|
# CHECK: vredand.vs v30, v28, v29
# CHECK-SAME: [0x57,0xaf,0xce,0x07]
vredand.vs v30, v28, v29
# Encoding: |000001|0|11111|00000|010|00001|1010111|
# CHECK: vredand.vs v1, v31, v0, v0.t
# CHECK-SAME: [0xd7,0x20,0xf0,0x05]
vredand.vs v1, v31, v0, v0.t

# Encoding: |000010|1|00010|00011|010|00100|1010111|
# CHECK: vredor.vs v4, v2, v3
# CHECK-SAME: [0x57,0xa2,0x21,0x0a]
vredor.vs v4, v2, v3
# Encoding: |000010|0|00101|00110|010|00111|1010111|
# CHECK: vredor.vs v7, v5, v6, v0.t
# CHECK-SAME: [0xd7,0x23,0x53,0x08]
vredor.vs v7, v5, v6, v0.t

# Encoding: |000011|1|01000|01001|010|01010|1010111|
# CHECK: vredxor.vs v10, v8, v9
# CHECK-SAME: [0x57,0xa5,0x84,0x0e]
vredxor.vs v10, v8, v9
# Encoding: |000011|0|01011|01100|010|01101|1010111|
# CHECK: vredxor.vs v13, v11, v12, v0.t
# CHECK-SAME: [0xd7,0x26,0xb6,0x0c]
vredxor.vs v13, v11, v12, v0.t

# Encoding: |000100|1|01110|01111|010|10000|1010111|
# CHECK: vredminu.vs v16, v14, v15
# CHECK-SAME: [0x57,0xa8,0xe7,0x12]
vredminu.vs v16, v14, v15
# Encoding: |000100|0|10001|10010|010|10011|1010111|
# CHECK: vredminu.vs v19, v17, v18, v0.t
# CHECK-SAME: [0xd7,0x29,0x19,0x11]
vredminu.vs v19, v17, v18, v0.t

# Encoding: |000101|1|10100|10101|010|10110|1010111|
# CHECK: vredmin.vs v22, v20, v21
# CHECK-SAME: [0x57,0xab,0x4a,0x17]
vredmin.vs v22, v20, v21
# Encoding: |000101|0|10111|11000|010|11001|1010111|
# CHECK: vredmin.vs v25, v23, v24, v0.t
# CHECK-SAME: [0xd7,0x2c,0x7c,0x15]
vredmin.vs v25, v23, v24, v0.t

# Encoding: |000110|1|11010|11011|010|11100|1010111|
# CHECK: vredmaxu.vs v28, v26, v27
# CHECK-SAME: [0x57,0xae,0xad,0x1b]
vredmaxu.vs v28, v26, v27
# Encoding: |000110|0|11101|11110|010|11111|1010111|
# CHECK: vredmaxu.vs v31, v29, v30, v0.t
# CHECK-SAME: [0xd7,0x2f,0xdf,0x19]
vredmaxu.vs v31, v29, v30, v0.t

# Encoding: |000111|1|00000|00001|010|00010|1010111|
# CHECK: vredmax.vs v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0x1e]
vredmax.vs v2, v0, v1
# Encoding: |000111|0|00011|00100|010|00101|1010111|
# CHECK: vredmax.vs v5, v3, v4, v0.t
# CHECK-SAME: [0xd7,0x22,0x32,0x1c]
vredmax.vs v5, v3, v4, v0.t

# Encoding: |001100|1|00110|10010|010|10100|1010111|
# CHECK: vext.x.v s4, v6, s2
# CHECK-SAME: [0x57,0x2a,0x69,0x32]
vext.x.v s4, v6, s2

# Encoding: |001101|1|00000|10110|110|00111|1010111|
# CHECK: vmv.s.x v7, s6
# CHECK-SAME: [0xd7,0x63,0x0b,0x36]
vmv.s.x v7, s6

# Encoding: |001110|1|01000|11000|110|01001|1010111|
# CHECK: vslide1up.vx v9, v8, s8
# CHECK-SAME: [0xd7,0x64,0x8c,0x3a]
vslide1up.vx v9, v8, s8
# Encoding: |001110|0|01010|11010|110|01011|1010111|
# CHECK: vslide1up.vx v11, v10, s10, v0.t
# CHECK-SAME: [0xd7,0x65,0xad,0x38]
vslide1up.vx v11, v10, s10, v0.t

# Encoding: |001111|1|01100|11100|110|01101|1010111|
# CHECK: vslide1down.vx v13, v12, t3
# CHECK-SAME: [0xd7,0x66,0xce,0x3e]
vslide1down.vx v13, v12, t3
# Encoding: |001111|0|01110|11110|110|01111|1010111|
# CHECK: vslide1down.vx v15, v14, t5, v0.t
# CHECK-SAME: [0xd7,0x67,0xef,0x3c]
vslide1down.vx v15, v14, t5, v0.t

# Encoding: |010100|1|10000|00000|010|00001|1010111|
# CHECK: vmpopc.m ra, v16
# CHECK-SAME: [0xd7,0x20,0x00,0x53]
vmpopc.m ra, v16
# Encoding: |010100|0|10001|00000|010|00011|1010111|
# CHECK: vmpopc.m gp, v17, v0.t
# CHECK-SAME: [0xd7,0x21,0x10,0x51]
vmpopc.m gp, v17, v0.t

# Encoding: |010101|1|10010|00000|010|00101|1010111|
# CHECK: vmfirst.m t0, v18
# CHECK-SAME: [0xd7,0x22,0x20,0x57]
vmfirst.m t0, v18
# Encoding: |010101|0|10011|00000|010|00111|1010111|
# CHECK: vmfirst.m t2, v19, v0.t
# CHECK-SAME: [0xd7,0x23,0x30,0x55]
vmfirst.m t2, v19, v0.t

# Encoding: |010111|1|10100|10101|010|10110|1010111|
# CHECK: vcompress.vm v22, v20, v21
# CHECK-SAME: [0x57,0xab,0x4a,0x5f]
vcompress.vm v22, v20, v21

# Encoding: |011000|1|10111|11000|010|11001|1010111|
# CHECK: vmandnot.mm v25, v23, v24
# CHECK-SAME: [0xd7,0x2c,0x7c,0x63]
vmandnot.mm v25, v23, v24

# Encoding: |011001|1|11010|11011|010|11100|1010111|
# CHECK: vmand.mm v28, v26, v27
# CHECK-SAME: [0x57,0xae,0xad,0x67]
vmand.mm v28, v26, v27

# Encoding: |011010|1|11101|11110|010|11111|1010111|
# CHECK: vmor.mm v31, v29, v30
# CHECK-SAME: [0xd7,0x2f,0xdf,0x6b]
vmor.mm v31, v29, v30

# Encoding: |011011|1|00000|00001|010|00010|1010111|
# CHECK: vmxor.mm v2, v0, v1
# CHECK-SAME: [0x57,0xa1,0x00,0x6e]
vmxor.mm v2, v0, v1

# Encoding: |011100|1|00011|00100|010|00101|1010111|
# CHECK: vmornot.mm v5, v3, v4
# CHECK-SAME: [0xd7,0x22,0x32,0x72]
vmornot.mm v5, v3, v4

# Encoding: |011101|1|00110|00111|010|01000|1010111|
# CHECK: vmnand.mm v8, v6, v7
# CHECK-SAME: [0x57,0xa4,0x63,0x76]
vmnand.mm v8, v6, v7

# Encoding: |011110|1|01001|01010|010|01011|1010111|
# CHECK: vmnor.mm v11, v9, v10
# CHECK-SAME: [0xd7,0x25,0x95,0x7a]
vmnor.mm v11, v9, v10

# Encoding: |011111|1|01100|01101|010|01110|1010111|
# CHECK: vmxnor.mm v14, v12, v13
# CHECK-SAME: [0x57,0xa7,0xc6,0x7e]
vmxnor.mm v14, v12, v13

# Encoding: |100000|1|01111|10000|010|10001|1010111|
# CHECK: vdivu.vv v17, v15, v16
# CHECK-SAME: [0xd7,0x28,0xf8,0x82]
vdivu.vv v17, v15, v16
# Encoding: |100000|0|10010|10011|010|10100|1010111|
# CHECK: vdivu.vv v20, v18, v19, v0.t
# CHECK-SAME: [0x57,0xaa,0x29,0x81]
vdivu.vv v20, v18, v19, v0.t

# Encoding: |100000|1|10101|01001|110|10110|1010111|
# CHECK: vdivu.vx v22, v21, s1
# CHECK-SAME: [0x57,0xeb,0x54,0x83]
vdivu.vx v22, v21, s1
# Encoding: |100000|0|10111|01011|110|11000|1010111|
# CHECK: vdivu.vx v24, v23, a1, v0.t
# CHECK-SAME: [0x57,0xec,0x75,0x81]
vdivu.vx v24, v23, a1, v0.t

# Encoding: |100001|1|11001|11010|010|11011|1010111|
# CHECK: vdiv.vv v27, v25, v26
# CHECK-SAME: [0xd7,0x2d,0x9d,0x87]
vdiv.vv v27, v25, v26
# Encoding: |100001|0|11100|11101|010|11110|1010111|
# CHECK: vdiv.vv v30, v28, v29, v0.t
# CHECK-SAME: [0x57,0xaf,0xce,0x85]
vdiv.vv v30, v28, v29, v0.t

# Encoding: |100001|1|11111|01101|110|00000|1010111|
# CHECK: vdiv.vx v0, v31, a3
# CHECK-SAME: [0x57,0xe0,0xf6,0x87]
vdiv.vx v0, v31, a3
# Encoding: |100001|0|00001|01111|110|00010|1010111|
# CHECK: vdiv.vx v2, v1, a5, v0.t
# CHECK-SAME: [0x57,0xe1,0x17,0x84]
vdiv.vx v2, v1, a5, v0.t

# Encoding: |100010|1|00011|00100|010|00101|1010111|
# CHECK: vremu.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x22,0x32,0x8a]
vremu.vv v5, v3, v4
# Encoding: |100010|0|00110|00111|010|01000|1010111|
# CHECK: vremu.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0xa4,0x63,0x88]
vremu.vv v8, v6, v7, v0.t

# Encoding: |100010|1|01001|10001|110|01010|1010111|
# CHECK: vremu.vx v10, v9, a7
# CHECK-SAME: [0x57,0xe5,0x98,0x8a]
vremu.vx v10, v9, a7
# Encoding: |100010|0|01011|10011|110|01100|1010111|
# CHECK: vremu.vx v12, v11, s3, v0.t
# CHECK-SAME: [0x57,0xe6,0xb9,0x88]
vremu.vx v12, v11, s3, v0.t

# Encoding: |100011|1|01101|01110|010|01111|1010111|
# CHECK: vrem.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x27,0xd7,0x8e]
vrem.vv v15, v13, v14
# Encoding: |100011|0|10000|10001|010|10010|1010111|
# CHECK: vrem.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0xa9,0x08,0x8d]
vrem.vv v18, v16, v17, v0.t

# Encoding: |100011|1|10011|10101|110|10100|1010111|
# CHECK: vrem.vx v20, v19, s5
# CHECK-SAME: [0x57,0xea,0x3a,0x8f]
vrem.vx v20, v19, s5
# Encoding: |100011|0|10101|10111|110|10110|1010111|
# CHECK: vrem.vx v22, v21, s7, v0.t
# CHECK-SAME: [0x57,0xeb,0x5b,0x8d]
vrem.vx v22, v21, s7, v0.t

# Encoding: |100100|1|10111|11000|010|11001|1010111|
# CHECK: vmulhu.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x2c,0x7c,0x93]
vmulhu.vv v25, v23, v24
# Encoding: |100100|0|11010|11011|010|11100|1010111|
# CHECK: vmulhu.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0xae,0xad,0x91]
vmulhu.vv v28, v26, v27, v0.t

# Encoding: |100100|1|11101|11001|110|11110|1010111|
# CHECK: vmulhu.vx v30, v29, s9
# CHECK-SAME: [0x57,0xef,0xdc,0x93]
vmulhu.vx v30, v29, s9
# Encoding: |100100|0|11111|11011|110|00000|1010111|
# CHECK: vmulhu.vx v0, v31, s11, v0.t
# CHECK-SAME: [0x57,0xe0,0xfd,0x91]
vmulhu.vx v0, v31, s11, v0.t

# Encoding: |100101|1|00001|00010|010|00011|1010111|
# CHECK: vmul.vv v3, v1, v2
# CHECK-SAME: [0xd7,0x21,0x11,0x96]
vmul.vv v3, v1, v2
# Encoding: |100101|0|00100|00101|010|00110|1010111|
# CHECK: vmul.vv v6, v4, v5, v0.t
# CHECK-SAME: [0x57,0xa3,0x42,0x94]
vmul.vv v6, v4, v5, v0.t

# Encoding: |100101|1|00111|11101|110|01000|1010111|
# CHECK: vmul.vx v8, v7, t4
# CHECK-SAME: [0x57,0xe4,0x7e,0x96]
vmul.vx v8, v7, t4
# Encoding: |100101|0|01001|11111|110|01010|1010111|
# CHECK: vmul.vx v10, v9, t6, v0.t
# CHECK-SAME: [0x57,0xe5,0x9f,0x94]
vmul.vx v10, v9, t6, v0.t

# Encoding: |100110|1|01011|01100|010|01101|1010111|
# CHECK: vmulhsu.vv v13, v11, v12
# CHECK-SAME: [0xd7,0x26,0xb6,0x9a]
vmulhsu.vv v13, v11, v12
# Encoding: |100110|0|01110|01111|010|10000|1010111|
# CHECK: vmulhsu.vv v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0xa8,0xe7,0x98]
vmulhsu.vv v16, v14, v15, v0.t

# Encoding: |100110|1|10001|00010|110|10010|1010111|
# CHECK: vmulhsu.vx v18, v17, sp
# CHECK-SAME: [0x57,0x69,0x11,0x9b]
vmulhsu.vx v18, v17, sp
# Encoding: |100110|0|10011|00100|110|10100|1010111|
# CHECK: vmulhsu.vx v20, v19, tp, v0.t
# CHECK-SAME: [0x57,0x6a,0x32,0x99]
vmulhsu.vx v20, v19, tp, v0.t

# Encoding: |100111|1|10101|10110|010|10111|1010111|
# CHECK: vmulh.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x2b,0x5b,0x9f]
vmulh.vv v23, v21, v22
# Encoding: |100111|0|11000|11001|010|11010|1010111|
# CHECK: vmulh.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0xad,0x8c,0x9d]
vmulh.vv v26, v24, v25, v0.t

# Encoding: |100111|1|11011|00110|110|11100|1010111|
# CHECK: vmulh.vx v28, v27, t1
# CHECK-SAME: [0x57,0x6e,0xb3,0x9f]
vmulh.vx v28, v27, t1
# Encoding: |100111|0|11101|01000|110|11110|1010111|
# CHECK: vmulh.vx v30, v29, s0, v0.t
# CHECK-SAME: [0x57,0x6f,0xd4,0x9d]
vmulh.vx v30, v29, s0, v0.t

# Encoding: |101001|1|11111|00000|010|00001|1010111|
# CHECK: vmadd.vv v1, v0, v31
# CHECK-SAME: [0xd7,0x20,0xf0,0xa7]
vmadd.vv v1, v0, v31
# Encoding: |101001|0|00010|00011|010|00100|1010111|
# CHECK: vmadd.vv v4, v3, v2, v0.t
# CHECK-SAME: [0x57,0xa2,0x21,0xa4]
vmadd.vv v4, v3, v2, v0.t

# Encoding: |101001|1|00101|01010|110|00110|1010111|
# CHECK: vmadd.vx v6, a0, v5
# CHECK-SAME: [0x57,0x63,0x55,0xa6]
vmadd.vx v6, a0, v5
# Encoding: |101001|0|00111|01100|110|01000|1010111|
# CHECK: vmadd.vx v8, a2, v7, v0.t
# CHECK-SAME: [0x57,0x64,0x76,0xa4]
vmadd.vx v8, a2, v7, v0.t

# Encoding: |101011|1|01001|01010|010|01011|1010111|
# CHECK: vmsub.vv v11, v10, v9
# CHECK-SAME: [0xd7,0x25,0x95,0xae]
vmsub.vv v11, v10, v9
# Encoding: |101011|0|01100|01101|010|01110|1010111|
# CHECK: vmsub.vv v14, v13, v12, v0.t
# CHECK-SAME: [0x57,0xa7,0xc6,0xac]
vmsub.vv v14, v13, v12, v0.t

# Encoding: |101011|1|01111|01110|110|10000|1010111|
# CHECK: vmsub.vx v16, a4, v15
# CHECK-SAME: [0x57,0x68,0xf7,0xae]
vmsub.vx v16, a4, v15
# Encoding: |101011|0|10001|10000|110|10010|1010111|
# CHECK: vmsub.vx v18, a6, v17, v0.t
# CHECK-SAME: [0x57,0x69,0x18,0xad]
vmsub.vx v18, a6, v17, v0.t

# Encoding: |101101|1|10011|10100|010|10101|1010111|
# CHECK: vmacc.vv v21, v20, v19
# CHECK-SAME: [0xd7,0x2a,0x3a,0xb7]
vmacc.vv v21, v20, v19
# Encoding: |101101|0|10110|10111|010|11000|1010111|
# CHECK: vmacc.vv v24, v23, v22, v0.t
# CHECK-SAME: [0x57,0xac,0x6b,0xb5]
vmacc.vv v24, v23, v22, v0.t

# Encoding: |101101|1|11001|10010|110|11010|1010111|
# CHECK: vmacc.vx v26, s2, v25
# CHECK-SAME: [0x57,0x6d,0x99,0xb7]
vmacc.vx v26, s2, v25
# Encoding: |101101|0|11011|10100|110|11100|1010111|
# CHECK: vmacc.vx v28, s4, v27, v0.t
# CHECK-SAME: [0x57,0x6e,0xba,0xb5]
vmacc.vx v28, s4, v27, v0.t

# Encoding: |101111|1|11101|11110|010|11111|1010111|
# CHECK: vmsac.vv v31, v30, v29
# CHECK-SAME: [0xd7,0x2f,0xdf,0xbf]
vmsac.vv v31, v30, v29
# Encoding: |101111|0|00000|00001|010|00010|1010111|
# CHECK: vmsac.vv v2, v1, v0, v0.t
# CHECK-SAME: [0x57,0xa1,0x00,0xbc]
vmsac.vv v2, v1, v0, v0.t

# Encoding: |101111|1|00011|10110|110|00100|1010111|
# CHECK: vmsac.vx v4, s6, v3
# CHECK-SAME: [0x57,0x62,0x3b,0xbe]
vmsac.vx v4, s6, v3
# Encoding: |101111|0|00101|11000|110|00110|1010111|
# CHECK: vmsac.vx v6, s8, v5, v0.t
# CHECK-SAME: [0x57,0x63,0x5c,0xbc]
vmsac.vx v6, s8, v5, v0.t

# Encoding: |110000|1|00111|01000|010|01001|1010111|
# CHECK: vwaddu.vv v9, v7, v8
# CHECK-SAME: [0xd7,0x24,0x74,0xc2]
vwaddu.vv v9, v7, v8
# Encoding: |110000|0|01010|01011|010|01100|1010111|
# CHECK: vwaddu.vv v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0xa6,0xa5,0xc0]
vwaddu.vv v12, v10, v11, v0.t

# Encoding: |110000|1|01101|11010|110|01110|1010111|
# CHECK: vwaddu.vx v14, v13, s10
# CHECK-SAME: [0x57,0x67,0xdd,0xc2]
vwaddu.vx v14, v13, s10
# Encoding: |110000|0|01111|11100|110|10000|1010111|
# CHECK: vwaddu.vx v16, v15, t3, v0.t
# CHECK-SAME: [0x57,0x68,0xfe,0xc0]
vwaddu.vx v16, v15, t3, v0.t

# Encoding: |110001|1|10001|10010|010|10011|1010111|
# CHECK: vwadd.vv v19, v17, v18
# CHECK-SAME: [0xd7,0x29,0x19,0xc7]
vwadd.vv v19, v17, v18
# Encoding: |110001|0|10100|10101|010|10110|1010111|
# CHECK: vwadd.vv v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0xab,0x4a,0xc5]
vwadd.vv v22, v20, v21, v0.t

# Encoding: |110001|1|10111|11110|110|11000|1010111|
# CHECK: vwadd.vx v24, v23, t5
# CHECK-SAME: [0x57,0x6c,0x7f,0xc7]
vwadd.vx v24, v23, t5
# Encoding: |110001|0|11001|00001|110|11010|1010111|
# CHECK: vwadd.vx v26, v25, ra, v0.t
# CHECK-SAME: [0x57,0xed,0x90,0xc5]
vwadd.vx v26, v25, ra, v0.t

# Encoding: |110010|1|11011|11100|010|11101|1010111|
# CHECK: vwsubu.vv v29, v27, v28
# CHECK-SAME: [0xd7,0x2e,0xbe,0xcb]
vwsubu.vv v29, v27, v28
# Encoding: |110010|0|11110|11111|010|00000|1010111|
# CHECK: vwsubu.vv v0, v30, v31, v0.t
# CHECK-SAME: [0x57,0xa0,0xef,0xc9]
vwsubu.vv v0, v30, v31, v0.t

# Encoding: |110010|1|00001|00011|110|00010|1010111|
# CHECK: vwsubu.vx v2, v1, gp
# CHECK-SAME: [0x57,0xe1,0x11,0xca]
vwsubu.vx v2, v1, gp
# Encoding: |110010|0|00011|00101|110|00100|1010111|
# CHECK: vwsubu.vx v4, v3, t0, v0.t
# CHECK-SAME: [0x57,0xe2,0x32,0xc8]
vwsubu.vx v4, v3, t0, v0.t

# Encoding: |110011|1|00101|00110|010|00111|1010111|
# CHECK: vwsub.vv v7, v5, v6
# CHECK-SAME: [0xd7,0x23,0x53,0xce]
vwsub.vv v7, v5, v6
# Encoding: |110011|0|01000|01001|010|01010|1010111|
# CHECK: vwsub.vv v10, v8, v9, v0.t
# CHECK-SAME: [0x57,0xa5,0x84,0xcc]
vwsub.vv v10, v8, v9, v0.t

# Encoding: |110011|1|01011|00111|110|01100|1010111|
# CHECK: vwsub.vx v12, v11, t2
# CHECK-SAME: [0x57,0xe6,0xb3,0xce]
vwsub.vx v12, v11, t2
# Encoding: |110011|0|01101|01001|110|01110|1010111|
# CHECK: vwsub.vx v14, v13, s1, v0.t
# CHECK-SAME: [0x57,0xe7,0xd4,0xcc]
vwsub.vx v14, v13, s1, v0.t

# Encoding: |110100|1|01111|10000|010|10001|1010111|
# CHECK: vwaddu.wv v17, v15, v16
# CHECK-SAME: [0xd7,0x28,0xf8,0xd2]
vwaddu.wv v17, v15, v16
# Encoding: |110100|0|10010|10011|010|10100|1010111|
# CHECK: vwaddu.wv v20, v18, v19, v0.t
# CHECK-SAME: [0x57,0xaa,0x29,0xd1]
vwaddu.wv v20, v18, v19, v0.t

# Encoding: |110100|1|10101|01011|110|10110|1010111|
# CHECK: vwaddu.wx v22, v21, a1
# CHECK-SAME: [0x57,0xeb,0x55,0xd3]
vwaddu.wx v22, v21, a1
# Encoding: |110100|0|10111|01101|110|11000|1010111|
# CHECK: vwaddu.wx v24, v23, a3, v0.t
# CHECK-SAME: [0x57,0xec,0x76,0xd1]
vwaddu.wx v24, v23, a3, v0.t

# Encoding: |110101|1|11001|11010|010|11011|1010111|
# CHECK: vwadd.wv v27, v25, v26
# CHECK-SAME: [0xd7,0x2d,0x9d,0xd7]
vwadd.wv v27, v25, v26
# Encoding: |110101|0|11100|11101|010|11110|1010111|
# CHECK: vwadd.wv v30, v28, v29, v0.t
# CHECK-SAME: [0x57,0xaf,0xce,0xd5]
vwadd.wv v30, v28, v29, v0.t

# Encoding: |110101|1|11111|01111|110|00000|1010111|
# CHECK: vwadd.wx v0, v31, a5
# CHECK-SAME: [0x57,0xe0,0xf7,0xd7]
vwadd.wx v0, v31, a5
# Encoding: |110101|0|00001|10001|110|00010|1010111|
# CHECK: vwadd.wx v2, v1, a7, v0.t
# CHECK-SAME: [0x57,0xe1,0x18,0xd4]
vwadd.wx v2, v1, a7, v0.t

# Encoding: |110110|1|00011|00100|010|00101|1010111|
# CHECK: vwsubu.wv v5, v3, v4
# CHECK-SAME: [0xd7,0x22,0x32,0xda]
vwsubu.wv v5, v3, v4
# Encoding: |110110|0|00110|00111|010|01000|1010111|
# CHECK: vwsubu.wv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0xa4,0x63,0xd8]
vwsubu.wv v8, v6, v7, v0.t

# Encoding: |110110|1|01001|10011|110|01010|1010111|
# CHECK: vwsubu.wx v10, v9, s3
# CHECK-SAME: [0x57,0xe5,0x99,0xda]
vwsubu.wx v10, v9, s3
# Encoding: |110110|0|01011|10101|110|01100|1010111|
# CHECK: vwsubu.wx v12, v11, s5, v0.t
# CHECK-SAME: [0x57,0xe6,0xba,0xd8]
vwsubu.wx v12, v11, s5, v0.t

# Encoding: |110111|1|01101|01110|010|01111|1010111|
# CHECK: vwsub.wv v15, v13, v14
# CHECK-SAME: [0xd7,0x27,0xd7,0xde]
vwsub.wv v15, v13, v14
# Encoding: |110111|0|10000|10001|010|10010|1010111|
# CHECK: vwsub.wv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0xa9,0x08,0xdd]
vwsub.wv v18, v16, v17, v0.t

# Encoding: |110111|1|10011|10111|110|10100|1010111|
# CHECK: vwsub.wx v20, v19, s7
# CHECK-SAME: [0x57,0xea,0x3b,0xdf]
vwsub.wx v20, v19, s7
# Encoding: |110111|0|10101|11001|110|10110|1010111|
# CHECK: vwsub.wx v22, v21, s9, v0.t
# CHECK-SAME: [0x57,0xeb,0x5c,0xdd]
vwsub.wx v22, v21, s9, v0.t

# Encoding: |111000|1|10111|11000|010|11001|1010111|
# CHECK: vwmulu.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x2c,0x7c,0xe3]
vwmulu.vv v25, v23, v24
# Encoding: |111000|0|11010|11011|010|11100|1010111|
# CHECK: vwmulu.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0xae,0xad,0xe1]
vwmulu.vv v28, v26, v27, v0.t

# Encoding: |111000|1|11101|11011|110|11110|1010111|
# CHECK: vwmulu.vx v30, v29, s11
# CHECK-SAME: [0x57,0xef,0xdd,0xe3]
vwmulu.vx v30, v29, s11
# Encoding: |111000|0|11111|11101|110|00000|1010111|
# CHECK: vwmulu.vx v0, v31, t4, v0.t
# CHECK-SAME: [0x57,0xe0,0xfe,0xe1]
vwmulu.vx v0, v31, t4, v0.t

# Encoding: |111010|1|00001|00010|010|00011|1010111|
# CHECK: vwmulsu.vv v3, v1, v2
# CHECK-SAME: [0xd7,0x21,0x11,0xea]
vwmulsu.vv v3, v1, v2
# Encoding: |111010|0|00100|00101|010|00110|1010111|
# CHECK: vwmulsu.vv v6, v4, v5, v0.t
# CHECK-SAME: [0x57,0xa3,0x42,0xe8]
vwmulsu.vv v6, v4, v5, v0.t

# Encoding: |111010|1|00111|11111|110|01000|1010111|
# CHECK: vwmulsu.vx v8, v7, t6
# CHECK-SAME: [0x57,0xe4,0x7f,0xea]
vwmulsu.vx v8, v7, t6
# Encoding: |111010|0|01001|00010|110|01010|1010111|
# CHECK: vwmulsu.vx v10, v9, sp, v0.t
# CHECK-SAME: [0x57,0x65,0x91,0xe8]
vwmulsu.vx v10, v9, sp, v0.t

# Encoding: |111011|1|01011|01100|010|01101|1010111|
# CHECK: vwmul.vv v13, v11, v12
# CHECK-SAME: [0xd7,0x26,0xb6,0xee]
vwmul.vv v13, v11, v12
# Encoding: |111011|0|01110|01111|010|10000|1010111|
# CHECK: vwmul.vv v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0xa8,0xe7,0xec]
vwmul.vv v16, v14, v15, v0.t

# Encoding: |111011|1|10001|00100|110|10010|1010111|
# CHECK: vwmul.vx v18, v17, tp
# CHECK-SAME: [0x57,0x69,0x12,0xef]
vwmul.vx v18, v17, tp
# Encoding: |111011|0|10011|00110|110|10100|1010111|
# CHECK: vwmul.vx v20, v19, t1, v0.t
# CHECK-SAME: [0x57,0x6a,0x33,0xed]
vwmul.vx v20, v19, t1, v0.t

# Encoding: |111100|1|10101|10110|010|10111|1010111|
# CHECK: vwmaccu.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x2b,0x5b,0xf3]
vwmaccu.vv v23, v21, v22
# Encoding: |111100|0|11000|11001|010|11010|1010111|
# CHECK: vwmaccu.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0xad,0x8c,0xf1]
vwmaccu.vv v26, v24, v25, v0.t

# Encoding: |111100|1|11011|01000|110|11100|1010111|
# CHECK: vwmaccu.vx v28, v27, s0
# CHECK-SAME: [0x57,0x6e,0xb4,0xf3]
vwmaccu.vx v28, v27, s0
# Encoding: |111100|0|11101|01010|110|11110|1010111|
# CHECK: vwmaccu.vx v30, v29, a0, v0.t
# CHECK-SAME: [0x57,0x6f,0xd5,0xf1]
vwmaccu.vx v30, v29, a0, v0.t

# Encoding: |111101|1|11111|00000|010|00001|1010111|
# CHECK: vwmacc.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x20,0xf0,0xf7]
vwmacc.vv v1, v31, v0
# Encoding: |111101|0|00010|00011|010|00100|1010111|
# CHECK: vwmacc.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0xa2,0x21,0xf4]
vwmacc.vv v4, v2, v3, v0.t

# Encoding: |111101|1|00101|01100|110|00110|1010111|
# CHECK: vwmacc.vx v6, v5, a2
# CHECK-SAME: [0x57,0x63,0x56,0xf6]
vwmacc.vx v6, v5, a2
# Encoding: |111101|0|00111|01110|110|01000|1010111|
# CHECK: vwmacc.vx v8, v7, a4, v0.t
# CHECK-SAME: [0x57,0x64,0x77,0xf4]
vwmacc.vx v8, v7, a4, v0.t

# Encoding: |111110|1|01001|01010|010|01011|1010111|
# CHECK: vwmsacu.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x25,0x95,0xfa]
vwmsacu.vv v11, v9, v10
# Encoding: |111110|0|01100|01101|010|01110|1010111|
# CHECK: vwmsacu.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0xa7,0xc6,0xf8]
vwmsacu.vv v14, v12, v13, v0.t

# Encoding: |111110|1|01111|10000|110|10000|1010111|
# CHECK: vwmsacu.vx v16, v15, a6
# CHECK-SAME: [0x57,0x68,0xf8,0xfa]
vwmsacu.vx v16, v15, a6
# Encoding: |111110|0|10001|10010|110|10010|1010111|
# CHECK: vwmsacu.vx v18, v17, s2, v0.t
# CHECK-SAME: [0x57,0x69,0x19,0xf9]
vwmsacu.vx v18, v17, s2, v0.t

# Encoding: |111111|1|10011|10100|010|10101|1010111|
# CHECK: vwmsac.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x2a,0x3a,0xff]
vwmsac.vv v21, v19, v20
# Encoding: |111111|0|10110|10111|010|11000|1010111|
# CHECK: vwmsac.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0xac,0x6b,0xfd]
vwmsac.vv v24, v22, v23, v0.t

# Encoding: |111111|1|11001|10100|110|11010|1010111|
# CHECK: vwmsac.vx v26, v25, s4
# CHECK-SAME: [0x57,0x6d,0x9a,0xff]
vwmsac.vx v26, v25, s4
# Encoding: |111111|0|11011|10110|110|11100|1010111|
# CHECK: vwmsac.vx v28, v27, s6, v0.t
# CHECK-SAME: [0x57,0x6e,0xbb,0xfd]
vwmsac.vx v28, v27, s6, v0.t

# Encoding: |000000|1|11101|11110|001|11111|1010111|
# CHECK: vfadd.vv v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0x03]
vfadd.vv v31, v29, v30
# Encoding: |000000|0|00000|00001|001|00010|1010111|
# CHECK: vfadd.vv v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0x00]
vfadd.vv v2, v0, v1, v0.t

# Encoding: |000000|1|00011|00000|101|00100|1010111|
# CHECK: vfadd.vf v4, v3, ft0
# CHECK-SAME: [0x57,0x52,0x30,0x02]
vfadd.vf v4, v3, ft0
# Encoding: |000000|0|00101|00001|101|00110|1010111|
# CHECK: vfadd.vf v6, v5, ft1, v0.t
# CHECK-SAME: [0x57,0xd3,0x50,0x00]
vfadd.vf v6, v5, ft1, v0.t

# Encoding: |000001|1|00111|01000|001|01001|1010111|
# CHECK: vfredsum.vs v9, v7, v8
# CHECK-SAME: [0xd7,0x14,0x74,0x06]
vfredsum.vs v9, v7, v8
# Encoding: |000001|0|01010|01011|001|01100|1010111|
# CHECK: vfredsum.vs v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0x96,0xa5,0x04]
vfredsum.vs v12, v10, v11, v0.t

# Encoding: |000010|1|01101|01110|001|01111|1010111|
# CHECK: vfsub.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0x0a]
vfsub.vv v15, v13, v14
# Encoding: |000010|0|10000|10001|001|10010|1010111|
# CHECK: vfsub.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0x09]
vfsub.vv v18, v16, v17, v0.t

# Encoding: |000010|1|10011|00010|101|10100|1010111|
# CHECK: vfsub.vf v20, v19, ft2
# CHECK-SAME: [0x57,0x5a,0x31,0x0b]
vfsub.vf v20, v19, ft2
# Encoding: |000010|0|10101|00011|101|10110|1010111|
# CHECK: vfsub.vf v22, v21, ft3, v0.t
# CHECK-SAME: [0x57,0xdb,0x51,0x09]
vfsub.vf v22, v21, ft3, v0.t

# Encoding: |000011|1|10111|11000|001|11001|1010111|
# CHECK: vfredosum.vs v25, v23, v24
# CHECK-SAME: [0xd7,0x1c,0x7c,0x0f]
vfredosum.vs v25, v23, v24
# Encoding: |000011|0|11010|11011|001|11100|1010111|
# CHECK: vfredosum.vs v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0x0d]
vfredosum.vs v28, v26, v27, v0.t

# Encoding: |000100|1|11101|11110|001|11111|1010111|
# CHECK: vfmin.vv v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0x13]
vfmin.vv v31, v29, v30
# Encoding: |000100|0|00000|00001|001|00010|1010111|
# CHECK: vfmin.vv v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0x10]
vfmin.vv v2, v0, v1, v0.t

# Encoding: |000100|1|00011|00100|101|00100|1010111|
# CHECK: vfmin.vf v4, v3, ft4
# CHECK-SAME: [0x57,0x52,0x32,0x12]
vfmin.vf v4, v3, ft4
# Encoding: |000100|0|00101|00101|101|00110|1010111|
# CHECK: vfmin.vf v6, v5, ft5, v0.t
# CHECK-SAME: [0x57,0xd3,0x52,0x10]
vfmin.vf v6, v5, ft5, v0.t

# Encoding: |000101|1|00111|01000|001|01001|1010111|
# CHECK: vfredmin.vs v9, v7, v8
# CHECK-SAME: [0xd7,0x14,0x74,0x16]
vfredmin.vs v9, v7, v8
# Encoding: |000101|0|01010|01011|001|01100|1010111|
# CHECK: vfredmin.vs v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0x96,0xa5,0x14]
vfredmin.vs v12, v10, v11, v0.t

# Encoding: |000110|1|01101|01110|001|01111|1010111|
# CHECK: vfmax.vv v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0x1a]
vfmax.vv v15, v13, v14
# Encoding: |000110|0|10000|10001|001|10010|1010111|
# CHECK: vfmax.vv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0x19]
vfmax.vv v18, v16, v17, v0.t

# Encoding: |000110|1|10011|00110|101|10100|1010111|
# CHECK: vfmax.vf v20, v19, ft6
# CHECK-SAME: [0x57,0x5a,0x33,0x1b]
vfmax.vf v20, v19, ft6
# Encoding: |000110|0|10101|00111|101|10110|1010111|
# CHECK: vfmax.vf v22, v21, ft7, v0.t
# CHECK-SAME: [0x57,0xdb,0x53,0x19]
vfmax.vf v22, v21, ft7, v0.t

# Encoding: |000111|1|10111|11000|001|11001|1010111|
# CHECK: vfredmax.vs v25, v23, v24
# CHECK-SAME: [0xd7,0x1c,0x7c,0x1f]
vfredmax.vs v25, v23, v24
# Encoding: |000111|0|11010|11011|001|11100|1010111|
# CHECK: vfredmax.vs v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0x1d]
vfredmax.vs v28, v26, v27, v0.t

# Encoding: |001000|1|11101|11110|001|11111|1010111|
# CHECK: vfsgnj.vv v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0x23]
vfsgnj.vv v31, v29, v30
# Encoding: |001000|0|00000|00001|001|00010|1010111|
# CHECK: vfsgnj.vv v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0x20]
vfsgnj.vv v2, v0, v1, v0.t

# Encoding: |001000|1|00011|01000|101|00100|1010111|
# CHECK: vfsgnj.vf v4, v3, fs0
# CHECK-SAME: [0x57,0x52,0x34,0x22]
vfsgnj.vf v4, v3, fs0
# Encoding: |001000|0|00101|01001|101|00110|1010111|
# CHECK: vfsgnj.vf v6, v5, fs1, v0.t
# CHECK-SAME: [0x57,0xd3,0x54,0x20]
vfsgnj.vf v6, v5, fs1, v0.t

# Encoding: |001001|1|00111|01000|001|01001|1010111|
# CHECK: vfsgnjn.vv v9, v7, v8
# CHECK-SAME: [0xd7,0x14,0x74,0x26]
vfsgnjn.vv v9, v7, v8
# Encoding: |001001|0|01010|01011|001|01100|1010111|
# CHECK: vfsgnjn.vv v12, v10, v11, v0.t
# CHECK-SAME: [0x57,0x96,0xa5,0x24]
vfsgnjn.vv v12, v10, v11, v0.t

# Encoding: |001001|1|01101|01010|101|01110|1010111|
# CHECK: vfsgnjn.vf v14, v13, fa0
# CHECK-SAME: [0x57,0x57,0xd5,0x26]
vfsgnjn.vf v14, v13, fa0
# Encoding: |001001|0|01111|01011|101|10000|1010111|
# CHECK: vfsgnjn.vf v16, v15, fa1, v0.t
# CHECK-SAME: [0x57,0xd8,0xf5,0x24]
vfsgnjn.vf v16, v15, fa1, v0.t

# Encoding: |001010|1|10001|10010|001|10011|1010111|
# CHECK: vfsgnjx.vv v19, v17, v18
# CHECK-SAME: [0xd7,0x19,0x19,0x2b]
vfsgnjx.vv v19, v17, v18
# Encoding: |001010|0|10100|10101|001|10110|1010111|
# CHECK: vfsgnjx.vv v22, v20, v21, v0.t
# CHECK-SAME: [0x57,0x9b,0x4a,0x29]
vfsgnjx.vv v22, v20, v21, v0.t

# Encoding: |001010|1|10111|01100|101|11000|1010111|
# CHECK: vfsgnjx.vf v24, v23, fa2
# CHECK-SAME: [0x57,0x5c,0x76,0x2b]
vfsgnjx.vf v24, v23, fa2
# Encoding: |001010|0|11001|01101|101|11010|1010111|
# CHECK: vfsgnjx.vf v26, v25, fa3, v0.t
# CHECK-SAME: [0x57,0xdd,0x96,0x29]
vfsgnjx.vf v26, v25, fa3, v0.t

# Encoding: |001100|1|11011|00000|001|01110|1010111|
# CHECK: vfmv.f.s fa4, v27
# CHECK-SAME: [0x57,0x17,0xb0,0x33]
vfmv.f.s fa4, v27

# Encoding: |001101|1|00000|01111|101|11100|1010111|
# CHECK: vfmv.s.f v28, fa5
# CHECK-SAME: [0x57,0xde,0x07,0x36]
vfmv.s.f v28, fa5

# Encoding: |010111|1|11101|10000|101|11110|1010111|
# CHECK: vfmerge.vf v30, v29, fa6
# CHECK-SAME: [0x57,0x5f,0xd8,0x5f]
vfmerge.vf v30, v29, fa6
# Encoding: |010111|0|11111|10001|101|00000|1010111|
# CHECK: vfmerge.vf v0, v31, fa7, v0.t
# CHECK-SAME: [0x57,0xd0,0xf8,0x5d]
vfmerge.vf v0, v31, fa7, v0.t

# Encoding: |011000|1|00001|00010|001|00011|1010111|
# CHECK: vfeq.vv v3, v1, v2
# CHECK-SAME: [0xd7,0x11,0x11,0x62]
vfeq.vv v3, v1, v2
# Encoding: |011000|0|00100|00101|001|00110|1010111|
# CHECK: vfeq.vv v6, v4, v5, v0.t
# CHECK-SAME: [0x57,0x93,0x42,0x60]
vfeq.vv v6, v4, v5, v0.t

# Encoding: |011000|1|00111|10010|101|01000|1010111|
# CHECK: vfeq.vf v8, v7, fs2
# CHECK-SAME: [0x57,0x54,0x79,0x62]
vfeq.vf v8, v7, fs2
# Encoding: |011000|0|01001|10011|101|01010|1010111|
# CHECK: vfeq.vf v10, v9, fs3, v0.t
# CHECK-SAME: [0x57,0xd5,0x99,0x60]
vfeq.vf v10, v9, fs3, v0.t

# Encoding: |011001|1|01011|01100|001|01101|1010111|
# CHECK: vfle.vv v13, v11, v12
# CHECK-SAME: [0xd7,0x16,0xb6,0x66]
vfle.vv v13, v11, v12
# Encoding: |011001|0|01110|01111|001|10000|1010111|
# CHECK: vfle.vv v16, v14, v15, v0.t
# CHECK-SAME: [0x57,0x98,0xe7,0x64]
vfle.vv v16, v14, v15, v0.t

# Encoding: |011001|1|10001|10100|101|10010|1010111|
# CHECK: vfle.vf v18, v17, fs4
# CHECK-SAME: [0x57,0x59,0x1a,0x67]
vfle.vf v18, v17, fs4
# Encoding: |011001|0|10011|10101|101|10100|1010111|
# CHECK: vfle.vf v20, v19, fs5, v0.t
# CHECK-SAME: [0x57,0xda,0x3a,0x65]
vfle.vf v20, v19, fs5, v0.t

# Encoding: |011010|1|10101|10110|001|10111|1010111|
# CHECK: vford.vv v23, v21, v22
# CHECK-SAME: [0xd7,0x1b,0x5b,0x6b]
vford.vv v23, v21, v22
# Encoding: |011010|0|11000|11001|001|11010|1010111|
# CHECK: vford.vv v26, v24, v25, v0.t
# CHECK-SAME: [0x57,0x9d,0x8c,0x69]
vford.vv v26, v24, v25, v0.t

# Encoding: |011010|1|11011|10110|101|11100|1010111|
# CHECK: vford.vf v28, v27, fs6
# CHECK-SAME: [0x57,0x5e,0xbb,0x6b]
vford.vf v28, v27, fs6
# Encoding: |011010|0|11101|10111|101|11110|1010111|
# CHECK: vford.vf v30, v29, fs7, v0.t
# CHECK-SAME: [0x57,0xdf,0xdb,0x69]
vford.vf v30, v29, fs7, v0.t

# Encoding: |011011|1|11111|00000|001|00001|1010111|
# CHECK: vflt.vv v1, v31, v0
# CHECK-SAME: [0xd7,0x10,0xf0,0x6f]
vflt.vv v1, v31, v0
# Encoding: |011011|0|00010|00011|001|00100|1010111|
# CHECK: vflt.vv v4, v2, v3, v0.t
# CHECK-SAME: [0x57,0x92,0x21,0x6c]
vflt.vv v4, v2, v3, v0.t

# Encoding: |011011|1|00101|11000|101|00110|1010111|
# CHECK: vflt.vf v6, v5, fs8
# CHECK-SAME: [0x57,0x53,0x5c,0x6e]
vflt.vf v6, v5, fs8
# Encoding: |011011|0|00111|11001|101|01000|1010111|
# CHECK: vflt.vf v8, v7, fs9, v0.t
# CHECK-SAME: [0x57,0xd4,0x7c,0x6c]
vflt.vf v8, v7, fs9, v0.t

# Encoding: |011100|1|01001|01010|001|01011|1010111|
# CHECK: vfne.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x15,0x95,0x72]
vfne.vv v11, v9, v10
# Encoding: |011100|0|01100|01101|001|01110|1010111|
# CHECK: vfne.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x97,0xc6,0x70]
vfne.vv v14, v12, v13, v0.t

# Encoding: |011100|1|01111|11010|101|10000|1010111|
# CHECK: vfne.vf v16, v15, fs10
# CHECK-SAME: [0x57,0x58,0xfd,0x72]
vfne.vf v16, v15, fs10
# Encoding: |011100|0|10001|11011|101|10010|1010111|
# CHECK: vfne.vf v18, v17, fs11, v0.t
# CHECK-SAME: [0x57,0xd9,0x1d,0x71]
vfne.vf v18, v17, fs11, v0.t

# Encoding: |011101|1|10011|11100|101|10100|1010111|
# CHECK: vfgt.vf v20, v19, ft8
# CHECK-SAME: [0x57,0x5a,0x3e,0x77]
vfgt.vf v20, v19, ft8
# Encoding: |011101|0|10101|11101|101|10110|1010111|
# CHECK: vfgt.vf v22, v21, ft9, v0.t
# CHECK-SAME: [0x57,0xdb,0x5e,0x75]
vfgt.vf v22, v21, ft9, v0.t

# Encoding: |011111|1|10111|11110|101|11000|1010111|
# CHECK: vfge.vf v24, v23, ft10
# CHECK-SAME: [0x57,0x5c,0x7f,0x7f]
vfge.vf v24, v23, ft10
# Encoding: |011111|0|11001|00000|101|11010|1010111|
# CHECK: vfge.vf v26, v25, ft0, v0.t
# CHECK-SAME: [0x57,0x5d,0x90,0x7d]
vfge.vf v26, v25, ft0, v0.t

# Encoding: |100000|1|11011|11100|001|11101|1010111|
# CHECK: vfdiv.vv v29, v27, v28
# CHECK-SAME: [0xd7,0x1e,0xbe,0x83]
vfdiv.vv v29, v27, v28
# Encoding: |100000|0|11110|11111|001|00000|1010111|
# CHECK: vfdiv.vv v0, v30, v31, v0.t
# CHECK-SAME: [0x57,0x90,0xef,0x81]
vfdiv.vv v0, v30, v31, v0.t

# Encoding: |100000|1|00001|00001|101|00010|1010111|
# CHECK: vfdiv.vf v2, v1, ft1
# CHECK-SAME: [0x57,0xd1,0x10,0x82]
vfdiv.vf v2, v1, ft1
# Encoding: |100000|0|00011|00010|101|00100|1010111|
# CHECK: vfdiv.vf v4, v3, ft2, v0.t
# CHECK-SAME: [0x57,0x52,0x31,0x80]
vfdiv.vf v4, v3, ft2, v0.t

# Encoding: |100001|1|00101|00011|101|00110|1010111|
# CHECK: vfrdiv.vf v6, v5, ft3
# CHECK-SAME: [0x57,0xd3,0x51,0x86]
vfrdiv.vf v6, v5, ft3
# Encoding: |100001|0|00111|00100|101|01000|1010111|
# CHECK: vfrdiv.vf v8, v7, ft4, v0.t
# CHECK-SAME: [0x57,0x54,0x72,0x84]
vfrdiv.vf v8, v7, ft4, v0.t

# Encoding: |100100|1|01001|01010|001|01011|1010111|
# CHECK: vfmul.vv v11, v9, v10
# CHECK-SAME: [0xd7,0x15,0x95,0x92]
vfmul.vv v11, v9, v10
# Encoding: |100100|0|01100|01101|001|01110|1010111|
# CHECK: vfmul.vv v14, v12, v13, v0.t
# CHECK-SAME: [0x57,0x97,0xc6,0x90]
vfmul.vv v14, v12, v13, v0.t

# Encoding: |100100|1|01111|00101|101|10000|1010111|
# CHECK: vfmul.vf v16, v15, ft5
# CHECK-SAME: [0x57,0xd8,0xf2,0x92]
vfmul.vf v16, v15, ft5
# Encoding: |100100|0|10001|00110|101|10010|1010111|
# CHECK: vfmul.vf v18, v17, ft6, v0.t
# CHECK-SAME: [0x57,0x59,0x13,0x91]
vfmul.vf v18, v17, ft6, v0.t

# Encoding: |101000|1|10011|10100|001|10101|1010111|
# CHECK: vfmadd.vv v21, v20, v19
# CHECK-SAME: [0xd7,0x1a,0x3a,0xa3]
vfmadd.vv v21, v20, v19
# Encoding: |101000|0|10110|10111|001|11000|1010111|
# CHECK: vfmadd.vv v24, v23, v22, v0.t
# CHECK-SAME: [0x57,0x9c,0x6b,0xa1]
vfmadd.vv v24, v23, v22, v0.t

# Encoding: |101000|1|11001|00111|101|11010|1010111|
# CHECK: vfmadd.vf v26, ft7, v25
# CHECK-SAME: [0x57,0xdd,0x93,0xa3]
vfmadd.vf v26, ft7, v25
# Encoding: |101000|0|11011|01000|101|11100|1010111|
# CHECK: vfmadd.vf v28, fs0, v27, v0.t
# CHECK-SAME: [0x57,0x5e,0xb4,0xa1]
vfmadd.vf v28, fs0, v27, v0.t

# Encoding: |101001|1|11101|11110|001|11111|1010111|
# CHECK: vfnmadd.vv v31, v30, v29
# CHECK-SAME: [0xd7,0x1f,0xdf,0xa7]
vfnmadd.vv v31, v30, v29
# Encoding: |101001|0|00000|00001|001|00010|1010111|
# CHECK: vfnmadd.vv v2, v1, v0, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0xa4]
vfnmadd.vv v2, v1, v0, v0.t

# Encoding: |101001|1|00011|01001|101|00100|1010111|
# CHECK: vfnmadd.vf v4, fs1, v3
# CHECK-SAME: [0x57,0xd2,0x34,0xa6]
vfnmadd.vf v4, fs1, v3
# Encoding: |101001|0|00101|01010|101|00110|1010111|
# CHECK: vfnmadd.vf v6, fa0, v5, v0.t
# CHECK-SAME: [0x57,0x53,0x55,0xa4]
vfnmadd.vf v6, fa0, v5, v0.t

# Encoding: |101010|1|00111|01000|001|01001|1010111|
# CHECK: vfmsub.vv v9, v8, v7
# CHECK-SAME: [0xd7,0x14,0x74,0xaa]
vfmsub.vv v9, v8, v7
# Encoding: |101010|0|01010|01011|001|01100|1010111|
# CHECK: vfmsub.vv v12, v11, v10, v0.t
# CHECK-SAME: [0x57,0x96,0xa5,0xa8]
vfmsub.vv v12, v11, v10, v0.t

# Encoding: |101010|1|01101|01011|101|01110|1010111|
# CHECK: vfmsub.vf v14, fa1, v13
# CHECK-SAME: [0x57,0xd7,0xd5,0xaa]
vfmsub.vf v14, fa1, v13
# Encoding: |101010|0|01111|01100|101|10000|1010111|
# CHECK: vfmsub.vf v16, fa2, v15, v0.t
# CHECK-SAME: [0x57,0x58,0xf6,0xa8]
vfmsub.vf v16, fa2, v15, v0.t

# Encoding: |101011|1|10001|10010|001|10011|1010111|
# CHECK: vfnmsub.vv v19, v18, v17
# CHECK-SAME: [0xd7,0x19,0x19,0xaf]
vfnmsub.vv v19, v18, v17
# Encoding: |101011|0|10100|10101|001|10110|1010111|
# CHECK: vfnmsub.vv v22, v21, v20, v0.t
# CHECK-SAME: [0x57,0x9b,0x4a,0xad]
vfnmsub.vv v22, v21, v20, v0.t

# Encoding: |101011|1|10111|01101|101|11000|1010111|
# CHECK: vfnmsub.vf v24, fa3, v23
# CHECK-SAME: [0x57,0xdc,0x76,0xaf]
vfnmsub.vf v24, fa3, v23
# Encoding: |101011|0|11001|01110|101|11010|1010111|
# CHECK: vfnmsub.vf v26, fa4, v25, v0.t
# CHECK-SAME: [0x57,0x5d,0x97,0xad]
vfnmsub.vf v26, fa4, v25, v0.t

# Encoding: |101100|1|11011|11100|001|11101|1010111|
# CHECK: vfmacc.vv v29, v28, v27
# CHECK-SAME: [0xd7,0x1e,0xbe,0xb3]
vfmacc.vv v29, v28, v27
# Encoding: |101100|0|11110|11111|001|00000|1010111|
# CHECK: vfmacc.vv v0, v31, v30, v0.t
# CHECK-SAME: [0x57,0x90,0xef,0xb1]
vfmacc.vv v0, v31, v30, v0.t

# Encoding: |101100|1|00001|01111|101|00010|1010111|
# CHECK: vfmacc.vf v2, fa5, v1
# CHECK-SAME: [0x57,0xd1,0x17,0xb2]
vfmacc.vf v2, fa5, v1
# Encoding: |101100|0|00011|10000|101|00100|1010111|
# CHECK: vfmacc.vf v4, fa6, v3, v0.t
# CHECK-SAME: [0x57,0x52,0x38,0xb0]
vfmacc.vf v4, fa6, v3, v0.t

# Encoding: |101101|1|00101|00110|001|00111|1010111|
# CHECK: vfnmacc.vv v7, v6, v5
# CHECK-SAME: [0xd7,0x13,0x53,0xb6]
vfnmacc.vv v7, v6, v5
# Encoding: |101101|0|01000|01001|001|01010|1010111|
# CHECK: vfnmacc.vv v10, v9, v8, v0.t
# CHECK-SAME: [0x57,0x95,0x84,0xb4]
vfnmacc.vv v10, v9, v8, v0.t

# Encoding: |101101|1|01011|10001|101|01100|1010111|
# CHECK: vfnmacc.vf v12, fa7, v11
# CHECK-SAME: [0x57,0xd6,0xb8,0xb6]
vfnmacc.vf v12, fa7, v11
# Encoding: |101101|0|01101|10010|101|01110|1010111|
# CHECK: vfnmacc.vf v14, fs2, v13, v0.t
# CHECK-SAME: [0x57,0x57,0xd9,0xb4]
vfnmacc.vf v14, fs2, v13, v0.t

# Encoding: |101110|1|01111|10000|001|10001|1010111|
# CHECK: vfmsac.vv v17, v16, v15
# CHECK-SAME: [0xd7,0x18,0xf8,0xba]
vfmsac.vv v17, v16, v15
# Encoding: |101110|0|10010|10011|001|10100|1010111|
# CHECK: vfmsac.vv v20, v19, v18, v0.t
# CHECK-SAME: [0x57,0x9a,0x29,0xb9]
vfmsac.vv v20, v19, v18, v0.t

# Encoding: |101110|1|10101|10011|101|10110|1010111|
# CHECK: vfmsac.vf v22, fs3, v21
# CHECK-SAME: [0x57,0xdb,0x59,0xbb]
vfmsac.vf v22, fs3, v21
# Encoding: |101110|0|10111|10100|101|11000|1010111|
# CHECK: vfmsac.vf v24, fs4, v23, v0.t
# CHECK-SAME: [0x57,0x5c,0x7a,0xb9]
vfmsac.vf v24, fs4, v23, v0.t

# Encoding: |101111|1|11001|11010|001|11011|1010111|
# CHECK: vfnmsac.vv v27, v26, v25
# CHECK-SAME: [0xd7,0x1d,0x9d,0xbf]
vfnmsac.vv v27, v26, v25
# Encoding: |101111|0|11100|11101|001|11110|1010111|
# CHECK: vfnmsac.vv v30, v29, v28, v0.t
# CHECK-SAME: [0x57,0x9f,0xce,0xbd]
vfnmsac.vv v30, v29, v28, v0.t

# Encoding: |101111|1|11111|10101|101|00000|1010111|
# CHECK: vfnmsac.vf v0, fs5, v31
# CHECK-SAME: [0x57,0xd0,0xfa,0xbf]
vfnmsac.vf v0, fs5, v31
# Encoding: |101111|0|00001|10110|101|00010|1010111|
# CHECK: vfnmsac.vf v2, fs6, v1, v0.t
# CHECK-SAME: [0x57,0x51,0x1b,0xbc]
vfnmsac.vf v2, fs6, v1, v0.t

# Encoding: |110000|1|00011|00100|001|00101|1010111|
# CHECK: vfwadd.vv v5, v3, v4
# CHECK-SAME: [0xd7,0x12,0x32,0xc2]
vfwadd.vv v5, v3, v4
# Encoding: |110000|0|00110|00111|001|01000|1010111|
# CHECK: vfwadd.vv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0xc0]
vfwadd.vv v8, v6, v7, v0.t

# Encoding: |110000|1|01001|10111|101|01010|1010111|
# CHECK: vfwadd.vf v10, v9, fs7
# CHECK-SAME: [0x57,0xd5,0x9b,0xc2]
vfwadd.vf v10, v9, fs7
# Encoding: |110000|0|01011|11000|101|01100|1010111|
# CHECK: vfwadd.vf v12, v11, fs8, v0.t
# CHECK-SAME: [0x57,0x56,0xbc,0xc0]
vfwadd.vf v12, v11, fs8, v0.t

# Encoding: |110001|1|01101|01110|001|01111|1010111|
# CHECK: vfwredsum.vs v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0xc6]
vfwredsum.vs v15, v13, v14
# Encoding: |110001|0|10000|10001|001|10010|1010111|
# CHECK: vfwredsum.vs v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0xc5]
vfwredsum.vs v18, v16, v17, v0.t

# Encoding: |110010|1|10011|10100|001|10101|1010111|
# CHECK: vfwsub.vv v21, v19, v20
# CHECK-SAME: [0xd7,0x1a,0x3a,0xcb]
vfwsub.vv v21, v19, v20
# Encoding: |110010|0|10110|10111|001|11000|1010111|
# CHECK: vfwsub.vv v24, v22, v23, v0.t
# CHECK-SAME: [0x57,0x9c,0x6b,0xc9]
vfwsub.vv v24, v22, v23, v0.t

# Encoding: |110010|1|11001|11001|101|11010|1010111|
# CHECK: vfwsub.vf v26, v25, fs9
# CHECK-SAME: [0x57,0xdd,0x9c,0xcb]
vfwsub.vf v26, v25, fs9
# Encoding: |110010|0|11011|11010|101|11100|1010111|
# CHECK: vfwsub.vf v28, v27, fs10, v0.t
# CHECK-SAME: [0x57,0x5e,0xbd,0xc9]
vfwsub.vf v28, v27, fs10, v0.t

# Encoding: |110011|1|11101|11110|001|11111|1010111|
# CHECK: vfwredosum.vs v31, v29, v30
# CHECK-SAME: [0xd7,0x1f,0xdf,0xcf]
vfwredosum.vs v31, v29, v30
# Encoding: |110011|0|00000|00001|001|00010|1010111|
# CHECK: vfwredosum.vs v2, v0, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x00,0xcc]
vfwredosum.vs v2, v0, v1, v0.t

# Encoding: |110100|1|00011|00100|001|00101|1010111|
# CHECK: vfwadd.wv v5, v3, v4
# CHECK-SAME: [0xd7,0x12,0x32,0xd2]
vfwadd.wv v5, v3, v4
# Encoding: |110100|0|00110|00111|001|01000|1010111|
# CHECK: vfwadd.wv v8, v6, v7, v0.t
# CHECK-SAME: [0x57,0x94,0x63,0xd0]
vfwadd.wv v8, v6, v7, v0.t

# Encoding: |110100|1|01001|11011|101|01010|1010111|
# CHECK: vfwadd.wf v10, v9, fs11
# CHECK-SAME: [0x57,0xd5,0x9d,0xd2]
vfwadd.wf v10, v9, fs11
# Encoding: |110100|0|01011|11100|101|01100|1010111|
# CHECK: vfwadd.wf v12, v11, ft8, v0.t
# CHECK-SAME: [0x57,0x56,0xbe,0xd0]
vfwadd.wf v12, v11, ft8, v0.t

# Encoding: |110110|1|01101|01110|001|01111|1010111|
# CHECK: vfwsub.wv v15, v13, v14
# CHECK-SAME: [0xd7,0x17,0xd7,0xda]
vfwsub.wv v15, v13, v14
# Encoding: |110110|0|10000|10001|001|10010|1010111|
# CHECK: vfwsub.wv v18, v16, v17, v0.t
# CHECK-SAME: [0x57,0x99,0x08,0xd9]
vfwsub.wv v18, v16, v17, v0.t

# Encoding: |110110|1|10011|11101|101|10100|1010111|
# CHECK: vfwsub.wf v20, v19, ft9
# CHECK-SAME: [0x57,0xda,0x3e,0xdb]
vfwsub.wf v20, v19, ft9
# Encoding: |110110|0|10101|11110|101|10110|1010111|
# CHECK: vfwsub.wf v22, v21, ft10, v0.t
# CHECK-SAME: [0x57,0x5b,0x5f,0xd9]
vfwsub.wf v22, v21, ft10, v0.t

# Encoding: |111000|1|10111|11000|001|11001|1010111|
# CHECK: vfwmul.vv v25, v23, v24
# CHECK-SAME: [0xd7,0x1c,0x7c,0xe3]
vfwmul.vv v25, v23, v24
# Encoding: |111000|0|11010|11011|001|11100|1010111|
# CHECK: vfwmul.vv v28, v26, v27, v0.t
# CHECK-SAME: [0x57,0x9e,0xad,0xe1]
vfwmul.vv v28, v26, v27, v0.t

# Encoding: |111000|1|11101|00000|101|11110|1010111|
# CHECK: vfwmul.vf v30, v29, ft0
# CHECK-SAME: [0x57,0x5f,0xd0,0xe3]
vfwmul.vf v30, v29, ft0
# Encoding: |111000|0|11111|00001|101|00000|1010111|
# CHECK: vfwmul.vf v0, v31, ft1, v0.t
# CHECK-SAME: [0x57,0xd0,0xf0,0xe1]
vfwmul.vf v0, v31, ft1, v0.t

# Encoding: |111001|1|00001|00010|001|00011|1010111|
# CHECK: vfdot.vv v3, v1, v2
# CHECK-SAME: [0xd7,0x11,0x11,0xe6]
vfdot.vv v3, v1, v2
# Encoding: |111001|0|00100|00101|001|00110|1010111|
# CHECK: vfdot.vv v6, v4, v5, v0.t
# CHECK-SAME: [0x57,0x93,0x42,0xe4]
vfdot.vv v6, v4, v5, v0.t

# Encoding: |111100|1|00111|01000|001|01001|1010111|
# CHECK: vfwmacc.vv v9, v8, v7
# CHECK-SAME: [0xd7,0x14,0x74,0xf2]
vfwmacc.vv v9, v8, v7
# Encoding: |111100|0|01010|01011|001|01100|1010111|
# CHECK: vfwmacc.vv v12, v11, v10, v0.t
# CHECK-SAME: [0x57,0x96,0xa5,0xf0]
vfwmacc.vv v12, v11, v10, v0.t

# Encoding: |111100|1|01101|00010|101|01110|1010111|
# CHECK: vfwmacc.vf v14, ft2, v13
# CHECK-SAME: [0x57,0x57,0xd1,0xf2]
vfwmacc.vf v14, ft2, v13
# Encoding: |111100|0|01111|00011|101|10000|1010111|
# CHECK: vfwmacc.vf v16, ft3, v15, v0.t
# CHECK-SAME: [0x57,0xd8,0xf1,0xf0]
vfwmacc.vf v16, ft3, v15, v0.t

# Encoding: |111101|1|10001|10010|001|10011|1010111|
# CHECK: vfwnmacc.vv v19, v18, v17
# CHECK-SAME: [0xd7,0x19,0x19,0xf7]
vfwnmacc.vv v19, v18, v17
# Encoding: |111101|0|10100|10101|001|10110|1010111|
# CHECK: vfwnmacc.vv v22, v21, v20, v0.t
# CHECK-SAME: [0x57,0x9b,0x4a,0xf5]
vfwnmacc.vv v22, v21, v20, v0.t

# Encoding: |111101|1|10111|00100|101|11000|1010111|
# CHECK: vfwnmacc.vf v24, ft4, v23
# CHECK-SAME: [0x57,0x5c,0x72,0xf7]
vfwnmacc.vf v24, ft4, v23
# Encoding: |111101|0|11001|00101|101|11010|1010111|
# CHECK: vfwnmacc.vf v26, ft5, v25, v0.t
# CHECK-SAME: [0x57,0xdd,0x92,0xf5]
vfwnmacc.vf v26, ft5, v25, v0.t

# Encoding: |111110|1|11011|11100|001|11101|1010111|
# CHECK: vfwmsac.vv v29, v28, v27
# CHECK-SAME: [0xd7,0x1e,0xbe,0xfb]
vfwmsac.vv v29, v28, v27
# Encoding: |111110|0|11110|11111|001|00000|1010111|
# CHECK: vfwmsac.vv v0, v31, v30, v0.t
# CHECK-SAME: [0x57,0x90,0xef,0xf9]
vfwmsac.vv v0, v31, v30, v0.t

# Encoding: |111110|1|00001|00110|101|00010|1010111|
# CHECK: vfwmsac.vf v2, ft6, v1
# CHECK-SAME: [0x57,0x51,0x13,0xfa]
vfwmsac.vf v2, ft6, v1
# Encoding: |111110|0|00011|00111|101|00100|1010111|
# CHECK: vfwmsac.vf v4, ft7, v3, v0.t
# CHECK-SAME: [0x57,0xd2,0x33,0xf8]
vfwmsac.vf v4, ft7, v3, v0.t

# Encoding: |111111|1|00101|00110|001|00111|1010111|
# CHECK: vfwnmsac.vv v7, v6, v5
# CHECK-SAME: [0xd7,0x13,0x53,0xfe]
vfwnmsac.vv v7, v6, v5
# Encoding: |111111|0|01000|01001|001|01010|1010111|
# CHECK: vfwnmsac.vv v10, v9, v8, v0.t
# CHECK-SAME: [0x57,0x95,0x84,0xfc]
vfwnmsac.vv v10, v9, v8, v0.t

# Encoding: |111111|1|01011|01000|101|01100|1010111|
# CHECK: vfwnmsac.vf v12, fs0, v11
# CHECK-SAME: [0x57,0x56,0xb4,0xfe]
vfwnmsac.vf v12, fs0, v11
# Encoding: |111111|0|01101|01001|101|01110|1010111|
# CHECK: vfwnmsac.vf v14, fs1, v13, v0.t
# CHECK-SAME: [0x57,0xd7,0xd4,0xfc]
vfwnmsac.vf v14, fs1, v13, v0.t

# Encoding: |100011|1|01111|00000|001|10000|1010111|
# CHECK: vfsqrt.v v16, v15
# CHECK-SAME: [0x57,0x18,0xf0,0x8e]
vfsqrt.v v16, v15
# Encoding: |100011|0|10001|00000|001|10010|1010111|
# CHECK: vfsqrt.v v18, v17, v0.t
# CHECK-SAME: [0x57,0x19,0x10,0x8d]
vfsqrt.v v18, v17, v0.t

# Encoding: |100011|1|10011|10000|001|10100|1010111|
# CHECK: vfclass.v v20, v19
# CHECK-SAME: [0x57,0x1a,0x38,0x8f]
vfclass.v v20, v19
# Encoding: |100011|0|10101|10000|001|10110|1010111|
# CHECK: vfclass.v v22, v21, v0.t
# CHECK-SAME: [0x57,0x1b,0x58,0x8d]
vfclass.v v22, v21, v0.t

# Encoding: |100010|1|10111|00000|001|11000|1010111|
# CHECK: vfcvt.xu.f.v v24, v23
# CHECK-SAME: [0x57,0x1c,0x70,0x8b]
vfcvt.xu.f.v v24, v23
# Encoding: |100010|0|11001|00000|001|11010|1010111|
# CHECK: vfcvt.xu.f.v v26, v25, v0.t
# CHECK-SAME: [0x57,0x1d,0x90,0x89]
vfcvt.xu.f.v v26, v25, v0.t

# Encoding: |100010|1|11011|00001|001|11100|1010111|
# CHECK: vfcvt.x.f.v v28, v27
# CHECK-SAME: [0x57,0x9e,0xb0,0x8b]
vfcvt.x.f.v v28, v27
# Encoding: |100010|0|11101|00001|001|11110|1010111|
# CHECK: vfcvt.x.f.v v30, v29, v0.t
# CHECK-SAME: [0x57,0x9f,0xd0,0x89]
vfcvt.x.f.v v30, v29, v0.t

# Encoding: |100010|1|11111|00010|001|00000|1010111|
# CHECK: vfcvt.f.xu.v v0, v31
# CHECK-SAME: [0x57,0x10,0xf1,0x8b]
vfcvt.f.xu.v v0, v31
# Encoding: |100010|0|00001|00010|001|00010|1010111|
# CHECK: vfcvt.f.xu.v v2, v1, v0.t
# CHECK-SAME: [0x57,0x11,0x11,0x88]
vfcvt.f.xu.v v2, v1, v0.t

# Encoding: |100010|1|00011|00011|001|00100|1010111|
# CHECK: vfcvt.f.x.v v4, v3
# CHECK-SAME: [0x57,0x92,0x31,0x8a]
vfcvt.f.x.v v4, v3
# Encoding: |100010|0|00101|00011|001|00110|1010111|
# CHECK: vfcvt.f.x.v v6, v5, v0.t
# CHECK-SAME: [0x57,0x93,0x51,0x88]
vfcvt.f.x.v v6, v5, v0.t

# Encoding: |100010|1|00111|01000|001|01000|1010111|
# CHECK: vfwcvt.xu.f.v v8, v7
# CHECK-SAME: [0x57,0x14,0x74,0x8a]
vfwcvt.xu.f.v v8, v7
# Encoding: |100010|0|01001|01000|001|01010|1010111|
# CHECK: vfwcvt.xu.f.v v10, v9, v0.t
# CHECK-SAME: [0x57,0x15,0x94,0x88]
vfwcvt.xu.f.v v10, v9, v0.t

# Encoding: |100010|1|01011|01001|001|01100|1010111|
# CHECK: vfwcvt.x.f.v v12, v11
# CHECK-SAME: [0x57,0x96,0xb4,0x8a]
vfwcvt.x.f.v v12, v11
# Encoding: |100010|0|01101|01001|001|01110|1010111|
# CHECK: vfwcvt.x.f.v v14, v13, v0.t
# CHECK-SAME: [0x57,0x97,0xd4,0x88]
vfwcvt.x.f.v v14, v13, v0.t

# Encoding: |100010|1|01111|01010|001|10000|1010111|
# CHECK: vfwcvt.f.xu.v v16, v15
# CHECK-SAME: [0x57,0x18,0xf5,0x8a]
vfwcvt.f.xu.v v16, v15
# Encoding: |100010|0|10001|01010|001|10010|1010111|
# CHECK: vfwcvt.f.xu.v v18, v17, v0.t
# CHECK-SAME: [0x57,0x19,0x15,0x89]
vfwcvt.f.xu.v v18, v17, v0.t

# Encoding: |100010|1|10011|01011|001|10100|1010111|
# CHECK: vfwcvt.f.x.v v20, v19
# CHECK-SAME: [0x57,0x9a,0x35,0x8b]
vfwcvt.f.x.v v20, v19
# Encoding: |100010|0|10101|01011|001|10110|1010111|
# CHECK: vfwcvt.f.x.v v22, v21, v0.t
# CHECK-SAME: [0x57,0x9b,0x55,0x89]
vfwcvt.f.x.v v22, v21, v0.t

# Encoding: |100010|1|10111|01100|001|11000|1010111|
# CHECK: vfwcvt.f.f.v v24, v23
# CHECK-SAME: [0x57,0x1c,0x76,0x8b]
vfwcvt.f.f.v v24, v23
# Encoding: |100010|0|11001|01100|001|11010|1010111|
# CHECK: vfwcvt.f.f.v v26, v25, v0.t
# CHECK-SAME: [0x57,0x1d,0x96,0x89]
vfwcvt.f.f.v v26, v25, v0.t

# Encoding: |100010|1|11011|10000|001|11100|1010111|
# CHECK: vfncvt.xu.f.v v28, v27
# CHECK-SAME: [0x57,0x1e,0xb8,0x8b]
vfncvt.xu.f.v v28, v27
# Encoding: |100010|0|11101|10000|001|11110|1010111|
# CHECK: vfncvt.xu.f.v v30, v29, v0.t
# CHECK-SAME: [0x57,0x1f,0xd8,0x89]
vfncvt.xu.f.v v30, v29, v0.t

# Encoding: |100010|1|11111|10001|001|00000|1010111|
# CHECK: vfncvt.x.f.v v0, v31
# CHECK-SAME: [0x57,0x90,0xf8,0x8b]
vfncvt.x.f.v v0, v31
# Encoding: |100010|0|00001|10001|001|00010|1010111|
# CHECK: vfncvt.x.f.v v2, v1, v0.t
# CHECK-SAME: [0x57,0x91,0x18,0x88]
vfncvt.x.f.v v2, v1, v0.t

# Encoding: |100010|1|00011|10010|001|00100|1010111|
# CHECK: vfncvt.f.xu.v v4, v3
# CHECK-SAME: [0x57,0x12,0x39,0x8a]
vfncvt.f.xu.v v4, v3
# Encoding: |100010|0|00101|10010|001|00110|1010111|
# CHECK: vfncvt.f.xu.v v6, v5, v0.t
# CHECK-SAME: [0x57,0x13,0x59,0x88]
vfncvt.f.xu.v v6, v5, v0.t

# Encoding: |100010|1|00111|10011|001|01000|1010111|
# CHECK: vfncvt.f.x.v v8, v7
# CHECK-SAME: [0x57,0x94,0x79,0x8a]
vfncvt.f.x.v v8, v7
# Encoding: |100010|0|01001|10011|001|01010|1010111|
# CHECK: vfncvt.f.x.v v10, v9, v0.t
# CHECK-SAME: [0x57,0x95,0x99,0x88]
vfncvt.f.x.v v10, v9, v0.t

# Encoding: |100010|1|01011|10100|001|01100|1010111|
# CHECK: vfncvt.f.f.v v12, v11
# CHECK-SAME: [0x57,0x16,0xba,0x8a]
vfncvt.f.f.v v12, v11
# Encoding: |100010|0|01101|10100|001|01110|1010111|
# CHECK: vfncvt.f.f.v v14, v13, v0.t
# CHECK-SAME: [0x57,0x17,0xda,0x88]
vfncvt.f.f.v v14, v13, v0.t

# Encoding: |010110|1|01111|00001|010|10000|1010111|
# CHECK: vmsbf.m v16, v15
# CHECK-SAME: [0x57,0xa8,0xf0,0x5a]
vmsbf.m v16, v15
# Encoding: |010110|0|10001|00001|010|10010|1010111|
# CHECK: vmsbf.m v18, v17, v0.t
# CHECK-SAME: [0x57,0xa9,0x10,0x59]
vmsbf.m v18, v17, v0.t

# Encoding: |010110|1|10011|00010|010|10100|1010111|
# CHECK: vmsof.m v20, v19
# CHECK-SAME: [0x57,0x2a,0x31,0x5b]
vmsof.m v20, v19
# Encoding: |010110|0|10101|00010|010|10110|1010111|
# CHECK: vmsof.m v22, v21, v0.t
# CHECK-SAME: [0x57,0x2b,0x51,0x59]
vmsof.m v22, v21, v0.t

# Encoding: |010110|1|10111|00011|010|11000|1010111|
# CHECK: vmsif.m v24, v23
# CHECK-SAME: [0x57,0xac,0x71,0x5b]
vmsif.m v24, v23
# Encoding: |010110|0|11001|00011|010|11010|1010111|
# CHECK: vmsif.m v26, v25, v0.t
# CHECK-SAME: [0x57,0xad,0x91,0x59]
vmsif.m v26, v25, v0.t

# Encoding: |010110|1|11011|10000|010|11100|1010111|
# CHECK: vmiota.m v28, v27
# CHECK-SAME: [0x57,0x2e,0xb8,0x5b]
vmiota.m v28, v27
# Encoding: |010110|0|11101|10000|010|11110|1010111|
# CHECK: vmiota.m v30, v29, v0.t
# CHECK-SAME: [0x57,0x2f,0xd8,0x59]
vmiota.m v30, v29, v0.t

# Encoding: |010110|1|00000|10001|010|11111|1010111|
# CHECK: vid.v v31
# CHECK-SAME: [0xd7,0xaf,0x08,0x5a]
vid.v v31
# Encoding: |010110|0|00000|10001|010|00000|1010111|
# CHECK: vid.v v0, v0.t
# CHECK-SAME: [0x57,0xa0,0x08,0x58]
vid.v v0, v0.t

# Encoding: |000|100|1|00000|11000|000|00001|0000111|
# CHECK: vlb.v v1, (s8)
# CHECK-SAME: [0x87,0x00,0x0c,0x12]
vlb.v v1, (s8)
# Encoding: |000|100|0|00000|11010|000|00010|0000111|
# CHECK: vlb.v v2, (s10), v0.t
# CHECK-SAME: [0x07,0x01,0x0d,0x10]
vlb.v v2, (s10), v0.t

# Encoding: |000|100|1|00000|11100|101|00011|0000111|
# CHECK: vlh.v v3, (t3)
# CHECK-SAME: [0x87,0x51,0x0e,0x12]
vlh.v v3, (t3)
# Encoding: |000|100|0|00000|11110|101|00100|0000111|
# CHECK: vlh.v v4, (t5), v0.t
# CHECK-SAME: [0x07,0x52,0x0f,0x10]
vlh.v v4, (t5), v0.t

# Encoding: |000|100|1|00000|00001|110|00101|0000111|
# CHECK: vlw.v v5, (ra)
# CHECK-SAME: [0x87,0xe2,0x00,0x12]
vlw.v v5, (ra)
# Encoding: |000|100|0|00000|00011|110|00110|0000111|
# CHECK: vlw.v v6, (gp), v0.t
# CHECK-SAME: [0x07,0xe3,0x01,0x10]
vlw.v v6, (gp), v0.t

# Encoding: |000|000|1|00000|00101|000|00111|0000111|
# CHECK: vlbu.v v7, (t0)
# CHECK-SAME: [0x87,0x83,0x02,0x02]
vlbu.v v7, (t0)
# Encoding: |000|000|0|00000|00111|000|01000|0000111|
# CHECK: vlbu.v v8, (t2), v0.t
# CHECK-SAME: [0x07,0x84,0x03,0x00]
vlbu.v v8, (t2), v0.t

# Encoding: |000|000|1|00000|01001|101|01001|0000111|
# CHECK: vlhu.v v9, (s1)
# CHECK-SAME: [0x87,0xd4,0x04,0x02]
vlhu.v v9, (s1)
# Encoding: |000|000|0|00000|01011|101|01010|0000111|
# CHECK: vlhu.v v10, (a1), v0.t
# CHECK-SAME: [0x07,0xd5,0x05,0x00]
vlhu.v v10, (a1), v0.t

# Encoding: |000|000|1|00000|01101|110|01011|0000111|
# CHECK: vlwu.v v11, (a3)
# CHECK-SAME: [0x87,0xe5,0x06,0x02]
vlwu.v v11, (a3)
# Encoding: |000|000|0|00000|01111|110|01100|0000111|
# CHECK: vlwu.v v12, (a5), v0.t
# CHECK-SAME: [0x07,0xe6,0x07,0x00]
vlwu.v v12, (a5), v0.t

# Encoding: |000|000|1|00000|10001|111|01101|0000111|
# CHECK: vle.v v13, (a7)
# CHECK-SAME: [0x87,0xf6,0x08,0x02]
vle.v v13, (a7)
# Encoding: |000|000|0|00000|10011|111|01110|0000111|
# CHECK: vle.v v14, (s3), v0.t
# CHECK-SAME: [0x07,0xf7,0x09,0x00]
vle.v v14, (s3), v0.t

# Encoding: |000|000|1|00000|10101|000|01111|0100111|
# CHECK: vsb.v v15, (s5)
# CHECK-SAME: [0xa7,0x87,0x0a,0x02]
vsb.v v15, (s5)
# Encoding: |000|000|0|00000|10111|000|10000|0100111|
# CHECK: vsb.v v16, (s7), v0.t
# CHECK-SAME: [0x27,0x88,0x0b,0x00]
vsb.v v16, (s7), v0.t

# Encoding: |000|000|1|00000|11001|101|10001|0100111|
# CHECK: vsh.v v17, (s9)
# CHECK-SAME: [0xa7,0xd8,0x0c,0x02]
vsh.v v17, (s9)
# Encoding: |000|000|0|00000|11011|101|10010|0100111|
# CHECK: vsh.v v18, (s11), v0.t
# CHECK-SAME: [0x27,0xd9,0x0d,0x00]
vsh.v v18, (s11), v0.t

# Encoding: |000|000|1|00000|11101|110|10011|0100111|
# CHECK: vsw.v v19, (t4)
# CHECK-SAME: [0xa7,0xe9,0x0e,0x02]
vsw.v v19, (t4)
# Encoding: |000|000|0|00000|11111|110|10100|0100111|
# CHECK: vsw.v v20, (t6), v0.t
# CHECK-SAME: [0x27,0xea,0x0f,0x00]
vsw.v v20, (t6), v0.t

# Encoding: |000|000|1|00000|00010|111|10101|0100111|
# CHECK: vse.v v21, (sp)
# CHECK-SAME: [0xa7,0x7a,0x01,0x02]
vse.v v21, (sp)
# Encoding: |000|000|0|00000|00100|111|10110|0100111|
# CHECK: vse.v v22, (tp), v0.t
# CHECK-SAME: [0x27,0x7b,0x02,0x00]
vse.v v22, (tp), v0.t

# Encoding: |000|110|1|00110|01000|000|10111|0000111|
# CHECK: vlsb.v v23, (s0), t1
# CHECK-SAME: [0x87,0x0b,0x64,0x1a]
vlsb.v v23, (s0), t1
# Encoding: |000|110|0|01010|01100|000|11000|0000111|
# CHECK: vlsb.v v24, (a2), a0, v0.t
# CHECK-SAME: [0x07,0x0c,0xa6,0x18]
vlsb.v v24, (a2), a0, v0.t

# Encoding: |000|110|1|01110|10000|101|11001|0000111|
# CHECK: vlsh.v v25, (a6), a4
# CHECK-SAME: [0x87,0x5c,0xe8,0x1a]
vlsh.v v25, (a6), a4
# Encoding: |000|110|0|10010|10100|101|11010|0000111|
# CHECK: vlsh.v v26, (s4), s2, v0.t
# CHECK-SAME: [0x07,0x5d,0x2a,0x19]
vlsh.v v26, (s4), s2, v0.t

# Encoding: |000|110|1|10110|11000|110|11011|0000111|
# CHECK: vlsw.v v27, (s8), s6
# CHECK-SAME: [0x87,0x6d,0x6c,0x1b]
vlsw.v v27, (s8), s6
# Encoding: |000|110|0|11010|11100|110|11100|0000111|
# CHECK: vlsw.v v28, (t3), s10, v0.t
# CHECK-SAME: [0x07,0x6e,0xae,0x19]
vlsw.v v28, (t3), s10, v0.t

# Encoding: |000|010|1|11110|00001|000|11101|0000111|
# CHECK: vlsbu.v v29, (ra), t5
# CHECK-SAME: [0x87,0x8e,0xe0,0x0b]
vlsbu.v v29, (ra), t5
# Encoding: |000|010|0|00011|00101|000|11110|0000111|
# CHECK: vlsbu.v v30, (t0), gp, v0.t
# CHECK-SAME: [0x07,0x8f,0x32,0x08]
vlsbu.v v30, (t0), gp, v0.t

# Encoding: |000|010|1|00111|01001|101|11111|0000111|
# CHECK: vlshu.v v31, (s1), t2
# CHECK-SAME: [0x87,0xdf,0x74,0x0a]
vlshu.v v31, (s1), t2
# Encoding: |000|010|0|01011|01101|101|00000|0000111|
# CHECK: vlshu.v v0, (a3), a1, v0.t
# CHECK-SAME: [0x07,0xd0,0xb6,0x08]
vlshu.v v0, (a3), a1, v0.t

# Encoding: |000|010|1|01111|10001|110|00001|0000111|
# CHECK: vlswu.v v1, (a7), a5
# CHECK-SAME: [0x87,0xe0,0xf8,0x0a]
vlswu.v v1, (a7), a5
# Encoding: |000|010|0|10011|10101|110|00010|0000111|
# CHECK: vlswu.v v2, (s5), s3, v0.t
# CHECK-SAME: [0x07,0xe1,0x3a,0x09]
vlswu.v v2, (s5), s3, v0.t

# Encoding: |000|010|1|10111|11001|111|00011|0000111|
# CHECK: vlse.v v3, (s9), s7
# CHECK-SAME: [0x87,0xf1,0x7c,0x0b]
vlse.v v3, (s9), s7
# Encoding: |000|010|0|11011|11101|111|00100|0000111|
# CHECK: vlse.v v4, (t4), s11, v0.t
# CHECK-SAME: [0x07,0xf2,0xbe,0x09]
vlse.v v4, (t4), s11, v0.t

# Encoding: |000|010|1|11111|00010|000|00101|0100111|
# CHECK: vssb.v v5, (sp), t6
# CHECK-SAME: [0xa7,0x02,0xf1,0x0b]
vssb.v v5, (sp), t6
# Encoding: |000|010|0|00100|00110|000|00110|0100111|
# CHECK: vssb.v v6, (t1), tp, v0.t
# CHECK-SAME: [0x27,0x03,0x43,0x08]
vssb.v v6, (t1), tp, v0.t

# Encoding: |000|010|1|01000|01010|101|00111|0100111|
# CHECK: vssh.v v7, (a0), s0
# CHECK-SAME: [0xa7,0x53,0x85,0x0a]
vssh.v v7, (a0), s0
# Encoding: |000|010|0|01100|01110|101|01000|0100111|
# CHECK: vssh.v v8, (a4), a2, v0.t
# CHECK-SAME: [0x27,0x54,0xc7,0x08]
vssh.v v8, (a4), a2, v0.t

# Encoding: |000|010|1|10000|10010|110|01001|0100111|
# CHECK: vssw.v v9, (s2), a6
# CHECK-SAME: [0xa7,0x64,0x09,0x0b]
vssw.v v9, (s2), a6
# Encoding: |000|010|0|10100|10110|110|01010|0100111|
# CHECK: vssw.v v10, (s6), s4, v0.t
# CHECK-SAME: [0x27,0x65,0x4b,0x09]
vssw.v v10, (s6), s4, v0.t

# Encoding: |000|010|1|11000|11010|111|01011|0100111|
# CHECK: vsse.v v11, (s10), s8
# CHECK-SAME: [0xa7,0x75,0x8d,0x0b]
vsse.v v11, (s10), s8
# Encoding: |000|010|0|11100|11110|111|01100|0100111|
# CHECK: vsse.v v12, (t5), t3, v0.t
# CHECK-SAME: [0x27,0x76,0xcf,0x09]
vsse.v v12, (t5), t3, v0.t

# Encoding: |000|111|1|01101|00001|000|01110|0000111|
# CHECK: vlxb.v v14, (ra), v13
# CHECK-SAME: [0x07,0x87,0xd0,0x1e]
vlxb.v v14, (ra), v13
# Encoding: |000|111|0|01111|00011|000|10000|0000111|
# CHECK: vlxb.v v16, (gp), v15, v0.t
# CHECK-SAME: [0x07,0x88,0xf1,0x1c]
vlxb.v v16, (gp), v15, v0.t

# Encoding: |000|111|1|10001|00101|101|10010|0000111|
# CHECK: vlxh.v v18, (t0), v17
# CHECK-SAME: [0x07,0xd9,0x12,0x1f]
vlxh.v v18, (t0), v17
# Encoding: |000|111|0|10011|00111|101|10100|0000111|
# CHECK: vlxh.v v20, (t2), v19, v0.t
# CHECK-SAME: [0x07,0xda,0x33,0x1d]
vlxh.v v20, (t2), v19, v0.t

# Encoding: |000|111|1|10101|01001|110|10110|0000111|
# CHECK: vlxw.v v22, (s1), v21
# CHECK-SAME: [0x07,0xeb,0x54,0x1f]
vlxw.v v22, (s1), v21
# Encoding: |000|111|0|10111|01011|110|11000|0000111|
# CHECK: vlxw.v v24, (a1), v23, v0.t
# CHECK-SAME: [0x07,0xec,0x75,0x1d]
vlxw.v v24, (a1), v23, v0.t

# Encoding: |000|011|1|11001|01101|000|11010|0000111|
# CHECK: vlxbu.v v26, (a3), v25
# CHECK-SAME: [0x07,0x8d,0x96,0x0f]
vlxbu.v v26, (a3), v25
# Encoding: |000|011|0|11011|01111|000|11100|0000111|
# CHECK: vlxbu.v v28, (a5), v27, v0.t
# CHECK-SAME: [0x07,0x8e,0xb7,0x0d]
vlxbu.v v28, (a5), v27, v0.t

# Encoding: |000|011|1|11101|10001|101|11110|0000111|
# CHECK: vlxhu.v v30, (a7), v29
# CHECK-SAME: [0x07,0xdf,0xd8,0x0f]
vlxhu.v v30, (a7), v29
# Encoding: |000|011|0|11111|10011|101|00000|0000111|
# CHECK: vlxhu.v v0, (s3), v31, v0.t
# CHECK-SAME: [0x07,0xd0,0xf9,0x0d]
vlxhu.v v0, (s3), v31, v0.t

# Encoding: |000|011|1|00001|10101|110|00010|0000111|
# CHECK: vlxwu.v v2, (s5), v1
# CHECK-SAME: [0x07,0xe1,0x1a,0x0e]
vlxwu.v v2, (s5), v1
# Encoding: |000|011|0|00011|10111|110|00100|0000111|
# CHECK: vlxwu.v v4, (s7), v3, v0.t
# CHECK-SAME: [0x07,0xe2,0x3b,0x0c]
vlxwu.v v4, (s7), v3, v0.t

# Encoding: |000|011|1|00101|11001|111|00110|0000111|
# CHECK: vlxe.v v6, (s9), v5
# CHECK-SAME: [0x07,0xf3,0x5c,0x0e]
vlxe.v v6, (s9), v5
# Encoding: |000|011|0|00111|11011|111|01000|0000111|
# CHECK: vlxe.v v8, (s11), v7, v0.t
# CHECK-SAME: [0x07,0xf4,0x7d,0x0c]
vlxe.v v8, (s11), v7, v0.t

# Encoding: |000|011|1|01001|11101|000|01010|0100111|
# CHECK: vsxb.v v10, (t4), v9
# CHECK-SAME: [0x27,0x85,0x9e,0x0e]
vsxb.v v10, (t4), v9
# Encoding: |000|011|0|01011|11111|000|01100|0100111|
# CHECK: vsxb.v v12, (t6), v11, v0.t
# CHECK-SAME: [0x27,0x86,0xbf,0x0c]
vsxb.v v12, (t6), v11, v0.t

# Encoding: |000|011|1|01101|00010|101|01110|0100111|
# CHECK: vsxh.v v14, (sp), v13
# CHECK-SAME: [0x27,0x57,0xd1,0x0e]
vsxh.v v14, (sp), v13
# Encoding: |000|011|0|01111|00100|101|10000|0100111|
# CHECK: vsxh.v v16, (tp), v15, v0.t
# CHECK-SAME: [0x27,0x58,0xf2,0x0c]
vsxh.v v16, (tp), v15, v0.t

# Encoding: |000|011|1|10001|00110|110|10010|0100111|
# CHECK: vsxw.v v18, (t1), v17
# CHECK-SAME: [0x27,0x69,0x13,0x0f]
vsxw.v v18, (t1), v17
# Encoding: |000|011|0|10011|01000|110|10100|0100111|
# CHECK: vsxw.v v20, (s0), v19, v0.t
# CHECK-SAME: [0x27,0x6a,0x34,0x0d]
vsxw.v v20, (s0), v19, v0.t

# Encoding: |000|011|1|10101|01010|111|10110|0100111|
# CHECK: vsxe.v v22, (a0), v21
# CHECK-SAME: [0x27,0x7b,0x55,0x0f]
vsxe.v v22, (a0), v21
# Encoding: |000|011|0|10111|01100|111|11000|0100111|
# CHECK: vsxe.v v24, (a2), v23, v0.t
# CHECK-SAME: [0x27,0x7c,0x76,0x0d]
vsxe.v v24, (a2), v23, v0.t

# Encoding: |000|111|1|11001|01110|000|11010|0100111|
# CHECK: vsuxb.v v26, (a4), v25
# CHECK-SAME: [0x27,0x0d,0x97,0x1f]
vsuxb.v v26, (a4), v25
# Encoding: |000|111|0|11011|10000|000|11100|0100111|
# CHECK: vsuxb.v v28, (a6), v27, v0.t
# CHECK-SAME: [0x27,0x0e,0xb8,0x1d]
vsuxb.v v28, (a6), v27, v0.t

# Encoding: |000|111|1|11101|10010|101|11110|0100111|
# CHECK: vsuxh.v v30, (s2), v29
# CHECK-SAME: [0x27,0x5f,0xd9,0x1f]
vsuxh.v v30, (s2), v29
# Encoding: |000|111|0|11111|10100|101|00000|0100111|
# CHECK: vsuxh.v v0, (s4), v31, v0.t
# CHECK-SAME: [0x27,0x50,0xfa,0x1d]
vsuxh.v v0, (s4), v31, v0.t

# Encoding: |000|111|1|00001|10110|110|00010|0100111|
# CHECK: vsuxw.v v2, (s6), v1
# CHECK-SAME: [0x27,0x61,0x1b,0x1e]
vsuxw.v v2, (s6), v1
# Encoding: |000|111|0|00011|11000|110|00100|0100111|
# CHECK: vsuxw.v v4, (s8), v3, v0.t
# CHECK-SAME: [0x27,0x62,0x3c,0x1c]
vsuxw.v v4, (s8), v3, v0.t

# Encoding: |000|111|1|00101|11010|111|00110|0100111|
# CHECK: vsuxe.v v6, (s10), v5
# CHECK-SAME: [0x27,0x73,0x5d,0x1e]
vsuxe.v v6, (s10), v5
# Encoding: |000|111|0|00111|11100|111|01000|0100111|
# CHECK: vsuxe.v v8, (t3), v7, v0.t
# CHECK-SAME: [0x27,0x74,0x7e,0x1c]
vsuxe.v v8, (t3), v7, v0.t

