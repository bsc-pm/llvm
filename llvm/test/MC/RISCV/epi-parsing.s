# Generated with utils/EPI/process.py
# RUN: llvm-mc < %s -arch=riscv64 -mattr=+m,+f,+d,+a,+epi | FileCheck %s

# CHECK: vadd.vv v2, v0, v1
vadd.vv v2, v0, v1
# CHECK: vadd.vv v5, v3, v4, v0.t
vadd.vv v5, v3, v4, v0.t

# CHECK: vadd.vx v7, v6, ra
vadd.vx v7, v6, ra
# CHECK: vadd.vx v9, v8, gp, v0.t
vadd.vx v9, v8, gp, v0.t

# CHECK: vadd.vi v11, v10, 0
vadd.vi v11, v10, 0
# CHECK: vadd.vi v13, v12, 1, v0.t
vadd.vi v13, v12, 1, v0.t

# CHECK: vsub.vv v16, v14, v15
vsub.vv v16, v14, v15
# CHECK: vsub.vv v19, v17, v18, v0.t
vsub.vv v19, v17, v18, v0.t

# CHECK: vsub.vx v21, v20, t0
vsub.vx v21, v20, t0
# CHECK: vsub.vx v23, v22, t2, v0.t
vsub.vx v23, v22, t2, v0.t

# CHECK: vrsub.vx v25, v24, s1
vrsub.vx v25, v24, s1
# CHECK: vrsub.vx v27, v26, a1, v0.t
vrsub.vx v27, v26, a1, v0.t

# CHECK: vrsub.vi v29, v28, 2
vrsub.vi v29, v28, 2
# CHECK: vrsub.vi v31, v30, 3, v0.t
vrsub.vi v31, v30, 3, v0.t

# CHECK: vminu.vv v2, v0, v1
vminu.vv v2, v0, v1
# CHECK: vminu.vv v5, v3, v4, v0.t
vminu.vv v5, v3, v4, v0.t

# CHECK: vminu.vx v7, v6, a3
vminu.vx v7, v6, a3
# CHECK: vminu.vx v9, v8, a5, v0.t
vminu.vx v9, v8, a5, v0.t

# CHECK: vmin.vv v12, v10, v11
vmin.vv v12, v10, v11
# CHECK: vmin.vv v15, v13, v14, v0.t
vmin.vv v15, v13, v14, v0.t

# CHECK: vmin.vx v17, v16, a7
vmin.vx v17, v16, a7
# CHECK: vmin.vx v19, v18, s3, v0.t
vmin.vx v19, v18, s3, v0.t

# CHECK: vmaxu.vv v22, v20, v21
vmaxu.vv v22, v20, v21
# CHECK: vmaxu.vv v25, v23, v24, v0.t
vmaxu.vv v25, v23, v24, v0.t

# CHECK: vmaxu.vx v27, v26, s5
vmaxu.vx v27, v26, s5
# CHECK: vmaxu.vx v29, v28, s7, v0.t
vmaxu.vx v29, v28, s7, v0.t

# CHECK: vmax.vv v0, v30, v31
vmax.vv v0, v30, v31
# CHECK: vmax.vv v3, v1, v2, v0.t
vmax.vv v3, v1, v2, v0.t

# CHECK: vmax.vx v5, v4, s9
vmax.vx v5, v4, s9
# CHECK: vmax.vx v7, v6, s11, v0.t
vmax.vx v7, v6, s11, v0.t

# CHECK: vand.vv v10, v8, v9
vand.vv v10, v8, v9
# CHECK: vand.vv v13, v11, v12, v0.t
vand.vv v13, v11, v12, v0.t

# CHECK: vand.vx v15, v14, t4
vand.vx v15, v14, t4
# CHECK: vand.vx v17, v16, t6, v0.t
vand.vx v17, v16, t6, v0.t

# CHECK: vand.vi v19, v18, 4
vand.vi v19, v18, 4
# CHECK: vand.vi v21, v20, 5, v0.t
vand.vi v21, v20, 5, v0.t

# CHECK: vor.vv v24, v22, v23
vor.vv v24, v22, v23
# CHECK: vor.vv v27, v25, v26, v0.t
vor.vv v27, v25, v26, v0.t

# CHECK: vor.vx v29, v28, sp
vor.vx v29, v28, sp
# CHECK: vor.vx v31, v30, tp, v0.t
vor.vx v31, v30, tp, v0.t

# CHECK: vor.vi v1, v0, 6
vor.vi v1, v0, 6
# CHECK: vor.vi v3, v2, 7, v0.t
vor.vi v3, v2, 7, v0.t

# CHECK: vxor.vv v6, v4, v5
vxor.vv v6, v4, v5
# CHECK: vxor.vv v9, v7, v8, v0.t
vxor.vv v9, v7, v8, v0.t

# CHECK: vxor.vx v11, v10, t1
vxor.vx v11, v10, t1
# CHECK: vxor.vx v13, v12, s0, v0.t
vxor.vx v13, v12, s0, v0.t

# CHECK: vxor.vi v15, v14, 8
vxor.vi v15, v14, 8
# CHECK: vxor.vi v17, v16, 9, v0.t
vxor.vi v17, v16, 9, v0.t

# CHECK: vrgather.vv v20, v18, v19
vrgather.vv v20, v18, v19
# CHECK: vrgather.vv v23, v21, v22, v0.t
vrgather.vv v23, v21, v22, v0.t

# CHECK: vrgather.vx v25, v24, a0
vrgather.vx v25, v24, a0
# CHECK: vrgather.vx v27, v26, a2, v0.t
vrgather.vx v27, v26, a2, v0.t

# CHECK: vrgather.vi v29, v28, 10
vrgather.vi v29, v28, 10
# CHECK: vrgather.vi v31, v30, 11, v0.t
vrgather.vi v31, v30, 11, v0.t

# CHECK: vslideup.vx v1, v0, a4
vslideup.vx v1, v0, a4
# CHECK: vslideup.vx v3, v2, a6, v0.t
vslideup.vx v3, v2, a6, v0.t

# CHECK: vslideup.vi v5, v4, 12
vslideup.vi v5, v4, 12
# CHECK: vslideup.vi v7, v6, 13, v0.t
vslideup.vi v7, v6, 13, v0.t

# CHECK: vslidedown.vx v9, v8, s2
vslidedown.vx v9, v8, s2
# CHECK: vslidedown.vx v11, v10, s4, v0.t
vslidedown.vx v11, v10, s4, v0.t

# CHECK: vslidedown.vi v13, v12, 14
vslidedown.vi v13, v12, 14
# CHECK: vslidedown.vi v15, v14, 15, v0.t
vslidedown.vi v15, v14, 15, v0.t

# CHECK: vadc.vv v18, v16, v17
vadc.vv v18, v16, v17

# CHECK: vadc.vx v20, v19, s6
vadc.vx v20, v19, s6

# CHECK: vadc.vi v22, v21, -16
vadc.vi v22, v21, -16

# CHECK: vsbc.vv v25, v23, v24
vsbc.vv v25, v23, v24

# CHECK: vsbc.vx v27, v26, s8
vsbc.vx v27, v26, s8

# CHECK: vmerge.vv v30, v28, v29
vmerge.vv v30, v28, v29
# CHECK: vmerge.vv v1, v31, v0, v0.t
vmerge.vv v1, v31, v0, v0.t

# CHECK: vmerge.vx v3, v2, s10
vmerge.vx v3, v2, s10
# CHECK: vmerge.vx v5, v4, t3, v0.t
vmerge.vx v5, v4, t3, v0.t

# CHECK: vmerge.vi v7, v6, -15
vmerge.vi v7, v6, -15
# CHECK: vmerge.vi v9, v8, -14, v0.t
vmerge.vi v9, v8, -14, v0.t

# CHECK: vseq.vv v12, v10, v11
vseq.vv v12, v10, v11
# CHECK: vseq.vv v15, v13, v14, v0.t
vseq.vv v15, v13, v14, v0.t

# CHECK: vseq.vx v17, v16, t5
vseq.vx v17, v16, t5
# CHECK: vseq.vx v19, v18, ra, v0.t
vseq.vx v19, v18, ra, v0.t

# CHECK: vseq.vi v21, v20, -13
vseq.vi v21, v20, -13
# CHECK: vseq.vi v23, v22, -12, v0.t
vseq.vi v23, v22, -12, v0.t

# CHECK: vsne.vv v26, v24, v25
vsne.vv v26, v24, v25
# CHECK: vsne.vv v29, v27, v28, v0.t
vsne.vv v29, v27, v28, v0.t

# CHECK: vsne.vx v31, v30, gp
vsne.vx v31, v30, gp
# CHECK: vsne.vx v1, v0, t0, v0.t
vsne.vx v1, v0, t0, v0.t

# CHECK: vsne.vi v3, v2, -11
vsne.vi v3, v2, -11
# CHECK: vsne.vi v5, v4, -10, v0.t
vsne.vi v5, v4, -10, v0.t

# CHECK: vsltu.vv v8, v6, v7
vsltu.vv v8, v6, v7
# CHECK: vsltu.vv v11, v9, v10, v0.t
vsltu.vv v11, v9, v10, v0.t

# CHECK: vsltu.vx v13, v12, t2
vsltu.vx v13, v12, t2
# CHECK: vsltu.vx v15, v14, s1, v0.t
vsltu.vx v15, v14, s1, v0.t

# CHECK: vslt.vv v18, v16, v17
vslt.vv v18, v16, v17
# CHECK: vslt.vv v21, v19, v20, v0.t
vslt.vv v21, v19, v20, v0.t

# CHECK: vslt.vx v23, v22, a1
vslt.vx v23, v22, a1
# CHECK: vslt.vx v25, v24, a3, v0.t
vslt.vx v25, v24, a3, v0.t

# CHECK: vsleu.vv v28, v26, v27
vsleu.vv v28, v26, v27
# CHECK: vsleu.vv v31, v29, v30, v0.t
vsleu.vv v31, v29, v30, v0.t

# CHECK: vsleu.vx v1, v0, a5
vsleu.vx v1, v0, a5
# CHECK: vsleu.vx v3, v2, a7, v0.t
vsleu.vx v3, v2, a7, v0.t

# CHECK: vsleu.vi v5, v4, -9
vsleu.vi v5, v4, -9
# CHECK: vsleu.vi v7, v6, -8, v0.t
vsleu.vi v7, v6, -8, v0.t

# CHECK: vsle.vv v10, v8, v9
vsle.vv v10, v8, v9
# CHECK: vsle.vv v13, v11, v12, v0.t
vsle.vv v13, v11, v12, v0.t

# CHECK: vsle.vx v15, v14, s3
vsle.vx v15, v14, s3
# CHECK: vsle.vx v17, v16, s5, v0.t
vsle.vx v17, v16, s5, v0.t

# CHECK: vsle.vi v19, v18, -7
vsle.vi v19, v18, -7
# CHECK: vsle.vi v21, v20, -6, v0.t
vsle.vi v21, v20, -6, v0.t

# CHECK: vsgtu.vx v23, v22, s7
vsgtu.vx v23, v22, s7
# CHECK: vsgtu.vx v25, v24, s9, v0.t
vsgtu.vx v25, v24, s9, v0.t

# CHECK: vsgtu.vi v27, v26, -5
vsgtu.vi v27, v26, -5
# CHECK: vsgtu.vi v29, v28, -4, v0.t
vsgtu.vi v29, v28, -4, v0.t

# CHECK: vsgt.vx v31, v30, s11
vsgt.vx v31, v30, s11
# CHECK: vsgt.vx v1, v0, t4, v0.t
vsgt.vx v1, v0, t4, v0.t

# CHECK: vsgt.vi v3, v2, -3
vsgt.vi v3, v2, -3
# CHECK: vsgt.vi v5, v4, -2, v0.t
vsgt.vi v5, v4, -2, v0.t

# CHECK: vsaddu.vv v8, v6, v7
vsaddu.vv v8, v6, v7
# CHECK: vsaddu.vv v11, v9, v10, v0.t
vsaddu.vv v11, v9, v10, v0.t

# CHECK: vsaddu.vx v13, v12, t6
vsaddu.vx v13, v12, t6
# CHECK: vsaddu.vx v15, v14, sp, v0.t
vsaddu.vx v15, v14, sp, v0.t

# CHECK: vsaddu.vi v17, v16, -1
vsaddu.vi v17, v16, -1
# CHECK: vsaddu.vi v19, v18, 0, v0.t
vsaddu.vi v19, v18, 0, v0.t

# CHECK: vsadd.vv v22, v20, v21
vsadd.vv v22, v20, v21
# CHECK: vsadd.vv v25, v23, v24, v0.t
vsadd.vv v25, v23, v24, v0.t

# CHECK: vsadd.vx v27, v26, tp
vsadd.vx v27, v26, tp
# CHECK: vsadd.vx v29, v28, t1, v0.t
vsadd.vx v29, v28, t1, v0.t

# CHECK: vsadd.vi v31, v30, 1
vsadd.vi v31, v30, 1
# CHECK: vsadd.vi v1, v0, 2, v0.t
vsadd.vi v1, v0, 2, v0.t

# CHECK: vssubu.vv v4, v2, v3
vssubu.vv v4, v2, v3
# CHECK: vssubu.vv v7, v5, v6, v0.t
vssubu.vv v7, v5, v6, v0.t

# CHECK: vssubu.vx v9, v8, s0
vssubu.vx v9, v8, s0
# CHECK: vssubu.vx v11, v10, a0, v0.t
vssubu.vx v11, v10, a0, v0.t

# CHECK: vssub.vv v14, v12, v13
vssub.vv v14, v12, v13
# CHECK: vssub.vv v17, v15, v16, v0.t
vssub.vv v17, v15, v16, v0.t

# CHECK: vssub.vx v19, v18, a2
vssub.vx v19, v18, a2
# CHECK: vssub.vx v21, v20, a4, v0.t
vssub.vx v21, v20, a4, v0.t

# CHECK: vaadd.vv v24, v22, v23
vaadd.vv v24, v22, v23
# CHECK: vaadd.vv v27, v25, v26, v0.t
vaadd.vv v27, v25, v26, v0.t

# CHECK: vaadd.vx v29, v28, a6
vaadd.vx v29, v28, a6
# CHECK: vaadd.vx v31, v30, s2, v0.t
vaadd.vx v31, v30, s2, v0.t

# CHECK: vaadd.vi v1, v0, 3
vaadd.vi v1, v0, 3
# CHECK: vaadd.vi v3, v2, 4, v0.t
vaadd.vi v3, v2, 4, v0.t

# CHECK: vsll.vv v6, v4, v5
vsll.vv v6, v4, v5
# CHECK: vsll.vv v9, v7, v8, v0.t
vsll.vv v9, v7, v8, v0.t

# CHECK: vsll.vx v11, v10, s4
vsll.vx v11, v10, s4
# CHECK: vsll.vx v13, v12, s6, v0.t
vsll.vx v13, v12, s6, v0.t

# CHECK: vsll.vi v15, v14, 5
vsll.vi v15, v14, 5
# CHECK: vsll.vi v17, v16, 6, v0.t
vsll.vi v17, v16, 6, v0.t

# CHECK: vasub.vv v20, v18, v19
vasub.vv v20, v18, v19
# CHECK: vasub.vv v23, v21, v22, v0.t
vasub.vv v23, v21, v22, v0.t

# CHECK: vasub.vx v25, v24, s8
vasub.vx v25, v24, s8
# CHECK: vasub.vx v27, v26, s10, v0.t
vasub.vx v27, v26, s10, v0.t

# CHECK: vsmul.vv v30, v28, v29
vsmul.vv v30, v28, v29
# CHECK: vsmul.vv v1, v31, v0, v0.t
vsmul.vv v1, v31, v0, v0.t

# CHECK: vsmul.vx v3, v2, t3
vsmul.vx v3, v2, t3
# CHECK: vsmul.vx v5, v4, t5, v0.t
vsmul.vx v5, v4, t5, v0.t

# CHECK: vsrl.vv v8, v6, v7
vsrl.vv v8, v6, v7
# CHECK: vsrl.vv v11, v9, v10, v0.t
vsrl.vv v11, v9, v10, v0.t

# CHECK: vsrl.vx v13, v12, ra
vsrl.vx v13, v12, ra
# CHECK: vsrl.vx v15, v14, gp, v0.t
vsrl.vx v15, v14, gp, v0.t

# CHECK: vsrl.vi v17, v16, 7
vsrl.vi v17, v16, 7
# CHECK: vsrl.vi v19, v18, 8, v0.t
vsrl.vi v19, v18, 8, v0.t

# CHECK: vsra.vv v22, v20, v21
vsra.vv v22, v20, v21
# CHECK: vsra.vv v25, v23, v24, v0.t
vsra.vv v25, v23, v24, v0.t

# CHECK: vsra.vx v27, v26, t0
vsra.vx v27, v26, t0
# CHECK: vsra.vx v29, v28, t2, v0.t
vsra.vx v29, v28, t2, v0.t

# CHECK: vsra.vi v31, v30, 9
vsra.vi v31, v30, 9
# CHECK: vsra.vi v1, v0, 10, v0.t
vsra.vi v1, v0, 10, v0.t

# CHECK: vssrl.vv v4, v2, v3
vssrl.vv v4, v2, v3
# CHECK: vssrl.vv v7, v5, v6, v0.t
vssrl.vv v7, v5, v6, v0.t

# CHECK: vssrl.vx v9, v8, s1
vssrl.vx v9, v8, s1
# CHECK: vssrl.vx v11, v10, a1, v0.t
vssrl.vx v11, v10, a1, v0.t

# CHECK: vssrl.vi v13, v12, 11
vssrl.vi v13, v12, 11
# CHECK: vssrl.vi v15, v14, 12, v0.t
vssrl.vi v15, v14, 12, v0.t

# CHECK: vssra.vv v18, v16, v17
vssra.vv v18, v16, v17
# CHECK: vssra.vv v21, v19, v20, v0.t
vssra.vv v21, v19, v20, v0.t

# CHECK: vssra.vx v23, v22, a3
vssra.vx v23, v22, a3
# CHECK: vssra.vx v25, v24, a5, v0.t
vssra.vx v25, v24, a5, v0.t

# CHECK: vssra.vi v27, v26, 13
vssra.vi v27, v26, 13
# CHECK: vssra.vi v29, v28, 14, v0.t
vssra.vi v29, v28, 14, v0.t

# CHECK: vnsrl.vv v0, v30, v31
vnsrl.vv v0, v30, v31
# CHECK: vnsrl.vv v3, v1, v2, v0.t
vnsrl.vv v3, v1, v2, v0.t

# CHECK: vnsrl.vx v5, v4, a7
vnsrl.vx v5, v4, a7
# CHECK: vnsrl.vx v7, v6, s3, v0.t
vnsrl.vx v7, v6, s3, v0.t

# CHECK: vnsrl.vi v9, v8, 15
vnsrl.vi v9, v8, 15
# CHECK: vnsrl.vi v11, v10, -16, v0.t
vnsrl.vi v11, v10, -16, v0.t

# CHECK: vnsra.vv v14, v12, v13
vnsra.vv v14, v12, v13
# CHECK: vnsra.vv v17, v15, v16, v0.t
vnsra.vv v17, v15, v16, v0.t

# CHECK: vnsra.vx v19, v18, s5
vnsra.vx v19, v18, s5
# CHECK: vnsra.vx v21, v20, s7, v0.t
vnsra.vx v21, v20, s7, v0.t

# CHECK: vnsra.vi v23, v22, -15
vnsra.vi v23, v22, -15
# CHECK: vnsra.vi v25, v24, -14, v0.t
vnsra.vi v25, v24, -14, v0.t

# CHECK: vnclipu.vv v28, v26, v27
vnclipu.vv v28, v26, v27
# CHECK: vnclipu.vv v31, v29, v30, v0.t
vnclipu.vv v31, v29, v30, v0.t

# CHECK: vnclipu.vx v1, v0, s9
vnclipu.vx v1, v0, s9
# CHECK: vnclipu.vx v3, v2, s11, v0.t
vnclipu.vx v3, v2, s11, v0.t

# CHECK: vnclipu.vi v5, v4, -13
vnclipu.vi v5, v4, -13
# CHECK: vnclipu.vi v7, v6, -12, v0.t
vnclipu.vi v7, v6, -12, v0.t

# CHECK: vnclip.vv v10, v8, v9
vnclip.vv v10, v8, v9
# CHECK: vnclip.vv v13, v11, v12, v0.t
vnclip.vv v13, v11, v12, v0.t

# CHECK: vnclip.vx v15, v14, t4
vnclip.vx v15, v14, t4
# CHECK: vnclip.vx v17, v16, t6, v0.t
vnclip.vx v17, v16, t6, v0.t

# CHECK: vnclip.vi v19, v18, -11
vnclip.vi v19, v18, -11
# CHECK: vnclip.vi v21, v20, -10, v0.t
vnclip.vi v21, v20, -10, v0.t

# CHECK: vwredsumu.vs v24, v22, v23
vwredsumu.vs v24, v22, v23
# CHECK: vwredsumu.vs v27, v25, v26, v0.t
vwredsumu.vs v27, v25, v26, v0.t

# CHECK: vwredsum.vs v30, v28, v29
vwredsum.vs v30, v28, v29
# CHECK: vwredsum.vs v1, v31, v0, v0.t
vwredsum.vs v1, v31, v0, v0.t

# CHECK: vdotu.vv v4, v2, v3
vdotu.vv v4, v2, v3
# CHECK: vdotu.vv v7, v5, v6, v0.t
vdotu.vv v7, v5, v6, v0.t

# CHECK: vdot.vv v10, v8, v9
vdot.vv v10, v8, v9
# CHECK: vdot.vv v13, v11, v12, v0.t
vdot.vv v13, v11, v12, v0.t

# CHECK: vwsmaccu.vv v16, v14, v15
vwsmaccu.vv v16, v14, v15
# CHECK: vwsmaccu.vv v19, v17, v18, v0.t
vwsmaccu.vv v19, v17, v18, v0.t

# CHECK: vwsmaccu.vx v21, v20, sp
vwsmaccu.vx v21, v20, sp
# CHECK: vwsmaccu.vx v23, v22, tp, v0.t
vwsmaccu.vx v23, v22, tp, v0.t

# CHECK: vwsmacc.vv v26, v24, v25
vwsmacc.vv v26, v24, v25
# CHECK: vwsmacc.vv v29, v27, v28, v0.t
vwsmacc.vv v29, v27, v28, v0.t

# CHECK: vwsmacc.vx v31, v30, t1
vwsmacc.vx v31, v30, t1
# CHECK: vwsmacc.vx v1, v0, s0, v0.t
vwsmacc.vx v1, v0, s0, v0.t

# CHECK: vwsmsacu.vv v4, v2, v3
vwsmsacu.vv v4, v2, v3
# CHECK: vwsmsacu.vv v7, v5, v6, v0.t
vwsmsacu.vv v7, v5, v6, v0.t

# CHECK: vwsmsacu.vx v9, v8, a0
vwsmsacu.vx v9, v8, a0
# CHECK: vwsmsacu.vx v11, v10, a2, v0.t
vwsmsacu.vx v11, v10, a2, v0.t

# CHECK: vwsmsac.vv v14, v12, v13
vwsmsac.vv v14, v12, v13
# CHECK: vwsmsac.vv v17, v15, v16, v0.t
vwsmsac.vv v17, v15, v16, v0.t

# CHECK: vwsmsac.vx v19, v18, a4
vwsmsac.vx v19, v18, a4
# CHECK: vwsmsac.vx v21, v20, a6, v0.t
vwsmsac.vx v21, v20, a6, v0.t

# CHECK: vredsum.vs v24, v22, v23
vredsum.vs v24, v22, v23
# CHECK: vredsum.vs v27, v25, v26, v0.t
vredsum.vs v27, v25, v26, v0.t

# CHECK: vredand.vs v30, v28, v29
vredand.vs v30, v28, v29
# CHECK: vredand.vs v1, v31, v0, v0.t
vredand.vs v1, v31, v0, v0.t

# CHECK: vredor.vs v4, v2, v3
vredor.vs v4, v2, v3
# CHECK: vredor.vs v7, v5, v6, v0.t
vredor.vs v7, v5, v6, v0.t

# CHECK: vredxor.vs v10, v8, v9
vredxor.vs v10, v8, v9
# CHECK: vredxor.vs v13, v11, v12, v0.t
vredxor.vs v13, v11, v12, v0.t

# CHECK: vredminu.vs v16, v14, v15
vredminu.vs v16, v14, v15
# CHECK: vredminu.vs v19, v17, v18, v0.t
vredminu.vs v19, v17, v18, v0.t

# CHECK: vredmin.vs v22, v20, v21
vredmin.vs v22, v20, v21
# CHECK: vredmin.vs v25, v23, v24, v0.t
vredmin.vs v25, v23, v24, v0.t

# CHECK: vredmaxu.vs v28, v26, v27
vredmaxu.vs v28, v26, v27
# CHECK: vredmaxu.vs v31, v29, v30, v0.t
vredmaxu.vs v31, v29, v30, v0.t

# CHECK: vredmax.vs v2, v0, v1
vredmax.vs v2, v0, v1
# CHECK: vredmax.vs v5, v3, v4, v0.t
vredmax.vs v5, v3, v4, v0.t

# CHECK: vext.x.v s4, v6, s2
vext.x.v s4, v6, s2

# CHECK: vmv.s.x v7, s6
vmv.s.x v7, s6

# CHECK: vslide1up.vx v9, v8, s8
vslide1up.vx v9, v8, s8
# CHECK: vslide1up.vx v11, v10, s10, v0.t
vslide1up.vx v11, v10, s10, v0.t

# CHECK: vslide1down.vx v13, v12, t3
vslide1down.vx v13, v12, t3
# CHECK: vslide1down.vx v15, v14, t5, v0.t
vslide1down.vx v15, v14, t5, v0.t

# CHECK: vmpopc.m ra, v16
vmpopc.m ra, v16
# CHECK: vmpopc.m gp, v17, v0.t
vmpopc.m gp, v17, v0.t

# CHECK: vmfirst.m t0, v18
vmfirst.m t0, v18
# CHECK: vmfirst.m t2, v19, v0.t
vmfirst.m t2, v19, v0.t

# CHECK: vcompress.vm v22, v20, v21
vcompress.vm v22, v20, v21

# CHECK: vmandnot.mm v25, v23, v24
vmandnot.mm v25, v23, v24

# CHECK: vmand.mm v28, v26, v27
vmand.mm v28, v26, v27

# CHECK: vmor.mm v31, v29, v30
vmor.mm v31, v29, v30

# CHECK: vmxor.mm v2, v0, v1
vmxor.mm v2, v0, v1

# CHECK: vmornot.mm v5, v3, v4
vmornot.mm v5, v3, v4

# CHECK: vmnand.mm v8, v6, v7
vmnand.mm v8, v6, v7

# CHECK: vmnor.mm v11, v9, v10
vmnor.mm v11, v9, v10

# CHECK: vmxnor.mm v14, v12, v13
vmxnor.mm v14, v12, v13

# CHECK: vdivu.vv v17, v15, v16
vdivu.vv v17, v15, v16
# CHECK: vdivu.vv v20, v18, v19, v0.t
vdivu.vv v20, v18, v19, v0.t

# CHECK: vdivu.vx v22, v21, s1
vdivu.vx v22, v21, s1
# CHECK: vdivu.vx v24, v23, a1, v0.t
vdivu.vx v24, v23, a1, v0.t

# CHECK: vdiv.vv v27, v25, v26
vdiv.vv v27, v25, v26
# CHECK: vdiv.vv v30, v28, v29, v0.t
vdiv.vv v30, v28, v29, v0.t

# CHECK: vdiv.vx v0, v31, a3
vdiv.vx v0, v31, a3
# CHECK: vdiv.vx v2, v1, a5, v0.t
vdiv.vx v2, v1, a5, v0.t

# CHECK: vremu.vv v5, v3, v4
vremu.vv v5, v3, v4
# CHECK: vremu.vv v8, v6, v7, v0.t
vremu.vv v8, v6, v7, v0.t

# CHECK: vremu.vx v10, v9, a7
vremu.vx v10, v9, a7
# CHECK: vremu.vx v12, v11, s3, v0.t
vremu.vx v12, v11, s3, v0.t

# CHECK: vrem.vv v15, v13, v14
vrem.vv v15, v13, v14
# CHECK: vrem.vv v18, v16, v17, v0.t
vrem.vv v18, v16, v17, v0.t

# CHECK: vrem.vx v20, v19, s5
vrem.vx v20, v19, s5
# CHECK: vrem.vx v22, v21, s7, v0.t
vrem.vx v22, v21, s7, v0.t

# CHECK: vmulhu.vv v25, v23, v24
vmulhu.vv v25, v23, v24
# CHECK: vmulhu.vv v28, v26, v27, v0.t
vmulhu.vv v28, v26, v27, v0.t

# CHECK: vmulhu.vx v30, v29, s9
vmulhu.vx v30, v29, s9
# CHECK: vmulhu.vx v0, v31, s11, v0.t
vmulhu.vx v0, v31, s11, v0.t

# CHECK: vmul.vv v3, v1, v2
vmul.vv v3, v1, v2
# CHECK: vmul.vv v6, v4, v5, v0.t
vmul.vv v6, v4, v5, v0.t

# CHECK: vmul.vx v8, v7, t4
vmul.vx v8, v7, t4
# CHECK: vmul.vx v10, v9, t6, v0.t
vmul.vx v10, v9, t6, v0.t

# CHECK: vmulhsu.vv v13, v11, v12
vmulhsu.vv v13, v11, v12
# CHECK: vmulhsu.vv v16, v14, v15, v0.t
vmulhsu.vv v16, v14, v15, v0.t

# CHECK: vmulhsu.vx v18, v17, sp
vmulhsu.vx v18, v17, sp
# CHECK: vmulhsu.vx v20, v19, tp, v0.t
vmulhsu.vx v20, v19, tp, v0.t

# CHECK: vmulh.vv v23, v21, v22
vmulh.vv v23, v21, v22
# CHECK: vmulh.vv v26, v24, v25, v0.t
vmulh.vv v26, v24, v25, v0.t

# CHECK: vmulh.vx v28, v27, t1
vmulh.vx v28, v27, t1
# CHECK: vmulh.vx v30, v29, s0, v0.t
vmulh.vx v30, v29, s0, v0.t

# CHECK: vmadd.vv v1, v0, v31
vmadd.vv v1, v0, v31
# CHECK: vmadd.vv v4, v3, v2, v0.t
vmadd.vv v4, v3, v2, v0.t

# CHECK: vmadd.vx v6, a0, v5
vmadd.vx v6, a0, v5
# CHECK: vmadd.vx v8, a2, v7, v0.t
vmadd.vx v8, a2, v7, v0.t

# CHECK: vmsub.vv v11, v10, v9
vmsub.vv v11, v10, v9
# CHECK: vmsub.vv v14, v13, v12, v0.t
vmsub.vv v14, v13, v12, v0.t

# CHECK: vmsub.vx v16, a4, v15
vmsub.vx v16, a4, v15
# CHECK: vmsub.vx v18, a6, v17, v0.t
vmsub.vx v18, a6, v17, v0.t

# CHECK: vmacc.vv v21, v20, v19
vmacc.vv v21, v20, v19
# CHECK: vmacc.vv v24, v23, v22, v0.t
vmacc.vv v24, v23, v22, v0.t

# CHECK: vmacc.vx v26, s2, v25
vmacc.vx v26, s2, v25
# CHECK: vmacc.vx v28, s4, v27, v0.t
vmacc.vx v28, s4, v27, v0.t

# CHECK: vmsac.vv v31, v30, v29
vmsac.vv v31, v30, v29
# CHECK: vmsac.vv v2, v1, v0, v0.t
vmsac.vv v2, v1, v0, v0.t

# CHECK: vmsac.vx v4, s6, v3
vmsac.vx v4, s6, v3
# CHECK: vmsac.vx v6, s8, v5, v0.t
vmsac.vx v6, s8, v5, v0.t

# CHECK: vwaddu.vv v9, v7, v8
vwaddu.vv v9, v7, v8
# CHECK: vwaddu.vv v12, v10, v11, v0.t
vwaddu.vv v12, v10, v11, v0.t

# CHECK: vwaddu.vx v14, v13, s10
vwaddu.vx v14, v13, s10
# CHECK: vwaddu.vx v16, v15, t3, v0.t
vwaddu.vx v16, v15, t3, v0.t

# CHECK: vwadd.vv v19, v17, v18
vwadd.vv v19, v17, v18
# CHECK: vwadd.vv v22, v20, v21, v0.t
vwadd.vv v22, v20, v21, v0.t

# CHECK: vwadd.vx v24, v23, t5
vwadd.vx v24, v23, t5
# CHECK: vwadd.vx v26, v25, ra, v0.t
vwadd.vx v26, v25, ra, v0.t

# CHECK: vwsubu.vv v29, v27, v28
vwsubu.vv v29, v27, v28
# CHECK: vwsubu.vv v0, v30, v31, v0.t
vwsubu.vv v0, v30, v31, v0.t

# CHECK: vwsubu.vx v2, v1, gp
vwsubu.vx v2, v1, gp
# CHECK: vwsubu.vx v4, v3, t0, v0.t
vwsubu.vx v4, v3, t0, v0.t

# CHECK: vwsub.vv v7, v5, v6
vwsub.vv v7, v5, v6
# CHECK: vwsub.vv v10, v8, v9, v0.t
vwsub.vv v10, v8, v9, v0.t

# CHECK: vwsub.vx v12, v11, t2
vwsub.vx v12, v11, t2
# CHECK: vwsub.vx v14, v13, s1, v0.t
vwsub.vx v14, v13, s1, v0.t

# CHECK: vwaddu.wv v17, v15, v16
vwaddu.wv v17, v15, v16
# CHECK: vwaddu.wv v20, v18, v19, v0.t
vwaddu.wv v20, v18, v19, v0.t

# CHECK: vwaddu.wx v22, v21, a1
vwaddu.wx v22, v21, a1
# CHECK: vwaddu.wx v24, v23, a3, v0.t
vwaddu.wx v24, v23, a3, v0.t

# CHECK: vwadd.wv v27, v25, v26
vwadd.wv v27, v25, v26
# CHECK: vwadd.wv v30, v28, v29, v0.t
vwadd.wv v30, v28, v29, v0.t

# CHECK: vwadd.wx v0, v31, a5
vwadd.wx v0, v31, a5
# CHECK: vwadd.wx v2, v1, a7, v0.t
vwadd.wx v2, v1, a7, v0.t

# CHECK: vwsubu.wv v5, v3, v4
vwsubu.wv v5, v3, v4
# CHECK: vwsubu.wv v8, v6, v7, v0.t
vwsubu.wv v8, v6, v7, v0.t

# CHECK: vwsubu.wx v10, v9, s3
vwsubu.wx v10, v9, s3
# CHECK: vwsubu.wx v12, v11, s5, v0.t
vwsubu.wx v12, v11, s5, v0.t

# CHECK: vwsub.wv v15, v13, v14
vwsub.wv v15, v13, v14
# CHECK: vwsub.wv v18, v16, v17, v0.t
vwsub.wv v18, v16, v17, v0.t

# CHECK: vwsub.wx v20, v19, s7
vwsub.wx v20, v19, s7
# CHECK: vwsub.wx v22, v21, s9, v0.t
vwsub.wx v22, v21, s9, v0.t

# CHECK: vwmulu.vv v25, v23, v24
vwmulu.vv v25, v23, v24
# CHECK: vwmulu.vv v28, v26, v27, v0.t
vwmulu.vv v28, v26, v27, v0.t

# CHECK: vwmulu.vx v30, v29, s11
vwmulu.vx v30, v29, s11
# CHECK: vwmulu.vx v0, v31, t4, v0.t
vwmulu.vx v0, v31, t4, v0.t

# CHECK: vwmulsu.vv v3, v1, v2
vwmulsu.vv v3, v1, v2
# CHECK: vwmulsu.vv v6, v4, v5, v0.t
vwmulsu.vv v6, v4, v5, v0.t

# CHECK: vwmulsu.vx v8, v7, t6
vwmulsu.vx v8, v7, t6
# CHECK: vwmulsu.vx v10, v9, sp, v0.t
vwmulsu.vx v10, v9, sp, v0.t

# CHECK: vwmul.vv v13, v11, v12
vwmul.vv v13, v11, v12
# CHECK: vwmul.vv v16, v14, v15, v0.t
vwmul.vv v16, v14, v15, v0.t

# CHECK: vwmul.vx v18, v17, tp
vwmul.vx v18, v17, tp
# CHECK: vwmul.vx v20, v19, t1, v0.t
vwmul.vx v20, v19, t1, v0.t

# CHECK: vwmaccu.vv v23, v21, v22
vwmaccu.vv v23, v21, v22
# CHECK: vwmaccu.vv v26, v24, v25, v0.t
vwmaccu.vv v26, v24, v25, v0.t

# CHECK: vwmaccu.vx v28, v27, s0
vwmaccu.vx v28, v27, s0
# CHECK: vwmaccu.vx v30, v29, a0, v0.t
vwmaccu.vx v30, v29, a0, v0.t

# CHECK: vwmacc.vv v1, v31, v0
vwmacc.vv v1, v31, v0
# CHECK: vwmacc.vv v4, v2, v3, v0.t
vwmacc.vv v4, v2, v3, v0.t

# CHECK: vwmacc.vx v6, v5, a2
vwmacc.vx v6, v5, a2
# CHECK: vwmacc.vx v8, v7, a4, v0.t
vwmacc.vx v8, v7, a4, v0.t

# CHECK: vwmsacu.vv v11, v9, v10
vwmsacu.vv v11, v9, v10
# CHECK: vwmsacu.vv v14, v12, v13, v0.t
vwmsacu.vv v14, v12, v13, v0.t

# CHECK: vwmsacu.vx v16, v15, a6
vwmsacu.vx v16, v15, a6
# CHECK: vwmsacu.vx v18, v17, s2, v0.t
vwmsacu.vx v18, v17, s2, v0.t

# CHECK: vwmsac.vv v21, v19, v20
vwmsac.vv v21, v19, v20
# CHECK: vwmsac.vv v24, v22, v23, v0.t
vwmsac.vv v24, v22, v23, v0.t

# CHECK: vwmsac.vx v26, v25, s4
vwmsac.vx v26, v25, s4
# CHECK: vwmsac.vx v28, v27, s6, v0.t
vwmsac.vx v28, v27, s6, v0.t

# CHECK: vfadd.vv v31, v29, v30
vfadd.vv v31, v29, v30
# CHECK: vfadd.vv v2, v0, v1, v0.t
vfadd.vv v2, v0, v1, v0.t

# CHECK: vfadd.vf v4, v3, ft0
vfadd.vf v4, v3, ft0
# CHECK: vfadd.vf v6, v5, ft1, v0.t
vfadd.vf v6, v5, ft1, v0.t

# CHECK: vfredsum.vs v9, v7, v8
vfredsum.vs v9, v7, v8
# CHECK: vfredsum.vs v12, v10, v11, v0.t
vfredsum.vs v12, v10, v11, v0.t

# CHECK: vfsub.vv v15, v13, v14
vfsub.vv v15, v13, v14
# CHECK: vfsub.vv v18, v16, v17, v0.t
vfsub.vv v18, v16, v17, v0.t

# CHECK: vfsub.vf v20, v19, ft2
vfsub.vf v20, v19, ft2
# CHECK: vfsub.vf v22, v21, ft3, v0.t
vfsub.vf v22, v21, ft3, v0.t

# CHECK: vfredosum.vs v25, v23, v24
vfredosum.vs v25, v23, v24
# CHECK: vfredosum.vs v28, v26, v27, v0.t
vfredosum.vs v28, v26, v27, v0.t

# CHECK: vfmin.vv v31, v29, v30
vfmin.vv v31, v29, v30
# CHECK: vfmin.vv v2, v0, v1, v0.t
vfmin.vv v2, v0, v1, v0.t

# CHECK: vfmin.vf v4, v3, ft4
vfmin.vf v4, v3, ft4
# CHECK: vfmin.vf v6, v5, ft5, v0.t
vfmin.vf v6, v5, ft5, v0.t

# CHECK: vfredmin.vs v9, v7, v8
vfredmin.vs v9, v7, v8
# CHECK: vfredmin.vs v12, v10, v11, v0.t
vfredmin.vs v12, v10, v11, v0.t

# CHECK: vfmax.vv v15, v13, v14
vfmax.vv v15, v13, v14
# CHECK: vfmax.vv v18, v16, v17, v0.t
vfmax.vv v18, v16, v17, v0.t

# CHECK: vfmax.vf v20, v19, ft6
vfmax.vf v20, v19, ft6
# CHECK: vfmax.vf v22, v21, ft7, v0.t
vfmax.vf v22, v21, ft7, v0.t

# CHECK: vfredmax.vs v25, v23, v24
vfredmax.vs v25, v23, v24
# CHECK: vfredmax.vs v28, v26, v27, v0.t
vfredmax.vs v28, v26, v27, v0.t

# CHECK: vfsgnj.vv v31, v29, v30
vfsgnj.vv v31, v29, v30
# CHECK: vfsgnj.vv v2, v0, v1, v0.t
vfsgnj.vv v2, v0, v1, v0.t

# CHECK: vfsgnj.vf v4, v3, fs0
vfsgnj.vf v4, v3, fs0
# CHECK: vfsgnj.vf v6, v5, fs1, v0.t
vfsgnj.vf v6, v5, fs1, v0.t

# CHECK: vfsgnjn.vv v9, v7, v8
vfsgnjn.vv v9, v7, v8
# CHECK: vfsgnjn.vv v12, v10, v11, v0.t
vfsgnjn.vv v12, v10, v11, v0.t

# CHECK: vfsgnjn.vf v14, v13, fa0
vfsgnjn.vf v14, v13, fa0
# CHECK: vfsgnjn.vf v16, v15, fa1, v0.t
vfsgnjn.vf v16, v15, fa1, v0.t

# CHECK: vfsgnjx.vv v19, v17, v18
vfsgnjx.vv v19, v17, v18
# CHECK: vfsgnjx.vv v22, v20, v21, v0.t
vfsgnjx.vv v22, v20, v21, v0.t

# CHECK: vfsgnjx.vf v24, v23, fa2
vfsgnjx.vf v24, v23, fa2
# CHECK: vfsgnjx.vf v26, v25, fa3, v0.t
vfsgnjx.vf v26, v25, fa3, v0.t

# CHECK: vfmv.f.s fa4, v27
vfmv.f.s fa4, v27

# CHECK: vfmv.s.f v28, fa5
vfmv.s.f v28, fa5

# CHECK: vfmerge.vf v30, v29, fa6
vfmerge.vf v30, v29, fa6
# CHECK: vfmerge.vf v0, v31, fa7, v0.t
vfmerge.vf v0, v31, fa7, v0.t

# CHECK: vfeq.vv v3, v1, v2
vfeq.vv v3, v1, v2
# CHECK: vfeq.vv v6, v4, v5, v0.t
vfeq.vv v6, v4, v5, v0.t

# CHECK: vfeq.vf v8, v7, fs2
vfeq.vf v8, v7, fs2
# CHECK: vfeq.vf v10, v9, fs3, v0.t
vfeq.vf v10, v9, fs3, v0.t

# CHECK: vflte.vv v13, v11, v12
vflte.vv v13, v11, v12
# CHECK: vflte.vv v16, v14, v15, v0.t
vflte.vv v16, v14, v15, v0.t

# CHECK: vflte.vf v18, v17, fs4
vflte.vf v18, v17, fs4
# CHECK: vflte.vf v20, v19, fs5, v0.t
vflte.vf v20, v19, fs5, v0.t

# CHECK: vford.vv v23, v21, v22
vford.vv v23, v21, v22
# CHECK: vford.vv v26, v24, v25, v0.t
vford.vv v26, v24, v25, v0.t

# CHECK: vford.vf v28, v27, fs6
vford.vf v28, v27, fs6
# CHECK: vford.vf v30, v29, fs7, v0.t
vford.vf v30, v29, fs7, v0.t

# CHECK: vflt.vv v1, v31, v0
vflt.vv v1, v31, v0
# CHECK: vflt.vv v4, v2, v3, v0.t
vflt.vv v4, v2, v3, v0.t

# CHECK: vflt.vf v6, v5, fs8
vflt.vf v6, v5, fs8
# CHECK: vflt.vf v8, v7, fs9, v0.t
vflt.vf v8, v7, fs9, v0.t

# CHECK: vfne.vv v11, v9, v10
vfne.vv v11, v9, v10
# CHECK: vfne.vv v14, v12, v13, v0.t
vfne.vv v14, v12, v13, v0.t

# CHECK: vfne.vf v16, v15, fs10
vfne.vf v16, v15, fs10
# CHECK: vfne.vf v18, v17, fs11, v0.t
vfne.vf v18, v17, fs11, v0.t

# CHECK: vfgt.vf v20, v19, ft8
vfgt.vf v20, v19, ft8
# CHECK: vfgt.vf v22, v21, ft9, v0.t
vfgt.vf v22, v21, ft9, v0.t

# CHECK: vfgte.vf v24, v23, ft10
vfgte.vf v24, v23, ft10
# CHECK: vfgte.vf v26, v25, ft0, v0.t
vfgte.vf v26, v25, ft0, v0.t

# CHECK: vfdiv.vv v29, v27, v28
vfdiv.vv v29, v27, v28
# CHECK: vfdiv.vv v0, v30, v31, v0.t
vfdiv.vv v0, v30, v31, v0.t

# CHECK: vfdiv.vf v2, v1, ft1
vfdiv.vf v2, v1, ft1
# CHECK: vfdiv.vf v4, v3, ft2, v0.t
vfdiv.vf v4, v3, ft2, v0.t

# CHECK: vfrdiv.vf v6, v5, ft3
vfrdiv.vf v6, v5, ft3
# CHECK: vfrdiv.vf v8, v7, ft4, v0.t
vfrdiv.vf v8, v7, ft4, v0.t

# CHECK: vfmul.vv v11, v9, v10
vfmul.vv v11, v9, v10
# CHECK: vfmul.vv v14, v12, v13, v0.t
vfmul.vv v14, v12, v13, v0.t

# CHECK: vfmul.vf v16, v15, ft5
vfmul.vf v16, v15, ft5
# CHECK: vfmul.vf v18, v17, ft6, v0.t
vfmul.vf v18, v17, ft6, v0.t

# CHECK: vfmadd.vv v21, v20, v19
vfmadd.vv v21, v20, v19
# CHECK: vfmadd.vv v24, v23, v22, v0.t
vfmadd.vv v24, v23, v22, v0.t

# CHECK: vfmadd.vf v26, ft7, v25
vfmadd.vf v26, ft7, v25
# CHECK: vfmadd.vf v28, fs0, v27, v0.t
vfmadd.vf v28, fs0, v27, v0.t

# CHECK: vfnmadd.vv v31, v30, v29
vfnmadd.vv v31, v30, v29
# CHECK: vfnmadd.vv v2, v1, v0, v0.t
vfnmadd.vv v2, v1, v0, v0.t

# CHECK: vfnmadd.vf v4, fs1, v3
vfnmadd.vf v4, fs1, v3
# CHECK: vfnmadd.vf v6, fa0, v5, v0.t
vfnmadd.vf v6, fa0, v5, v0.t

# CHECK: vfmsub.vv v9, v8, v7
vfmsub.vv v9, v8, v7
# CHECK: vfmsub.vv v12, v11, v10, v0.t
vfmsub.vv v12, v11, v10, v0.t

# CHECK: vfmsub.vf v14, fa1, v13
vfmsub.vf v14, fa1, v13
# CHECK: vfmsub.vf v16, fa2, v15, v0.t
vfmsub.vf v16, fa2, v15, v0.t

# CHECK: vfnmsub.vv v19, v18, v17
vfnmsub.vv v19, v18, v17
# CHECK: vfnmsub.vv v22, v21, v20, v0.t
vfnmsub.vv v22, v21, v20, v0.t

# CHECK: vfnmsub.vf v24, fa3, v23
vfnmsub.vf v24, fa3, v23
# CHECK: vfnmsub.vf v26, fa4, v25, v0.t
vfnmsub.vf v26, fa4, v25, v0.t

# CHECK: vfmacc.vv v29, v28, v27
vfmacc.vv v29, v28, v27
# CHECK: vfmacc.vv v0, v31, v30, v0.t
vfmacc.vv v0, v31, v30, v0.t

# CHECK: vfmacc.vf v2, fa5, v1
vfmacc.vf v2, fa5, v1
# CHECK: vfmacc.vf v4, fa6, v3, v0.t
vfmacc.vf v4, fa6, v3, v0.t

# CHECK: vfnmacc.vv v7, v6, v5
vfnmacc.vv v7, v6, v5
# CHECK: vfnmacc.vv v10, v9, v8, v0.t
vfnmacc.vv v10, v9, v8, v0.t

# CHECK: vfnmacc.vf v12, fa7, v11
vfnmacc.vf v12, fa7, v11
# CHECK: vfnmacc.vf v14, fs2, v13, v0.t
vfnmacc.vf v14, fs2, v13, v0.t

# CHECK: vfmsac.vv v17, v16, v15
vfmsac.vv v17, v16, v15
# CHECK: vfmsac.vv v20, v19, v18, v0.t
vfmsac.vv v20, v19, v18, v0.t

# CHECK: vfmsac.vf v22, fs3, v21
vfmsac.vf v22, fs3, v21
# CHECK: vfmsac.vf v24, fs4, v23, v0.t
vfmsac.vf v24, fs4, v23, v0.t

# CHECK: vfnmsac.vv v27, v26, v25
vfnmsac.vv v27, v26, v25
# CHECK: vfnmsac.vv v30, v29, v28, v0.t
vfnmsac.vv v30, v29, v28, v0.t

# CHECK: vfnmsac.vf v0, fs5, v31
vfnmsac.vf v0, fs5, v31
# CHECK: vfnmsac.vf v2, fs6, v1, v0.t
vfnmsac.vf v2, fs6, v1, v0.t

# CHECK: vfwadd.vv v5, v3, v4
vfwadd.vv v5, v3, v4
# CHECK: vfwadd.vv v8, v6, v7, v0.t
vfwadd.vv v8, v6, v7, v0.t

# CHECK: vfwadd.vf v10, v9, fs7
vfwadd.vf v10, v9, fs7
# CHECK: vfwadd.vf v12, v11, fs8, v0.t
vfwadd.vf v12, v11, fs8, v0.t

# CHECK: vfwredsum.vs v15, v13, v14
vfwredsum.vs v15, v13, v14
# CHECK: vfwredsum.vs v18, v16, v17, v0.t
vfwredsum.vs v18, v16, v17, v0.t

# CHECK: vfwsub.vv v21, v19, v20
vfwsub.vv v21, v19, v20
# CHECK: vfwsub.vv v24, v22, v23, v0.t
vfwsub.vv v24, v22, v23, v0.t

# CHECK: vfwsub.vf v26, v25, fs9
vfwsub.vf v26, v25, fs9
# CHECK: vfwsub.vf v28, v27, fs10, v0.t
vfwsub.vf v28, v27, fs10, v0.t

# CHECK: vfwredosum.vs v31, v29, v30
vfwredosum.vs v31, v29, v30
# CHECK: vfwredosum.vs v2, v0, v1, v0.t
vfwredosum.vs v2, v0, v1, v0.t

# CHECK: vfwadd.wv v5, v3, v4
vfwadd.wv v5, v3, v4
# CHECK: vfwadd.wv v8, v6, v7, v0.t
vfwadd.wv v8, v6, v7, v0.t

# CHECK: vfwadd.wf v10, v9, fs11
vfwadd.wf v10, v9, fs11
# CHECK: vfwadd.wf v12, v11, ft8, v0.t
vfwadd.wf v12, v11, ft8, v0.t

# CHECK: vfwsub.wv v15, v13, v14
vfwsub.wv v15, v13, v14
# CHECK: vfwsub.wv v18, v16, v17, v0.t
vfwsub.wv v18, v16, v17, v0.t

# CHECK: vfwsub.wf v20, v19, ft9
vfwsub.wf v20, v19, ft9
# CHECK: vfwsub.wf v22, v21, ft10, v0.t
vfwsub.wf v22, v21, ft10, v0.t

# CHECK: vfwmul.vv v25, v23, v24
vfwmul.vv v25, v23, v24
# CHECK: vfwmul.vv v28, v26, v27, v0.t
vfwmul.vv v28, v26, v27, v0.t

# CHECK: vfwmul.vf v30, v29, ft0
vfwmul.vf v30, v29, ft0
# CHECK: vfwmul.vf v0, v31, ft1, v0.t
vfwmul.vf v0, v31, ft1, v0.t

# CHECK: vfdot.vv v3, v1, v2
vfdot.vv v3, v1, v2
# CHECK: vfdot.vv v6, v4, v5, v0.t
vfdot.vv v6, v4, v5, v0.t

# CHECK: vfwmacc.vv v9, v8, v7
vfwmacc.vv v9, v8, v7
# CHECK: vfwmacc.vv v12, v11, v10, v0.t
vfwmacc.vv v12, v11, v10, v0.t

# CHECK: vfwmacc.vf v14, ft2, v13
vfwmacc.vf v14, ft2, v13
# CHECK: vfwmacc.vf v16, ft3, v15, v0.t
vfwmacc.vf v16, ft3, v15, v0.t

# CHECK: vfwnmacc.vv v19, v18, v17
vfwnmacc.vv v19, v18, v17
# CHECK: vfwnmacc.vv v22, v21, v20, v0.t
vfwnmacc.vv v22, v21, v20, v0.t

# CHECK: vfwnmacc.vf v24, ft4, v23
vfwnmacc.vf v24, ft4, v23
# CHECK: vfwnmacc.vf v26, ft5, v25, v0.t
vfwnmacc.vf v26, ft5, v25, v0.t

# CHECK: vfwmsac.vv v29, v28, v27
vfwmsac.vv v29, v28, v27
# CHECK: vfwmsac.vv v0, v31, v30, v0.t
vfwmsac.vv v0, v31, v30, v0.t

# CHECK: vfwmsac.vf v2, ft6, v1
vfwmsac.vf v2, ft6, v1
# CHECK: vfwmsac.vf v4, ft7, v3, v0.t
vfwmsac.vf v4, ft7, v3, v0.t

# CHECK: vfwnmsac.vv v7, v6, v5
vfwnmsac.vv v7, v6, v5
# CHECK: vfwnmsac.vv v10, v9, v8, v0.t
vfwnmsac.vv v10, v9, v8, v0.t

# CHECK: vfwnmsac.vf v12, fs0, v11
vfwnmsac.vf v12, fs0, v11
# CHECK: vfwnmsac.vf v14, fs1, v13, v0.t
vfwnmsac.vf v14, fs1, v13, v0.t

# CHECK: vfsqrt.v v16, v15
vfsqrt.v v16, v15
# CHECK: vfsqrt.v v18, v17, v0.t
vfsqrt.v v18, v17, v0.t

# CHECK: vfclass.v v20, v19
vfclass.v v20, v19
# CHECK: vfclass.v v22, v21, v0.t
vfclass.v v22, v21, v0.t

# CHECK: vfcvt.xu.f.v v24, v23
vfcvt.xu.f.v v24, v23
# CHECK: vfcvt.xu.f.v v26, v25, v0.t
vfcvt.xu.f.v v26, v25, v0.t

# CHECK: vfcvt.x.f.v v28, v27
vfcvt.x.f.v v28, v27
# CHECK: vfcvt.x.f.v v30, v29, v0.t
vfcvt.x.f.v v30, v29, v0.t

# CHECK: vfcvt.f.xu.v v0, v31
vfcvt.f.xu.v v0, v31
# CHECK: vfcvt.f.xu.v v2, v1, v0.t
vfcvt.f.xu.v v2, v1, v0.t

# CHECK: vfcvt.f.x.v v4, v3
vfcvt.f.x.v v4, v3
# CHECK: vfcvt.f.x.v v6, v5, v0.t
vfcvt.f.x.v v6, v5, v0.t

# CHECK: vfwcvt.xu.f.v v8, v7
vfwcvt.xu.f.v v8, v7
# CHECK: vfwcvt.xu.f.v v10, v9, v0.t
vfwcvt.xu.f.v v10, v9, v0.t

# CHECK: vfwcvt.x.f.v v12, v11
vfwcvt.x.f.v v12, v11
# CHECK: vfwcvt.x.f.v v14, v13, v0.t
vfwcvt.x.f.v v14, v13, v0.t

# CHECK: vfwcvt.f.xu.v v16, v15
vfwcvt.f.xu.v v16, v15
# CHECK: vfwcvt.f.xu.v v18, v17, v0.t
vfwcvt.f.xu.v v18, v17, v0.t

# CHECK: vfwcvt.f.x.v v20, v19
vfwcvt.f.x.v v20, v19
# CHECK: vfwcvt.f.x.v v22, v21, v0.t
vfwcvt.f.x.v v22, v21, v0.t

# CHECK: vfwcvt.f.f.v v24, v23
vfwcvt.f.f.v v24, v23
# CHECK: vfwcvt.f.f.v v26, v25, v0.t
vfwcvt.f.f.v v26, v25, v0.t

# CHECK: vfncvt.xu.f.v v28, v27
vfncvt.xu.f.v v28, v27
# CHECK: vfncvt.xu.f.v v30, v29, v0.t
vfncvt.xu.f.v v30, v29, v0.t

# CHECK: vfncvt.x.f.v v0, v31
vfncvt.x.f.v v0, v31
# CHECK: vfncvt.x.f.v v2, v1, v0.t
vfncvt.x.f.v v2, v1, v0.t

# CHECK: vfncvt.f.xu.v v4, v3
vfncvt.f.xu.v v4, v3
# CHECK: vfncvt.f.xu.v v6, v5, v0.t
vfncvt.f.xu.v v6, v5, v0.t

# CHECK: vfncvt.f.x.v v8, v7
vfncvt.f.x.v v8, v7
# CHECK: vfncvt.f.x.v v10, v9, v0.t
vfncvt.f.x.v v10, v9, v0.t

# CHECK: vfncvt.f.f.v v12, v11
vfncvt.f.f.v v12, v11
# CHECK: vfncvt.f.f.v v14, v13, v0.t
vfncvt.f.f.v v14, v13, v0.t

# CHECK: vmsbf.m v16, v15
vmsbf.m v16, v15
# CHECK: vmsbf.m v18, v17, v0.t
vmsbf.m v18, v17, v0.t

# CHECK: vmsof.m v20, v19
vmsof.m v20, v19
# CHECK: vmsof.m v22, v21, v0.t
vmsof.m v22, v21, v0.t

# CHECK: vmsif.m v24, v23
vmsif.m v24, v23
# CHECK: vmsif.m v26, v25, v0.t
vmsif.m v26, v25, v0.t

# CHECK: vmiota.m v28, v27
vmiota.m v28, v27
# CHECK: vmiota.m v30, v29, v0.t
vmiota.m v30, v29, v0.t

# CHECK: vid.v v31
vid.v v31
# CHECK: vid.v v0, v0.t
vid.v v0, v0.t

# CHECK: vlb.v v1, (s8)
vlb.v v1, (s8)
# CHECK: vlb.v v2, (s10), v0.t
vlb.v v2, (s10), v0.t

# CHECK: vlh.v v3, (t3)
vlh.v v3, (t3)
# CHECK: vlh.v v4, (t5), v0.t
vlh.v v4, (t5), v0.t

# CHECK: vlw.v v5, (ra)
vlw.v v5, (ra)
# CHECK: vlw.v v6, (gp), v0.t
vlw.v v6, (gp), v0.t

# CHECK: vlbu.v v7, (t0)
vlbu.v v7, (t0)
# CHECK: vlbu.v v8, (t2), v0.t
vlbu.v v8, (t2), v0.t

# CHECK: vlhu.v v9, (s1)
vlhu.v v9, (s1)
# CHECK: vlhu.v v10, (a1), v0.t
vlhu.v v10, (a1), v0.t

# CHECK: vlwu.v v11, (a3)
vlwu.v v11, (a3)
# CHECK: vlwu.v v12, (a5), v0.t
vlwu.v v12, (a5), v0.t

# CHECK: vle.v v13, (a7)
vle.v v13, (a7)
# CHECK: vle.v v14, (s3), v0.t
vle.v v14, (s3), v0.t

# CHECK: vsb.v v15, (s5)
vsb.v v15, (s5)
# CHECK: vsb.v v16, (s7), v0.t
vsb.v v16, (s7), v0.t

# CHECK: vsh.v v17, (s9)
vsh.v v17, (s9)
# CHECK: vsh.v v18, (s11), v0.t
vsh.v v18, (s11), v0.t

# CHECK: vsw.v v19, (t4)
vsw.v v19, (t4)
# CHECK: vsw.v v20, (t6), v0.t
vsw.v v20, (t6), v0.t

# CHECK: vse.v v21, (sp)
vse.v v21, (sp)
# CHECK: vse.v v22, (tp), v0.t
vse.v v22, (tp), v0.t

# CHECK: vlsb.v v23, (s0), t1
vlsb.v v23, (s0), t1
# CHECK: vlsb.v v24, (a2), a0, v0.t
vlsb.v v24, (a2), a0, v0.t

# CHECK: vlsh.v v25, (a6), a4
vlsh.v v25, (a6), a4
# CHECK: vlsh.v v26, (s4), s2, v0.t
vlsh.v v26, (s4), s2, v0.t

# CHECK: vlsw.v v27, (s8), s6
vlsw.v v27, (s8), s6
# CHECK: vlsw.v v28, (t3), s10, v0.t
vlsw.v v28, (t3), s10, v0.t

# CHECK: vlsbu.v v29, (ra), t5
vlsbu.v v29, (ra), t5
# CHECK: vlsbu.v v30, (t0), gp, v0.t
vlsbu.v v30, (t0), gp, v0.t

# CHECK: vlshu.v v31, (s1), t2
vlshu.v v31, (s1), t2
# CHECK: vlshu.v v0, (a3), a1, v0.t
vlshu.v v0, (a3), a1, v0.t

# CHECK: vlswu.v v1, (a7), a5
vlswu.v v1, (a7), a5
# CHECK: vlswu.v v2, (s5), s3, v0.t
vlswu.v v2, (s5), s3, v0.t

# CHECK: vlse.v v3, (s9), s7
vlse.v v3, (s9), s7
# CHECK: vlse.v v4, (t4), s11, v0.t
vlse.v v4, (t4), s11, v0.t

# CHECK: vssb.v v5, (sp), t6
vssb.v v5, (sp), t6
# CHECK: vssb.v v6, (t1), tp, v0.t
vssb.v v6, (t1), tp, v0.t

# CHECK: vssh.v v7, (a0), s0
vssh.v v7, (a0), s0
# CHECK: vssh.v v8, (a4), a2, v0.t
vssh.v v8, (a4), a2, v0.t

# CHECK: vssw.v v9, (s2), a6
vssw.v v9, (s2), a6
# CHECK: vssw.v v10, (s6), s4, v0.t
vssw.v v10, (s6), s4, v0.t

# CHECK: vsse.v v11, (s10), s8
vsse.v v11, (s10), s8
# CHECK: vsse.v v12, (t5), t3, v0.t
vsse.v v12, (t5), t3, v0.t

# CHECK: vlxb.v v14, (ra), v13
vlxb.v v14, (ra), v13
# CHECK: vlxb.v v16, (gp), v15, v0.t
vlxb.v v16, (gp), v15, v0.t

# CHECK: vlxh.v v18, (t0), v17
vlxh.v v18, (t0), v17
# CHECK: vlxh.v v20, (t2), v19, v0.t
vlxh.v v20, (t2), v19, v0.t

# CHECK: vlxw.v v22, (s1), v21
vlxw.v v22, (s1), v21
# CHECK: vlxw.v v24, (a1), v23, v0.t
vlxw.v v24, (a1), v23, v0.t

# CHECK: vlxbu.v v26, (a3), v25
vlxbu.v v26, (a3), v25
# CHECK: vlxbu.v v28, (a5), v27, v0.t
vlxbu.v v28, (a5), v27, v0.t

# CHECK: vlxhu.v v30, (a7), v29
vlxhu.v v30, (a7), v29
# CHECK: vlxhu.v v0, (s3), v31, v0.t
vlxhu.v v0, (s3), v31, v0.t

# CHECK: vlxwu.v v2, (s5), v1
vlxwu.v v2, (s5), v1
# CHECK: vlxwu.v v4, (s7), v3, v0.t
vlxwu.v v4, (s7), v3, v0.t

# CHECK: vlxe.v v6, (s9), v5
vlxe.v v6, (s9), v5
# CHECK: vlxe.v v8, (s11), v7, v0.t
vlxe.v v8, (s11), v7, v0.t

# CHECK: vsxb.v v10, (t4), v9
vsxb.v v10, (t4), v9
# CHECK: vsxb.v v12, (t6), v11, v0.t
vsxb.v v12, (t6), v11, v0.t

# CHECK: vsxh.v v14, (sp), v13
vsxh.v v14, (sp), v13
# CHECK: vsxh.v v16, (tp), v15, v0.t
vsxh.v v16, (tp), v15, v0.t

# CHECK: vsxw.v v18, (t1), v17
vsxw.v v18, (t1), v17
# CHECK: vsxw.v v20, (s0), v19, v0.t
vsxw.v v20, (s0), v19, v0.t

# CHECK: vsxe.v v22, (a0), v21
vsxe.v v22, (a0), v21
# CHECK: vsxe.v v24, (a2), v23, v0.t
vsxe.v v24, (a2), v23, v0.t

# CHECK: vsuxb.v v26, (a4), v25
vsuxb.v v26, (a4), v25
# CHECK: vsuxb.v v28, (a6), v27, v0.t
vsuxb.v v28, (a6), v27, v0.t

# CHECK: vsuxh.v v30, (s2), v29
vsuxh.v v30, (s2), v29
# CHECK: vsuxh.v v0, (s4), v31, v0.t
vsuxh.v v0, (s4), v31, v0.t

# CHECK: vsuxw.v v2, (s6), v1
vsuxw.v v2, (s6), v1
# CHECK: vsuxw.v v4, (s8), v3, v0.t
vsuxw.v v4, (s8), v3, v0.t

# CHECK: vsuxe.v v6, (s10), v5
vsuxe.v v6, (s10), v5
# CHECK: vsuxe.v v8, (t3), v7, v0.t
vsuxe.v v8, (t3), v7, v0.t

