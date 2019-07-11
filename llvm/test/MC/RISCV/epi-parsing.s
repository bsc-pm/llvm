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

# CHECK: vadc.vvm v18, v16, v17, v0
vadc.vvm v18, v16, v17, v0

# CHECK: vadc.vxm v20, v19, s6, v0
vadc.vxm v20, v19, s6, v0

# CHECK: vadc.vim v22, v21, -16, v0
vadc.vim v22, v21, -16, v0

# CHECK: vsbc.vvm v25, v23, v24, v0
vsbc.vvm v25, v23, v24, v0

# CHECK: vsbc.vxm v27, v26, s8, v0
vsbc.vxm v27, v26, s8, v0

# CHECK: vmv.v.v v29, v28
vmv.v.v v29, v28

# CHECK: vmv.v.x v30, s10
vmv.v.x v30, s10

# CHECK: vmv.v.i v31, -15
vmv.v.i v31, -15

# CHECK: vmerge.vvm v2, v0, v1, v0
vmerge.vvm v2, v0, v1, v0

# CHECK: vmerge.vxm v4, v3, t3, v0
vmerge.vxm v4, v3, t3, v0

# CHECK: vmerge.vim v6, v5, -14, v0
vmerge.vim v6, v5, -14, v0

# CHECK: vmseq.vv v9, v7, v8
vmseq.vv v9, v7, v8
# CHECK: vmseq.vv v12, v10, v11, v0.t
vmseq.vv v12, v10, v11, v0.t

# CHECK: vmseq.vx v14, v13, t5
vmseq.vx v14, v13, t5
# CHECK: vmseq.vx v16, v15, ra, v0.t
vmseq.vx v16, v15, ra, v0.t

# CHECK: vmseq.vi v18, v17, -13
vmseq.vi v18, v17, -13
# CHECK: vmseq.vi v20, v19, -12, v0.t
vmseq.vi v20, v19, -12, v0.t

# CHECK: vmsne.vv v23, v21, v22
vmsne.vv v23, v21, v22
# CHECK: vmsne.vv v26, v24, v25, v0.t
vmsne.vv v26, v24, v25, v0.t

# CHECK: vmsne.vx v28, v27, gp
vmsne.vx v28, v27, gp
# CHECK: vmsne.vx v30, v29, t0, v0.t
vmsne.vx v30, v29, t0, v0.t

# CHECK: vmsne.vi v0, v31, -11
vmsne.vi v0, v31, -11
# CHECK: vmsne.vi v2, v1, -10, v0.t
vmsne.vi v2, v1, -10, v0.t

# CHECK: vmsltu.vv v5, v3, v4
vmsltu.vv v5, v3, v4
# CHECK: vmsltu.vv v8, v6, v7, v0.t
vmsltu.vv v8, v6, v7, v0.t

# CHECK: vmsltu.vx v10, v9, t2
vmsltu.vx v10, v9, t2
# CHECK: vmsltu.vx v12, v11, s1, v0.t
vmsltu.vx v12, v11, s1, v0.t

# CHECK: vmslt.vv v15, v13, v14
vmslt.vv v15, v13, v14
# CHECK: vmslt.vv v18, v16, v17, v0.t
vmslt.vv v18, v16, v17, v0.t

# CHECK: vmslt.vx v20, v19, a1
vmslt.vx v20, v19, a1
# CHECK: vmslt.vx v22, v21, a3, v0.t
vmslt.vx v22, v21, a3, v0.t

# CHECK: vmsleu.vv v25, v23, v24
vmsleu.vv v25, v23, v24
# CHECK: vmsleu.vv v28, v26, v27, v0.t
vmsleu.vv v28, v26, v27, v0.t

# CHECK: vmsleu.vx v30, v29, a5
vmsleu.vx v30, v29, a5
# CHECK: vmsleu.vx v0, v31, a7, v0.t
vmsleu.vx v0, v31, a7, v0.t

# CHECK: vmsleu.vi v2, v1, -9
vmsleu.vi v2, v1, -9
# CHECK: vmsleu.vi v4, v3, -8, v0.t
vmsleu.vi v4, v3, -8, v0.t

# CHECK: vmsle.vv v7, v5, v6
vmsle.vv v7, v5, v6
# CHECK: vmsle.vv v10, v8, v9, v0.t
vmsle.vv v10, v8, v9, v0.t

# CHECK: vmsle.vx v12, v11, s3
vmsle.vx v12, v11, s3
# CHECK: vmsle.vx v14, v13, s5, v0.t
vmsle.vx v14, v13, s5, v0.t

# CHECK: vmsle.vi v16, v15, -7
vmsle.vi v16, v15, -7
# CHECK: vmsle.vi v18, v17, -6, v0.t
vmsle.vi v18, v17, -6, v0.t

# CHECK: vmsgtu.vx v20, v19, s7
vmsgtu.vx v20, v19, s7
# CHECK: vmsgtu.vx v22, v21, s9, v0.t
vmsgtu.vx v22, v21, s9, v0.t

# CHECK: vmsgtu.vi v24, v23, 27
vmsgtu.vi v24, v23, 27
# CHECK: vmsgtu.vi v26, v25, 28, v0.t
vmsgtu.vi v26, v25, 28, v0.t

# CHECK: vmsgt.vx v28, v27, s11
vmsgt.vx v28, v27, s11
# CHECK: vmsgt.vx v30, v29, t4, v0.t
vmsgt.vx v30, v29, t4, v0.t

# CHECK: vmsgt.vi v0, v31, -3
vmsgt.vi v0, v31, -3
# CHECK: vmsgt.vi v2, v1, -2, v0.t
vmsgt.vi v2, v1, -2, v0.t

# CHECK: vsaddu.vv v5, v3, v4
vsaddu.vv v5, v3, v4
# CHECK: vsaddu.vv v8, v6, v7, v0.t
vsaddu.vv v8, v6, v7, v0.t

# CHECK: vsaddu.vx v10, v9, t6
vsaddu.vx v10, v9, t6
# CHECK: vsaddu.vx v12, v11, sp, v0.t
vsaddu.vx v12, v11, sp, v0.t

# CHECK: vsaddu.vi v14, v13, -1
vsaddu.vi v14, v13, -1
# CHECK: vsaddu.vi v16, v15, 0, v0.t
vsaddu.vi v16, v15, 0, v0.t

# CHECK: vsadd.vv v19, v17, v18
vsadd.vv v19, v17, v18
# CHECK: vsadd.vv v22, v20, v21, v0.t
vsadd.vv v22, v20, v21, v0.t

# CHECK: vsadd.vx v24, v23, tp
vsadd.vx v24, v23, tp
# CHECK: vsadd.vx v26, v25, t1, v0.t
vsadd.vx v26, v25, t1, v0.t

# CHECK: vsadd.vi v28, v27, 1
vsadd.vi v28, v27, 1
# CHECK: vsadd.vi v30, v29, 2, v0.t
vsadd.vi v30, v29, 2, v0.t

# CHECK: vssubu.vv v1, v31, v0
vssubu.vv v1, v31, v0
# CHECK: vssubu.vv v4, v2, v3, v0.t
vssubu.vv v4, v2, v3, v0.t

# CHECK: vssubu.vx v6, v5, s0
vssubu.vx v6, v5, s0
# CHECK: vssubu.vx v8, v7, a0, v0.t
vssubu.vx v8, v7, a0, v0.t

# CHECK: vssub.vv v11, v9, v10
vssub.vv v11, v9, v10
# CHECK: vssub.vv v14, v12, v13, v0.t
vssub.vv v14, v12, v13, v0.t

# CHECK: vssub.vx v16, v15, a2
vssub.vx v16, v15, a2
# CHECK: vssub.vx v18, v17, a4, v0.t
vssub.vx v18, v17, a4, v0.t

# CHECK: vaadd.vv v21, v19, v20
vaadd.vv v21, v19, v20
# CHECK: vaadd.vv v24, v22, v23, v0.t
vaadd.vv v24, v22, v23, v0.t

# CHECK: vaadd.vx v26, v25, a6
vaadd.vx v26, v25, a6
# CHECK: vaadd.vx v28, v27, s2, v0.t
vaadd.vx v28, v27, s2, v0.t

# CHECK: vaadd.vi v30, v29, 3
vaadd.vi v30, v29, 3
# CHECK: vaadd.vi v0, v31, 4, v0.t
vaadd.vi v0, v31, 4, v0.t

# CHECK: vsll.vv v3, v1, v2
vsll.vv v3, v1, v2
# CHECK: vsll.vv v6, v4, v5, v0.t
vsll.vv v6, v4, v5, v0.t

# CHECK: vsll.vx v8, v7, s4
vsll.vx v8, v7, s4
# CHECK: vsll.vx v10, v9, s6, v0.t
vsll.vx v10, v9, s6, v0.t

# CHECK: vsll.vi v12, v11, 5
vsll.vi v12, v11, 5
# CHECK: vsll.vi v14, v13, 6, v0.t
vsll.vi v14, v13, 6, v0.t

# CHECK: vasub.vv v17, v15, v16
vasub.vv v17, v15, v16
# CHECK: vasub.vv v20, v18, v19, v0.t
vasub.vv v20, v18, v19, v0.t

# CHECK: vasub.vx v22, v21, s8
vasub.vx v22, v21, s8
# CHECK: vasub.vx v24, v23, s10, v0.t
vasub.vx v24, v23, s10, v0.t

# CHECK: vsmul.vv v27, v25, v26
vsmul.vv v27, v25, v26
# CHECK: vsmul.vv v30, v28, v29, v0.t
vsmul.vv v30, v28, v29, v0.t

# CHECK: vsmul.vx v0, v31, t3
vsmul.vx v0, v31, t3
# CHECK: vsmul.vx v2, v1, t5, v0.t
vsmul.vx v2, v1, t5, v0.t

# CHECK: vsrl.vv v5, v3, v4
vsrl.vv v5, v3, v4
# CHECK: vsrl.vv v8, v6, v7, v0.t
vsrl.vv v8, v6, v7, v0.t

# CHECK: vsrl.vx v10, v9, ra
vsrl.vx v10, v9, ra
# CHECK: vsrl.vx v12, v11, gp, v0.t
vsrl.vx v12, v11, gp, v0.t

# CHECK: vsrl.vi v14, v13, 7
vsrl.vi v14, v13, 7
# CHECK: vsrl.vi v16, v15, 8, v0.t
vsrl.vi v16, v15, 8, v0.t

# CHECK: vsra.vv v19, v17, v18
vsra.vv v19, v17, v18
# CHECK: vsra.vv v22, v20, v21, v0.t
vsra.vv v22, v20, v21, v0.t

# CHECK: vsra.vx v24, v23, t0
vsra.vx v24, v23, t0
# CHECK: vsra.vx v26, v25, t2, v0.t
vsra.vx v26, v25, t2, v0.t

# CHECK: vsra.vi v28, v27, 9
vsra.vi v28, v27, 9
# CHECK: vsra.vi v30, v29, 10, v0.t
vsra.vi v30, v29, 10, v0.t

# CHECK: vssrl.vv v1, v31, v0
vssrl.vv v1, v31, v0
# CHECK: vssrl.vv v4, v2, v3, v0.t
vssrl.vv v4, v2, v3, v0.t

# CHECK: vssrl.vx v6, v5, s1
vssrl.vx v6, v5, s1
# CHECK: vssrl.vx v8, v7, a1, v0.t
vssrl.vx v8, v7, a1, v0.t

# CHECK: vssrl.vi v10, v9, 11
vssrl.vi v10, v9, 11
# CHECK: vssrl.vi v12, v11, 12, v0.t
vssrl.vi v12, v11, 12, v0.t

# CHECK: vssra.vv v15, v13, v14
vssra.vv v15, v13, v14
# CHECK: vssra.vv v18, v16, v17, v0.t
vssra.vv v18, v16, v17, v0.t

# CHECK: vssra.vx v20, v19, a3
vssra.vx v20, v19, a3
# CHECK: vssra.vx v22, v21, a5, v0.t
vssra.vx v22, v21, a5, v0.t

# CHECK: vssra.vi v24, v23, 13
vssra.vi v24, v23, 13
# CHECK: vssra.vi v26, v25, 14, v0.t
vssra.vi v26, v25, 14, v0.t

# CHECK: vnsrl.vv v29, v27, v28
vnsrl.vv v29, v27, v28
# CHECK: vnsrl.vv v0, v30, v31, v0.t
vnsrl.vv v0, v30, v31, v0.t

# CHECK: vnsrl.vx v2, v1, a7
vnsrl.vx v2, v1, a7
# CHECK: vnsrl.vx v4, v3, s3, v0.t
vnsrl.vx v4, v3, s3, v0.t

# CHECK: vnsrl.vi v6, v5, 15
vnsrl.vi v6, v5, 15
# CHECK: vnsrl.vi v8, v7, -16, v0.t
vnsrl.vi v8, v7, -16, v0.t

# CHECK: vnsra.vv v11, v9, v10
vnsra.vv v11, v9, v10
# CHECK: vnsra.vv v14, v12, v13, v0.t
vnsra.vv v14, v12, v13, v0.t

# CHECK: vnsra.vx v16, v15, s5
vnsra.vx v16, v15, s5
# CHECK: vnsra.vx v18, v17, s7, v0.t
vnsra.vx v18, v17, s7, v0.t

# CHECK: vnsra.vi v20, v19, -15
vnsra.vi v20, v19, -15
# CHECK: vnsra.vi v22, v21, -14, v0.t
vnsra.vi v22, v21, -14, v0.t

# CHECK: vnclipu.vv v25, v23, v24
vnclipu.vv v25, v23, v24
# CHECK: vnclipu.vv v28, v26, v27, v0.t
vnclipu.vv v28, v26, v27, v0.t

# CHECK: vnclipu.vx v30, v29, s9
vnclipu.vx v30, v29, s9
# CHECK: vnclipu.vx v0, v31, s11, v0.t
vnclipu.vx v0, v31, s11, v0.t

# CHECK: vnclipu.vi v2, v1, -13
vnclipu.vi v2, v1, -13
# CHECK: vnclipu.vi v4, v3, -12, v0.t
vnclipu.vi v4, v3, -12, v0.t

# CHECK: vnclip.vv v7, v5, v6
vnclip.vv v7, v5, v6
# CHECK: vnclip.vv v10, v8, v9, v0.t
vnclip.vv v10, v8, v9, v0.t

# CHECK: vnclip.vx v12, v11, t4
vnclip.vx v12, v11, t4
# CHECK: vnclip.vx v14, v13, t6, v0.t
vnclip.vx v14, v13, t6, v0.t

# CHECK: vnclip.vi v16, v15, -11
vnclip.vi v16, v15, -11
# CHECK: vnclip.vi v18, v17, -10, v0.t
vnclip.vi v18, v17, -10, v0.t

# CHECK: vwredsumu.vs v21, v19, v20
vwredsumu.vs v21, v19, v20
# CHECK: vwredsumu.vs v24, v22, v23, v0.t
vwredsumu.vs v24, v22, v23, v0.t

# CHECK: vwredsum.vs v27, v25, v26
vwredsum.vs v27, v25, v26
# CHECK: vwredsum.vs v30, v28, v29, v0.t
vwredsum.vs v30, v28, v29, v0.t

# CHECK: vdotu.vv v1, v31, v0
vdotu.vv v1, v31, v0
# CHECK: vdotu.vv v4, v2, v3, v0.t
vdotu.vv v4, v2, v3, v0.t

# CHECK: vdot.vv v7, v5, v6
vdot.vv v7, v5, v6
# CHECK: vdot.vv v10, v8, v9, v0.t
vdot.vv v10, v8, v9, v0.t

# CHECK: vwsmaccu.vv v13, v11, v12
vwsmaccu.vv v13, v11, v12
# CHECK: vwsmaccu.vv v16, v14, v15, v0.t
vwsmaccu.vv v16, v14, v15, v0.t

# CHECK: vwsmaccu.vx v18, v17, sp
vwsmaccu.vx v18, v17, sp
# CHECK: vwsmaccu.vx v20, v19, tp, v0.t
vwsmaccu.vx v20, v19, tp, v0.t

# CHECK: vwsmacc.vv v23, v21, v22
vwsmacc.vv v23, v21, v22
# CHECK: vwsmacc.vv v26, v24, v25, v0.t
vwsmacc.vv v26, v24, v25, v0.t

# CHECK: vwsmacc.vx v28, v27, t1
vwsmacc.vx v28, v27, t1
# CHECK: vwsmacc.vx v30, v29, s0, v0.t
vwsmacc.vx v30, v29, s0, v0.t

# CHECK: vwsmsacu.vv v1, v31, v0
vwsmsacu.vv v1, v31, v0
# CHECK: vwsmsacu.vv v4, v2, v3, v0.t
vwsmsacu.vv v4, v2, v3, v0.t

# CHECK: vwsmsacu.vx v6, v5, a0
vwsmsacu.vx v6, v5, a0
# CHECK: vwsmsacu.vx v8, v7, a2, v0.t
vwsmsacu.vx v8, v7, a2, v0.t

# CHECK: vwsmsac.vv v11, v9, v10
vwsmsac.vv v11, v9, v10
# CHECK: vwsmsac.vv v14, v12, v13, v0.t
vwsmsac.vv v14, v12, v13, v0.t

# CHECK: vwsmsac.vx v16, v15, a4
vwsmsac.vx v16, v15, a4
# CHECK: vwsmsac.vx v18, v17, a6, v0.t
vwsmsac.vx v18, v17, a6, v0.t

# CHECK: vredsum.vs v21, v19, v20
vredsum.vs v21, v19, v20
# CHECK: vredsum.vs v24, v22, v23, v0.t
vredsum.vs v24, v22, v23, v0.t

# CHECK: vredand.vs v27, v25, v26
vredand.vs v27, v25, v26
# CHECK: vredand.vs v30, v28, v29, v0.t
vredand.vs v30, v28, v29, v0.t

# CHECK: vredor.vs v1, v31, v0
vredor.vs v1, v31, v0
# CHECK: vredor.vs v4, v2, v3, v0.t
vredor.vs v4, v2, v3, v0.t

# CHECK: vredxor.vs v7, v5, v6
vredxor.vs v7, v5, v6
# CHECK: vredxor.vs v10, v8, v9, v0.t
vredxor.vs v10, v8, v9, v0.t

# CHECK: vredminu.vs v13, v11, v12
vredminu.vs v13, v11, v12
# CHECK: vredminu.vs v16, v14, v15, v0.t
vredminu.vs v16, v14, v15, v0.t

# CHECK: vredmin.vs v19, v17, v18
vredmin.vs v19, v17, v18
# CHECK: vredmin.vs v22, v20, v21, v0.t
vredmin.vs v22, v20, v21, v0.t

# CHECK: vredmaxu.vs v25, v23, v24
vredmaxu.vs v25, v23, v24
# CHECK: vredmaxu.vs v28, v26, v27, v0.t
vredmaxu.vs v28, v26, v27, v0.t

# CHECK: vredmax.vs v31, v29, v30
vredmax.vs v31, v29, v30
# CHECK: vredmax.vs v2, v0, v1, v0.t
vredmax.vs v2, v0, v1, v0.t

# CHECK: vext.x.v s4, v3, s2
vext.x.v s4, v3, s2

# CHECK: vmv.s.x v4, s6
vmv.s.x v4, s6

# CHECK: vslide1up.vx v6, v5, s8
vslide1up.vx v6, v5, s8
# CHECK: vslide1up.vx v8, v7, s10, v0.t
vslide1up.vx v8, v7, s10, v0.t

# CHECK: vslide1down.vx v10, v9, t3
vslide1down.vx v10, v9, t3
# CHECK: vslide1down.vx v12, v11, t5, v0.t
vslide1down.vx v12, v11, t5, v0.t

# CHECK: vmpopc.m ra, v13
vmpopc.m ra, v13
# CHECK: vmpopc.m gp, v14, v0.t
vmpopc.m gp, v14, v0.t

# CHECK: vmfirst.m t0, v15
vmfirst.m t0, v15
# CHECK: vmfirst.m t2, v16, v0.t
vmfirst.m t2, v16, v0.t

# CHECK: vcompress.vm v19, v17, v18
vcompress.vm v19, v17, v18

# CHECK: vmandnot.mm v22, v20, v21
vmandnot.mm v22, v20, v21

# CHECK: vmand.mm v25, v23, v24
vmand.mm v25, v23, v24

# CHECK: vmor.mm v28, v26, v27
vmor.mm v28, v26, v27

# CHECK: vmxor.mm v31, v29, v30
vmxor.mm v31, v29, v30

# CHECK: vmornot.mm v2, v0, v1
vmornot.mm v2, v0, v1

# CHECK: vmnand.mm v5, v3, v4
vmnand.mm v5, v3, v4

# CHECK: vmnor.mm v8, v6, v7
vmnor.mm v8, v6, v7

# CHECK: vmxnor.mm v11, v9, v10
vmxnor.mm v11, v9, v10

# CHECK: vdivu.vv v14, v12, v13
vdivu.vv v14, v12, v13
# CHECK: vdivu.vv v17, v15, v16, v0.t
vdivu.vv v17, v15, v16, v0.t

# CHECK: vdivu.vx v19, v18, s1
vdivu.vx v19, v18, s1
# CHECK: vdivu.vx v21, v20, a1, v0.t
vdivu.vx v21, v20, a1, v0.t

# CHECK: vdiv.vv v24, v22, v23
vdiv.vv v24, v22, v23
# CHECK: vdiv.vv v27, v25, v26, v0.t
vdiv.vv v27, v25, v26, v0.t

# CHECK: vdiv.vx v29, v28, a3
vdiv.vx v29, v28, a3
# CHECK: vdiv.vx v31, v30, a5, v0.t
vdiv.vx v31, v30, a5, v0.t

# CHECK: vremu.vv v2, v0, v1
vremu.vv v2, v0, v1
# CHECK: vremu.vv v5, v3, v4, v0.t
vremu.vv v5, v3, v4, v0.t

# CHECK: vremu.vx v7, v6, a7
vremu.vx v7, v6, a7
# CHECK: vremu.vx v9, v8, s3, v0.t
vremu.vx v9, v8, s3, v0.t

# CHECK: vrem.vv v12, v10, v11
vrem.vv v12, v10, v11
# CHECK: vrem.vv v15, v13, v14, v0.t
vrem.vv v15, v13, v14, v0.t

# CHECK: vrem.vx v17, v16, s5
vrem.vx v17, v16, s5
# CHECK: vrem.vx v19, v18, s7, v0.t
vrem.vx v19, v18, s7, v0.t

# CHECK: vmulhu.vv v22, v20, v21
vmulhu.vv v22, v20, v21
# CHECK: vmulhu.vv v25, v23, v24, v0.t
vmulhu.vv v25, v23, v24, v0.t

# CHECK: vmulhu.vx v27, v26, s9
vmulhu.vx v27, v26, s9
# CHECK: vmulhu.vx v29, v28, s11, v0.t
vmulhu.vx v29, v28, s11, v0.t

# CHECK: vmul.vv v0, v30, v31
vmul.vv v0, v30, v31
# CHECK: vmul.vv v3, v1, v2, v0.t
vmul.vv v3, v1, v2, v0.t

# CHECK: vmul.vx v5, v4, t4
vmul.vx v5, v4, t4
# CHECK: vmul.vx v7, v6, t6, v0.t
vmul.vx v7, v6, t6, v0.t

# CHECK: vmulhsu.vv v10, v8, v9
vmulhsu.vv v10, v8, v9
# CHECK: vmulhsu.vv v13, v11, v12, v0.t
vmulhsu.vv v13, v11, v12, v0.t

# CHECK: vmulhsu.vx v15, v14, sp
vmulhsu.vx v15, v14, sp
# CHECK: vmulhsu.vx v17, v16, tp, v0.t
vmulhsu.vx v17, v16, tp, v0.t

# CHECK: vmulh.vv v20, v18, v19
vmulh.vv v20, v18, v19
# CHECK: vmulh.vv v23, v21, v22, v0.t
vmulh.vv v23, v21, v22, v0.t

# CHECK: vmulh.vx v25, v24, t1
vmulh.vx v25, v24, t1
# CHECK: vmulh.vx v27, v26, s0, v0.t
vmulh.vx v27, v26, s0, v0.t

# CHECK: vmadd.vv v30, v29, v28
vmadd.vv v30, v29, v28
# CHECK: vmadd.vv v1, v0, v31, v0.t
vmadd.vv v1, v0, v31, v0.t

# CHECK: vmadd.vx v3, a0, v2
vmadd.vx v3, a0, v2
# CHECK: vmadd.vx v5, a2, v4, v0.t
vmadd.vx v5, a2, v4, v0.t

# CHECK: vmsub.vv v8, v7, v6
vmsub.vv v8, v7, v6
# CHECK: vmsub.vv v11, v10, v9, v0.t
vmsub.vv v11, v10, v9, v0.t

# CHECK: vmsub.vx v13, a4, v12
vmsub.vx v13, a4, v12
# CHECK: vmsub.vx v15, a6, v14, v0.t
vmsub.vx v15, a6, v14, v0.t

# CHECK: vmacc.vv v18, v17, v16
vmacc.vv v18, v17, v16
# CHECK: vmacc.vv v21, v20, v19, v0.t
vmacc.vv v21, v20, v19, v0.t

# CHECK: vmacc.vx v23, s2, v22
vmacc.vx v23, s2, v22
# CHECK: vmacc.vx v25, s4, v24, v0.t
vmacc.vx v25, s4, v24, v0.t

# CHECK: vmsac.vv v28, v27, v26
vmsac.vv v28, v27, v26
# CHECK: vmsac.vv v31, v30, v29, v0.t
vmsac.vv v31, v30, v29, v0.t

# CHECK: vmsac.vx v1, s6, v0
vmsac.vx v1, s6, v0
# CHECK: vmsac.vx v3, s8, v2, v0.t
vmsac.vx v3, s8, v2, v0.t

# CHECK: vwaddu.vv v6, v4, v5
vwaddu.vv v6, v4, v5
# CHECK: vwaddu.vv v9, v7, v8, v0.t
vwaddu.vv v9, v7, v8, v0.t

# CHECK: vwaddu.vx v11, v10, s10
vwaddu.vx v11, v10, s10
# CHECK: vwaddu.vx v13, v12, t3, v0.t
vwaddu.vx v13, v12, t3, v0.t

# CHECK: vwadd.vv v16, v14, v15
vwadd.vv v16, v14, v15
# CHECK: vwadd.vv v19, v17, v18, v0.t
vwadd.vv v19, v17, v18, v0.t

# CHECK: vwadd.vx v21, v20, t5
vwadd.vx v21, v20, t5
# CHECK: vwadd.vx v23, v22, ra, v0.t
vwadd.vx v23, v22, ra, v0.t

# CHECK: vwsubu.vv v26, v24, v25
vwsubu.vv v26, v24, v25
# CHECK: vwsubu.vv v29, v27, v28, v0.t
vwsubu.vv v29, v27, v28, v0.t

# CHECK: vwsubu.vx v31, v30, gp
vwsubu.vx v31, v30, gp
# CHECK: vwsubu.vx v1, v0, t0, v0.t
vwsubu.vx v1, v0, t0, v0.t

# CHECK: vwsub.vv v4, v2, v3
vwsub.vv v4, v2, v3
# CHECK: vwsub.vv v7, v5, v6, v0.t
vwsub.vv v7, v5, v6, v0.t

# CHECK: vwsub.vx v9, v8, t2
vwsub.vx v9, v8, t2
# CHECK: vwsub.vx v11, v10, s1, v0.t
vwsub.vx v11, v10, s1, v0.t

# CHECK: vwaddu.wv v14, v12, v13
vwaddu.wv v14, v12, v13
# CHECK: vwaddu.wv v17, v15, v16, v0.t
vwaddu.wv v17, v15, v16, v0.t

# CHECK: vwaddu.wx v19, v18, a1
vwaddu.wx v19, v18, a1
# CHECK: vwaddu.wx v21, v20, a3, v0.t
vwaddu.wx v21, v20, a3, v0.t

# CHECK: vwadd.wv v24, v22, v23
vwadd.wv v24, v22, v23
# CHECK: vwadd.wv v27, v25, v26, v0.t
vwadd.wv v27, v25, v26, v0.t

# CHECK: vwadd.wx v29, v28, a5
vwadd.wx v29, v28, a5
# CHECK: vwadd.wx v31, v30, a7, v0.t
vwadd.wx v31, v30, a7, v0.t

# CHECK: vwsubu.wv v2, v0, v1
vwsubu.wv v2, v0, v1
# CHECK: vwsubu.wv v5, v3, v4, v0.t
vwsubu.wv v5, v3, v4, v0.t

# CHECK: vwsubu.wx v7, v6, s3
vwsubu.wx v7, v6, s3
# CHECK: vwsubu.wx v9, v8, s5, v0.t
vwsubu.wx v9, v8, s5, v0.t

# CHECK: vwsub.wv v12, v10, v11
vwsub.wv v12, v10, v11
# CHECK: vwsub.wv v15, v13, v14, v0.t
vwsub.wv v15, v13, v14, v0.t

# CHECK: vwsub.wx v17, v16, s7
vwsub.wx v17, v16, s7
# CHECK: vwsub.wx v19, v18, s9, v0.t
vwsub.wx v19, v18, s9, v0.t

# CHECK: vwmulu.vv v22, v20, v21
vwmulu.vv v22, v20, v21
# CHECK: vwmulu.vv v25, v23, v24, v0.t
vwmulu.vv v25, v23, v24, v0.t

# CHECK: vwmulu.vx v27, v26, s11
vwmulu.vx v27, v26, s11
# CHECK: vwmulu.vx v29, v28, t4, v0.t
vwmulu.vx v29, v28, t4, v0.t

# CHECK: vwmulsu.vv v0, v30, v31
vwmulsu.vv v0, v30, v31
# CHECK: vwmulsu.vv v3, v1, v2, v0.t
vwmulsu.vv v3, v1, v2, v0.t

# CHECK: vwmulsu.vx v5, v4, t6
vwmulsu.vx v5, v4, t6
# CHECK: vwmulsu.vx v7, v6, sp, v0.t
vwmulsu.vx v7, v6, sp, v0.t

# CHECK: vwmul.vv v10, v8, v9
vwmul.vv v10, v8, v9
# CHECK: vwmul.vv v13, v11, v12, v0.t
vwmul.vv v13, v11, v12, v0.t

# CHECK: vwmul.vx v15, v14, tp
vwmul.vx v15, v14, tp
# CHECK: vwmul.vx v17, v16, t1, v0.t
vwmul.vx v17, v16, t1, v0.t

# CHECK: vwmaccu.vv v20, v18, v19
vwmaccu.vv v20, v18, v19
# CHECK: vwmaccu.vv v23, v21, v22, v0.t
vwmaccu.vv v23, v21, v22, v0.t

# CHECK: vwmaccu.vx v25, v24, s0
vwmaccu.vx v25, v24, s0
# CHECK: vwmaccu.vx v27, v26, a0, v0.t
vwmaccu.vx v27, v26, a0, v0.t

# CHECK: vwmacc.vv v30, v28, v29
vwmacc.vv v30, v28, v29
# CHECK: vwmacc.vv v1, v31, v0, v0.t
vwmacc.vv v1, v31, v0, v0.t

# CHECK: vwmacc.vx v3, v2, a2
vwmacc.vx v3, v2, a2
# CHECK: vwmacc.vx v5, v4, a4, v0.t
vwmacc.vx v5, v4, a4, v0.t

# CHECK: vwmsacu.vv v8, v6, v7
vwmsacu.vv v8, v6, v7
# CHECK: vwmsacu.vv v11, v9, v10, v0.t
vwmsacu.vv v11, v9, v10, v0.t

# CHECK: vwmsacu.vx v13, v12, a6
vwmsacu.vx v13, v12, a6
# CHECK: vwmsacu.vx v15, v14, s2, v0.t
vwmsacu.vx v15, v14, s2, v0.t

# CHECK: vwmsac.vv v18, v16, v17
vwmsac.vv v18, v16, v17
# CHECK: vwmsac.vv v21, v19, v20, v0.t
vwmsac.vv v21, v19, v20, v0.t

# CHECK: vwmsac.vx v23, v22, s4
vwmsac.vx v23, v22, s4
# CHECK: vwmsac.vx v25, v24, s6, v0.t
vwmsac.vx v25, v24, s6, v0.t

# CHECK: vfadd.vv v28, v26, v27
vfadd.vv v28, v26, v27
# CHECK: vfadd.vv v31, v29, v30, v0.t
vfadd.vv v31, v29, v30, v0.t

# CHECK: vfadd.vf v1, v0, ft0
vfadd.vf v1, v0, ft0
# CHECK: vfadd.vf v3, v2, ft1, v0.t
vfadd.vf v3, v2, ft1, v0.t

# CHECK: vfredsum.vs v6, v4, v5
vfredsum.vs v6, v4, v5
# CHECK: vfredsum.vs v9, v7, v8, v0.t
vfredsum.vs v9, v7, v8, v0.t

# CHECK: vfsub.vv v12, v10, v11
vfsub.vv v12, v10, v11
# CHECK: vfsub.vv v15, v13, v14, v0.t
vfsub.vv v15, v13, v14, v0.t

# CHECK: vfsub.vf v17, v16, ft2
vfsub.vf v17, v16, ft2
# CHECK: vfsub.vf v19, v18, ft3, v0.t
vfsub.vf v19, v18, ft3, v0.t

# CHECK: vfredosum.vs v22, v20, v21
vfredosum.vs v22, v20, v21
# CHECK: vfredosum.vs v25, v23, v24, v0.t
vfredosum.vs v25, v23, v24, v0.t

# CHECK: vfmin.vv v28, v26, v27
vfmin.vv v28, v26, v27
# CHECK: vfmin.vv v31, v29, v30, v0.t
vfmin.vv v31, v29, v30, v0.t

# CHECK: vfmin.vf v1, v0, ft4
vfmin.vf v1, v0, ft4
# CHECK: vfmin.vf v3, v2, ft5, v0.t
vfmin.vf v3, v2, ft5, v0.t

# CHECK: vfredmin.vs v6, v4, v5
vfredmin.vs v6, v4, v5
# CHECK: vfredmin.vs v9, v7, v8, v0.t
vfredmin.vs v9, v7, v8, v0.t

# CHECK: vfmax.vv v12, v10, v11
vfmax.vv v12, v10, v11
# CHECK: vfmax.vv v15, v13, v14, v0.t
vfmax.vv v15, v13, v14, v0.t

# CHECK: vfmax.vf v17, v16, ft6
vfmax.vf v17, v16, ft6
# CHECK: vfmax.vf v19, v18, ft7, v0.t
vfmax.vf v19, v18, ft7, v0.t

# CHECK: vfredmax.vs v22, v20, v21
vfredmax.vs v22, v20, v21
# CHECK: vfredmax.vs v25, v23, v24, v0.t
vfredmax.vs v25, v23, v24, v0.t

# CHECK: vfsgnj.vv v28, v26, v27
vfsgnj.vv v28, v26, v27
# CHECK: vfsgnj.vv v31, v29, v30, v0.t
vfsgnj.vv v31, v29, v30, v0.t

# CHECK: vfsgnj.vf v1, v0, fs0
vfsgnj.vf v1, v0, fs0
# CHECK: vfsgnj.vf v3, v2, fs1, v0.t
vfsgnj.vf v3, v2, fs1, v0.t

# CHECK: vfsgnjn.vv v6, v4, v5
vfsgnjn.vv v6, v4, v5
# CHECK: vfsgnjn.vv v9, v7, v8, v0.t
vfsgnjn.vv v9, v7, v8, v0.t

# CHECK: vfsgnjn.vf v11, v10, fa0
vfsgnjn.vf v11, v10, fa0
# CHECK: vfsgnjn.vf v13, v12, fa1, v0.t
vfsgnjn.vf v13, v12, fa1, v0.t

# CHECK: vfsgnjx.vv v16, v14, v15
vfsgnjx.vv v16, v14, v15
# CHECK: vfsgnjx.vv v19, v17, v18, v0.t
vfsgnjx.vv v19, v17, v18, v0.t

# CHECK: vfsgnjx.vf v21, v20, fa2
vfsgnjx.vf v21, v20, fa2
# CHECK: vfsgnjx.vf v23, v22, fa3, v0.t
vfsgnjx.vf v23, v22, fa3, v0.t

# CHECK: vfmv.f.s fa4, v24
vfmv.f.s fa4, v24

# CHECK: vfmv.s.f v25, fa5
vfmv.s.f v25, fa5

# CHECK: vfmv.v.f v26, fa6
vfmv.v.f v26, fa6

# CHECK: vfmerge.vfm v28, v27, fa7, v0
vfmerge.vfm v28, v27, fa7, v0

# CHECK: vmfeq.vv v31, v29, v30
vmfeq.vv v31, v29, v30
# CHECK: vmfeq.vv v2, v0, v1, v0.t
vmfeq.vv v2, v0, v1, v0.t

# CHECK: vmfeq.vf v4, v3, fs2
vmfeq.vf v4, v3, fs2
# CHECK: vmfeq.vf v6, v5, fs3, v0.t
vmfeq.vf v6, v5, fs3, v0.t

# CHECK: vmfle.vv v9, v7, v8
vmfle.vv v9, v7, v8
# CHECK: vmfle.vv v12, v10, v11, v0.t
vmfle.vv v12, v10, v11, v0.t

# CHECK: vmfle.vf v14, v13, fs4
vmfle.vf v14, v13, fs4
# CHECK: vmfle.vf v16, v15, fs5, v0.t
vmfle.vf v16, v15, fs5, v0.t

# CHECK: vmford.vv v19, v17, v18
vmford.vv v19, v17, v18
# CHECK: vmford.vv v22, v20, v21, v0.t
vmford.vv v22, v20, v21, v0.t

# CHECK: vmford.vf v24, v23, fs6
vmford.vf v24, v23, fs6
# CHECK: vmford.vf v26, v25, fs7, v0.t
vmford.vf v26, v25, fs7, v0.t

# CHECK: vmflt.vv v29, v27, v28
vmflt.vv v29, v27, v28
# CHECK: vmflt.vv v0, v30, v31, v0.t
vmflt.vv v0, v30, v31, v0.t

# CHECK: vmflt.vf v2, v1, fs8
vmflt.vf v2, v1, fs8
# CHECK: vmflt.vf v4, v3, fs9, v0.t
vmflt.vf v4, v3, fs9, v0.t

# CHECK: vmfne.vv v7, v5, v6
vmfne.vv v7, v5, v6
# CHECK: vmfne.vv v10, v8, v9, v0.t
vmfne.vv v10, v8, v9, v0.t

# CHECK: vmfne.vf v12, v11, fs10
vmfne.vf v12, v11, fs10
# CHECK: vmfne.vf v14, v13, fs11, v0.t
vmfne.vf v14, v13, fs11, v0.t

# CHECK: vmfgt.vf v16, v15, ft8
vmfgt.vf v16, v15, ft8
# CHECK: vmfgt.vf v18, v17, ft9, v0.t
vmfgt.vf v18, v17, ft9, v0.t

# CHECK: vmfge.vf v20, v19, ft10
vmfge.vf v20, v19, ft10
# CHECK: vmfge.vf v22, v21, ft0, v0.t
vmfge.vf v22, v21, ft0, v0.t

# CHECK: vfdiv.vv v25, v23, v24
vfdiv.vv v25, v23, v24
# CHECK: vfdiv.vv v28, v26, v27, v0.t
vfdiv.vv v28, v26, v27, v0.t

# CHECK: vfdiv.vf v30, v29, ft1
vfdiv.vf v30, v29, ft1
# CHECK: vfdiv.vf v0, v31, ft2, v0.t
vfdiv.vf v0, v31, ft2, v0.t

# CHECK: vfrdiv.vf v2, v1, ft3
vfrdiv.vf v2, v1, ft3
# CHECK: vfrdiv.vf v4, v3, ft4, v0.t
vfrdiv.vf v4, v3, ft4, v0.t

# CHECK: vfmul.vv v7, v5, v6
vfmul.vv v7, v5, v6
# CHECK: vfmul.vv v10, v8, v9, v0.t
vfmul.vv v10, v8, v9, v0.t

# CHECK: vfmul.vf v12, v11, ft5
vfmul.vf v12, v11, ft5
# CHECK: vfmul.vf v14, v13, ft6, v0.t
vfmul.vf v14, v13, ft6, v0.t

# CHECK: vfmadd.vv v17, v16, v15
vfmadd.vv v17, v16, v15
# CHECK: vfmadd.vv v20, v19, v18, v0.t
vfmadd.vv v20, v19, v18, v0.t

# CHECK: vfmadd.vf v22, ft7, v21
vfmadd.vf v22, ft7, v21
# CHECK: vfmadd.vf v24, fs0, v23, v0.t
vfmadd.vf v24, fs0, v23, v0.t

# CHECK: vfnmadd.vv v27, v26, v25
vfnmadd.vv v27, v26, v25
# CHECK: vfnmadd.vv v30, v29, v28, v0.t
vfnmadd.vv v30, v29, v28, v0.t

# CHECK: vfnmadd.vf v0, fs1, v31
vfnmadd.vf v0, fs1, v31
# CHECK: vfnmadd.vf v2, fa0, v1, v0.t
vfnmadd.vf v2, fa0, v1, v0.t

# CHECK: vfmsub.vv v5, v4, v3
vfmsub.vv v5, v4, v3
# CHECK: vfmsub.vv v8, v7, v6, v0.t
vfmsub.vv v8, v7, v6, v0.t

# CHECK: vfmsub.vf v10, fa1, v9
vfmsub.vf v10, fa1, v9
# CHECK: vfmsub.vf v12, fa2, v11, v0.t
vfmsub.vf v12, fa2, v11, v0.t

# CHECK: vfnmsub.vv v15, v14, v13
vfnmsub.vv v15, v14, v13
# CHECK: vfnmsub.vv v18, v17, v16, v0.t
vfnmsub.vv v18, v17, v16, v0.t

# CHECK: vfnmsub.vf v20, fa3, v19
vfnmsub.vf v20, fa3, v19
# CHECK: vfnmsub.vf v22, fa4, v21, v0.t
vfnmsub.vf v22, fa4, v21, v0.t

# CHECK: vfmacc.vv v25, v24, v23
vfmacc.vv v25, v24, v23
# CHECK: vfmacc.vv v28, v27, v26, v0.t
vfmacc.vv v28, v27, v26, v0.t

# CHECK: vfmacc.vf v30, fa5, v29
vfmacc.vf v30, fa5, v29
# CHECK: vfmacc.vf v0, fa6, v31, v0.t
vfmacc.vf v0, fa6, v31, v0.t

# CHECK: vfnmacc.vv v3, v2, v1
vfnmacc.vv v3, v2, v1
# CHECK: vfnmacc.vv v6, v5, v4, v0.t
vfnmacc.vv v6, v5, v4, v0.t

# CHECK: vfnmacc.vf v8, fa7, v7
vfnmacc.vf v8, fa7, v7
# CHECK: vfnmacc.vf v10, fs2, v9, v0.t
vfnmacc.vf v10, fs2, v9, v0.t

# CHECK: vfmsac.vv v13, v12, v11
vfmsac.vv v13, v12, v11
# CHECK: vfmsac.vv v16, v15, v14, v0.t
vfmsac.vv v16, v15, v14, v0.t

# CHECK: vfmsac.vf v18, fs3, v17
vfmsac.vf v18, fs3, v17
# CHECK: vfmsac.vf v20, fs4, v19, v0.t
vfmsac.vf v20, fs4, v19, v0.t

# CHECK: vfnmsac.vv v23, v22, v21
vfnmsac.vv v23, v22, v21
# CHECK: vfnmsac.vv v26, v25, v24, v0.t
vfnmsac.vv v26, v25, v24, v0.t

# CHECK: vfnmsac.vf v28, fs5, v27
vfnmsac.vf v28, fs5, v27
# CHECK: vfnmsac.vf v30, fs6, v29, v0.t
vfnmsac.vf v30, fs6, v29, v0.t

# CHECK: vfwadd.vv v1, v31, v0
vfwadd.vv v1, v31, v0
# CHECK: vfwadd.vv v4, v2, v3, v0.t
vfwadd.vv v4, v2, v3, v0.t

# CHECK: vfwadd.vf v6, v5, fs7
vfwadd.vf v6, v5, fs7
# CHECK: vfwadd.vf v8, v7, fs8, v0.t
vfwadd.vf v8, v7, fs8, v0.t

# CHECK: vfwredsum.vs v11, v9, v10
vfwredsum.vs v11, v9, v10
# CHECK: vfwredsum.vs v14, v12, v13, v0.t
vfwredsum.vs v14, v12, v13, v0.t

# CHECK: vfwsub.vv v17, v15, v16
vfwsub.vv v17, v15, v16
# CHECK: vfwsub.vv v20, v18, v19, v0.t
vfwsub.vv v20, v18, v19, v0.t

# CHECK: vfwsub.vf v22, v21, fs9
vfwsub.vf v22, v21, fs9
# CHECK: vfwsub.vf v24, v23, fs10, v0.t
vfwsub.vf v24, v23, fs10, v0.t

# CHECK: vfwredosum.vs v27, v25, v26
vfwredosum.vs v27, v25, v26
# CHECK: vfwredosum.vs v30, v28, v29, v0.t
vfwredosum.vs v30, v28, v29, v0.t

# CHECK: vfwadd.wv v1, v31, v0
vfwadd.wv v1, v31, v0
# CHECK: vfwadd.wv v4, v2, v3, v0.t
vfwadd.wv v4, v2, v3, v0.t

# CHECK: vfwadd.wf v6, v5, fs11
vfwadd.wf v6, v5, fs11
# CHECK: vfwadd.wf v8, v7, ft8, v0.t
vfwadd.wf v8, v7, ft8, v0.t

# CHECK: vfwsub.wv v11, v9, v10
vfwsub.wv v11, v9, v10
# CHECK: vfwsub.wv v14, v12, v13, v0.t
vfwsub.wv v14, v12, v13, v0.t

# CHECK: vfwsub.wf v16, v15, ft9
vfwsub.wf v16, v15, ft9
# CHECK: vfwsub.wf v18, v17, ft10, v0.t
vfwsub.wf v18, v17, ft10, v0.t

# CHECK: vfwmul.vv v21, v19, v20
vfwmul.vv v21, v19, v20
# CHECK: vfwmul.vv v24, v22, v23, v0.t
vfwmul.vv v24, v22, v23, v0.t

# CHECK: vfwmul.vf v26, v25, ft0
vfwmul.vf v26, v25, ft0
# CHECK: vfwmul.vf v28, v27, ft1, v0.t
vfwmul.vf v28, v27, ft1, v0.t

# CHECK: vfdot.vv v31, v29, v30
vfdot.vv v31, v29, v30
# CHECK: vfdot.vv v2, v0, v1, v0.t
vfdot.vv v2, v0, v1, v0.t

# CHECK: vfwmacc.vv v5, v4, v3
vfwmacc.vv v5, v4, v3
# CHECK: vfwmacc.vv v8, v7, v6, v0.t
vfwmacc.vv v8, v7, v6, v0.t

# CHECK: vfwmacc.vf v10, ft2, v9
vfwmacc.vf v10, ft2, v9
# CHECK: vfwmacc.vf v12, ft3, v11, v0.t
vfwmacc.vf v12, ft3, v11, v0.t

# CHECK: vfwnmacc.vv v15, v14, v13
vfwnmacc.vv v15, v14, v13
# CHECK: vfwnmacc.vv v18, v17, v16, v0.t
vfwnmacc.vv v18, v17, v16, v0.t

# CHECK: vfwnmacc.vf v20, ft4, v19
vfwnmacc.vf v20, ft4, v19
# CHECK: vfwnmacc.vf v22, ft5, v21, v0.t
vfwnmacc.vf v22, ft5, v21, v0.t

# CHECK: vfwmsac.vv v25, v24, v23
vfwmsac.vv v25, v24, v23
# CHECK: vfwmsac.vv v28, v27, v26, v0.t
vfwmsac.vv v28, v27, v26, v0.t

# CHECK: vfwmsac.vf v30, ft6, v29
vfwmsac.vf v30, ft6, v29
# CHECK: vfwmsac.vf v0, ft7, v31, v0.t
vfwmsac.vf v0, ft7, v31, v0.t

# CHECK: vfwnmsac.vv v3, v2, v1
vfwnmsac.vv v3, v2, v1
# CHECK: vfwnmsac.vv v6, v5, v4, v0.t
vfwnmsac.vv v6, v5, v4, v0.t

# CHECK: vfwnmsac.vf v8, fs0, v7
vfwnmsac.vf v8, fs0, v7
# CHECK: vfwnmsac.vf v10, fs1, v9, v0.t
vfwnmsac.vf v10, fs1, v9, v0.t

# CHECK: vfsqrt.v v12, v11
vfsqrt.v v12, v11
# CHECK: vfsqrt.v v14, v13, v0.t
vfsqrt.v v14, v13, v0.t

# CHECK: vfclass.v v16, v15
vfclass.v v16, v15
# CHECK: vfclass.v v18, v17, v0.t
vfclass.v v18, v17, v0.t

# CHECK: vfcvt.xu.f.v v20, v19
vfcvt.xu.f.v v20, v19
# CHECK: vfcvt.xu.f.v v22, v21, v0.t
vfcvt.xu.f.v v22, v21, v0.t

# CHECK: vfcvt.x.f.v v24, v23
vfcvt.x.f.v v24, v23
# CHECK: vfcvt.x.f.v v26, v25, v0.t
vfcvt.x.f.v v26, v25, v0.t

# CHECK: vfcvt.f.xu.v v28, v27
vfcvt.f.xu.v v28, v27
# CHECK: vfcvt.f.xu.v v30, v29, v0.t
vfcvt.f.xu.v v30, v29, v0.t

# CHECK: vfcvt.f.x.v v0, v31
vfcvt.f.x.v v0, v31
# CHECK: vfcvt.f.x.v v2, v1, v0.t
vfcvt.f.x.v v2, v1, v0.t

# CHECK: vfwcvt.xu.f.v v4, v3
vfwcvt.xu.f.v v4, v3
# CHECK: vfwcvt.xu.f.v v6, v5, v0.t
vfwcvt.xu.f.v v6, v5, v0.t

# CHECK: vfwcvt.x.f.v v8, v7
vfwcvt.x.f.v v8, v7
# CHECK: vfwcvt.x.f.v v10, v9, v0.t
vfwcvt.x.f.v v10, v9, v0.t

# CHECK: vfwcvt.f.xu.v v12, v11
vfwcvt.f.xu.v v12, v11
# CHECK: vfwcvt.f.xu.v v14, v13, v0.t
vfwcvt.f.xu.v v14, v13, v0.t

# CHECK: vfwcvt.f.x.v v16, v15
vfwcvt.f.x.v v16, v15
# CHECK: vfwcvt.f.x.v v18, v17, v0.t
vfwcvt.f.x.v v18, v17, v0.t

# CHECK: vfwcvt.f.f.v v20, v19
vfwcvt.f.f.v v20, v19
# CHECK: vfwcvt.f.f.v v22, v21, v0.t
vfwcvt.f.f.v v22, v21, v0.t

# CHECK: vfncvt.xu.f.v v24, v23
vfncvt.xu.f.v v24, v23
# CHECK: vfncvt.xu.f.v v26, v25, v0.t
vfncvt.xu.f.v v26, v25, v0.t

# CHECK: vfncvt.x.f.v v28, v27
vfncvt.x.f.v v28, v27
# CHECK: vfncvt.x.f.v v30, v29, v0.t
vfncvt.x.f.v v30, v29, v0.t

# CHECK: vfncvt.f.xu.v v0, v31
vfncvt.f.xu.v v0, v31
# CHECK: vfncvt.f.xu.v v2, v1, v0.t
vfncvt.f.xu.v v2, v1, v0.t

# CHECK: vfncvt.f.x.v v4, v3
vfncvt.f.x.v v4, v3
# CHECK: vfncvt.f.x.v v6, v5, v0.t
vfncvt.f.x.v v6, v5, v0.t

# CHECK: vfncvt.f.f.v v8, v7
vfncvt.f.f.v v8, v7
# CHECK: vfncvt.f.f.v v10, v9, v0.t
vfncvt.f.f.v v10, v9, v0.t

# CHECK: vmsbf.m v12, v11
vmsbf.m v12, v11
# CHECK: vmsbf.m v14, v13, v0.t
vmsbf.m v14, v13, v0.t

# CHECK: vmsof.m v16, v15
vmsof.m v16, v15
# CHECK: vmsof.m v18, v17, v0.t
vmsof.m v18, v17, v0.t

# CHECK: vmsif.m v20, v19
vmsif.m v20, v19
# CHECK: vmsif.m v22, v21, v0.t
vmsif.m v22, v21, v0.t

# CHECK: viota.m v24, v23
viota.m v24, v23
# CHECK: viota.m v26, v25, v0.t
viota.m v26, v25, v0.t

# CHECK: vid.v v27
vid.v v27
# CHECK: vid.v v28, v0.t
vid.v v28, v0.t

# CHECK: vlb.v v29, (s8)
vlb.v v29, (s8)
# CHECK: vlb.v v30, (s10), v0.t
vlb.v v30, (s10), v0.t

# CHECK: vlh.v v31, (t3)
vlh.v v31, (t3)
# CHECK: vlh.v v0, (t5), v0.t
vlh.v v0, (t5), v0.t

# CHECK: vlw.v v1, (ra)
vlw.v v1, (ra)
# CHECK: vlw.v v2, (gp), v0.t
vlw.v v2, (gp), v0.t

# CHECK: vlbu.v v3, (t0)
vlbu.v v3, (t0)
# CHECK: vlbu.v v4, (t2), v0.t
vlbu.v v4, (t2), v0.t

# CHECK: vlhu.v v5, (s1)
vlhu.v v5, (s1)
# CHECK: vlhu.v v6, (a1), v0.t
vlhu.v v6, (a1), v0.t

# CHECK: vlwu.v v7, (a3)
vlwu.v v7, (a3)
# CHECK: vlwu.v v8, (a5), v0.t
vlwu.v v8, (a5), v0.t

# CHECK: vle.v v9, (a7)
vle.v v9, (a7)
# CHECK: vle.v v10, (s3), v0.t
vle.v v10, (s3), v0.t

# CHECK: vsb.v v11, (s5)
vsb.v v11, (s5)
# CHECK: vsb.v v12, (s7), v0.t
vsb.v v12, (s7), v0.t

# CHECK: vsh.v v13, (s9)
vsh.v v13, (s9)
# CHECK: vsh.v v14, (s11), v0.t
vsh.v v14, (s11), v0.t

# CHECK: vsw.v v15, (t4)
vsw.v v15, (t4)
# CHECK: vsw.v v16, (t6), v0.t
vsw.v v16, (t6), v0.t

# CHECK: vse.v v17, (sp)
vse.v v17, (sp)
# CHECK: vse.v v18, (tp), v0.t
vse.v v18, (tp), v0.t

# CHECK: vlsb.v v19, (s0), t1
vlsb.v v19, (s0), t1
# CHECK: vlsb.v v20, (a2), a0, v0.t
vlsb.v v20, (a2), a0, v0.t

# CHECK: vlsh.v v21, (a6), a4
vlsh.v v21, (a6), a4
# CHECK: vlsh.v v22, (s4), s2, v0.t
vlsh.v v22, (s4), s2, v0.t

# CHECK: vlsw.v v23, (s8), s6
vlsw.v v23, (s8), s6
# CHECK: vlsw.v v24, (t3), s10, v0.t
vlsw.v v24, (t3), s10, v0.t

# CHECK: vlsbu.v v25, (ra), t5
vlsbu.v v25, (ra), t5
# CHECK: vlsbu.v v26, (t0), gp, v0.t
vlsbu.v v26, (t0), gp, v0.t

# CHECK: vlshu.v v27, (s1), t2
vlshu.v v27, (s1), t2
# CHECK: vlshu.v v28, (a3), a1, v0.t
vlshu.v v28, (a3), a1, v0.t

# CHECK: vlswu.v v29, (a7), a5
vlswu.v v29, (a7), a5
# CHECK: vlswu.v v30, (s5), s3, v0.t
vlswu.v v30, (s5), s3, v0.t

# CHECK: vlse.v v31, (s9), s7
vlse.v v31, (s9), s7
# CHECK: vlse.v v0, (t4), s11, v0.t
vlse.v v0, (t4), s11, v0.t

# CHECK: vssb.v v1, (sp), t6
vssb.v v1, (sp), t6
# CHECK: vssb.v v2, (t1), tp, v0.t
vssb.v v2, (t1), tp, v0.t

# CHECK: vssh.v v3, (a0), s0
vssh.v v3, (a0), s0
# CHECK: vssh.v v4, (a4), a2, v0.t
vssh.v v4, (a4), a2, v0.t

# CHECK: vssw.v v5, (s2), a6
vssw.v v5, (s2), a6
# CHECK: vssw.v v6, (s6), s4, v0.t
vssw.v v6, (s6), s4, v0.t

# CHECK: vsse.v v7, (s10), s8
vsse.v v7, (s10), s8
# CHECK: vsse.v v8, (t5), t3, v0.t
vsse.v v8, (t5), t3, v0.t

# CHECK: vlxb.v v10, (ra), v9
vlxb.v v10, (ra), v9
# CHECK: vlxb.v v12, (gp), v11, v0.t
vlxb.v v12, (gp), v11, v0.t

# CHECK: vlxh.v v14, (t0), v13
vlxh.v v14, (t0), v13
# CHECK: vlxh.v v16, (t2), v15, v0.t
vlxh.v v16, (t2), v15, v0.t

# CHECK: vlxw.v v18, (s1), v17
vlxw.v v18, (s1), v17
# CHECK: vlxw.v v20, (a1), v19, v0.t
vlxw.v v20, (a1), v19, v0.t

# CHECK: vlxbu.v v22, (a3), v21
vlxbu.v v22, (a3), v21
# CHECK: vlxbu.v v24, (a5), v23, v0.t
vlxbu.v v24, (a5), v23, v0.t

# CHECK: vlxhu.v v26, (a7), v25
vlxhu.v v26, (a7), v25
# CHECK: vlxhu.v v28, (s3), v27, v0.t
vlxhu.v v28, (s3), v27, v0.t

# CHECK: vlxwu.v v30, (s5), v29
vlxwu.v v30, (s5), v29
# CHECK: vlxwu.v v0, (s7), v31, v0.t
vlxwu.v v0, (s7), v31, v0.t

# CHECK: vlxe.v v2, (s9), v1
vlxe.v v2, (s9), v1
# CHECK: vlxe.v v4, (s11), v3, v0.t
vlxe.v v4, (s11), v3, v0.t

# CHECK: vsxb.v v6, (t4), v5
vsxb.v v6, (t4), v5
# CHECK: vsxb.v v8, (t6), v7, v0.t
vsxb.v v8, (t6), v7, v0.t

# CHECK: vsxh.v v10, (sp), v9
vsxh.v v10, (sp), v9
# CHECK: vsxh.v v12, (tp), v11, v0.t
vsxh.v v12, (tp), v11, v0.t

# CHECK: vsxw.v v14, (t1), v13
vsxw.v v14, (t1), v13
# CHECK: vsxw.v v16, (s0), v15, v0.t
vsxw.v v16, (s0), v15, v0.t

# CHECK: vsxe.v v18, (a0), v17
vsxe.v v18, (a0), v17
# CHECK: vsxe.v v20, (a2), v19, v0.t
vsxe.v v20, (a2), v19, v0.t

# CHECK: vsuxb.v v22, (a4), v21
vsuxb.v v22, (a4), v21
# CHECK: vsuxb.v v24, (a6), v23, v0.t
vsuxb.v v24, (a6), v23, v0.t

# CHECK: vsuxh.v v26, (s2), v25
vsuxh.v v26, (s2), v25
# CHECK: vsuxh.v v28, (s4), v27, v0.t
vsuxh.v v28, (s4), v27, v0.t

# CHECK: vsuxw.v v30, (s6), v29
vsuxw.v v30, (s6), v29
# CHECK: vsuxw.v v0, (s8), v31, v0.t
vsuxw.v v0, (s8), v31, v0.t

# CHECK: vsuxe.v v2, (s10), v1
vsuxe.v v2, (s10), v1
# CHECK: vsuxe.v v4, (t3), v3, v0.t
vsuxe.v v4, (t3), v3, v0.t

