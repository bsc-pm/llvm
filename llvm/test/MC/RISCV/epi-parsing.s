# Generated with utils/EPI/process.py
# RUN: llvm-mc < %s -arch=riscv64 -mattr=+m,+f,+d,+a,+v | FileCheck %s

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

# CHECK: vmsgtu.vi v24, v23, -5
vmsgtu.vi v24, v23, -5
# CHECK: vmsgtu.vi v26, v25, -4, v0.t
vmsgtu.vi v26, v25, -4, v0.t

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

# CHECK: vsll.vv v31, v29, v30
vsll.vv v31, v29, v30
# CHECK: vsll.vv v2, v0, v1, v0.t
vsll.vv v2, v0, v1, v0.t

# CHECK: vsll.vx v4, v3, s4
vsll.vx v4, v3, s4
# CHECK: vsll.vx v6, v5, s6, v0.t
vsll.vx v6, v5, s6, v0.t

# CHECK: vsll.vi v8, v7, 3
vsll.vi v8, v7, 3
# CHECK: vsll.vi v10, v9, 4, v0.t
vsll.vi v10, v9, 4, v0.t

# CHECK: vasub.vv v13, v11, v12
vasub.vv v13, v11, v12
# CHECK: vasub.vv v16, v14, v15, v0.t
vasub.vv v16, v14, v15, v0.t

# CHECK: vasub.vx v18, v17, s8
vasub.vx v18, v17, s8
# CHECK: vasub.vx v20, v19, s10, v0.t
vasub.vx v20, v19, s10, v0.t

# CHECK: vsmul.vv v23, v21, v22
vsmul.vv v23, v21, v22
# CHECK: vsmul.vv v26, v24, v25, v0.t
vsmul.vv v26, v24, v25, v0.t

# CHECK: vsmul.vx v28, v27, t3
vsmul.vx v28, v27, t3
# CHECK: vsmul.vx v30, v29, t5, v0.t
vsmul.vx v30, v29, t5, v0.t

# CHECK: vsrl.vv v1, v31, v0
vsrl.vv v1, v31, v0
# CHECK: vsrl.vv v4, v2, v3, v0.t
vsrl.vv v4, v2, v3, v0.t

# CHECK: vsrl.vx v6, v5, ra
vsrl.vx v6, v5, ra
# CHECK: vsrl.vx v8, v7, gp, v0.t
vsrl.vx v8, v7, gp, v0.t

# CHECK: vsrl.vi v10, v9, 5
vsrl.vi v10, v9, 5
# CHECK: vsrl.vi v12, v11, 6, v0.t
vsrl.vi v12, v11, 6, v0.t

# CHECK: vsra.vv v15, v13, v14
vsra.vv v15, v13, v14
# CHECK: vsra.vv v18, v16, v17, v0.t
vsra.vv v18, v16, v17, v0.t

# CHECK: vsra.vx v20, v19, t0
vsra.vx v20, v19, t0
# CHECK: vsra.vx v22, v21, t2, v0.t
vsra.vx v22, v21, t2, v0.t

# CHECK: vsra.vi v24, v23, 7
vsra.vi v24, v23, 7
# CHECK: vsra.vi v26, v25, 8, v0.t
vsra.vi v26, v25, 8, v0.t

# CHECK: vssrl.vv v29, v27, v28
vssrl.vv v29, v27, v28
# CHECK: vssrl.vv v0, v30, v31, v0.t
vssrl.vv v0, v30, v31, v0.t

# CHECK: vssrl.vx v2, v1, s1
vssrl.vx v2, v1, s1
# CHECK: vssrl.vx v4, v3, a1, v0.t
vssrl.vx v4, v3, a1, v0.t

# CHECK: vssrl.vi v6, v5, 9
vssrl.vi v6, v5, 9
# CHECK: vssrl.vi v8, v7, 10, v0.t
vssrl.vi v8, v7, 10, v0.t

# CHECK: vssra.vv v11, v9, v10
vssra.vv v11, v9, v10
# CHECK: vssra.vv v14, v12, v13, v0.t
vssra.vv v14, v12, v13, v0.t

# CHECK: vssra.vx v16, v15, a3
vssra.vx v16, v15, a3
# CHECK: vssra.vx v18, v17, a5, v0.t
vssra.vx v18, v17, a5, v0.t

# CHECK: vssra.vi v20, v19, 11
vssra.vi v20, v19, 11
# CHECK: vssra.vi v22, v21, 12, v0.t
vssra.vi v22, v21, 12, v0.t

# CHECK: vnsrl.wv v25, v23, v24
vnsrl.wv v25, v23, v24
# CHECK: vnsrl.wv v28, v26, v27, v0.t
vnsrl.wv v28, v26, v27, v0.t

# CHECK: vnsrl.wx v30, v29, a7
vnsrl.wx v30, v29, a7
# CHECK: vnsrl.wx v0, v31, s3, v0.t
vnsrl.wx v0, v31, s3, v0.t

# CHECK: vnsrl.wi v2, v1, 13
vnsrl.wi v2, v1, 13
# CHECK: vnsrl.wi v4, v3, 14, v0.t
vnsrl.wi v4, v3, 14, v0.t

# CHECK: vnsra.wv v7, v5, v6
vnsra.wv v7, v5, v6
# CHECK: vnsra.wv v10, v8, v9, v0.t
vnsra.wv v10, v8, v9, v0.t

# CHECK: vnsra.wx v12, v11, s5
vnsra.wx v12, v11, s5
# CHECK: vnsra.wx v14, v13, s7, v0.t
vnsra.wx v14, v13, s7, v0.t

# CHECK: vnsra.wi v16, v15, 15
vnsra.wi v16, v15, 15
# CHECK: vnsra.wi v18, v17, 16, v0.t
vnsra.wi v18, v17, 16, v0.t

# CHECK: vnclipu.wv v21, v19, v20
vnclipu.wv v21, v19, v20
# CHECK: vnclipu.wv v24, v22, v23, v0.t
vnclipu.wv v24, v22, v23, v0.t

# CHECK: vnclipu.wx v26, v25, s9
vnclipu.wx v26, v25, s9
# CHECK: vnclipu.wx v28, v27, s11, v0.t
vnclipu.wx v28, v27, s11, v0.t

# CHECK: vnclipu.wi v30, v29, 17
vnclipu.wi v30, v29, 17
# CHECK: vnclipu.wi v0, v31, 18, v0.t
vnclipu.wi v0, v31, 18, v0.t

# CHECK: vnclip.wv v3, v1, v2
vnclip.wv v3, v1, v2
# CHECK: vnclip.wv v6, v4, v5, v0.t
vnclip.wv v6, v4, v5, v0.t

# CHECK: vnclip.wx v8, v7, t4
vnclip.wx v8, v7, t4
# CHECK: vnclip.wx v10, v9, t6, v0.t
vnclip.wx v10, v9, t6, v0.t

# CHECK: vnclip.wi v12, v11, 19
vnclip.wi v12, v11, 19
# CHECK: vnclip.wi v14, v13, 20, v0.t
vnclip.wi v14, v13, 20, v0.t

# CHECK: vwredsumu.vs v18, v15, v16
vwredsumu.vs v18, v15, v16
# CHECK: vwredsumu.vs v22, v19, v20, v0.t
vwredsumu.vs v22, v19, v20, v0.t

# CHECK: vwredsum.vs v26, v23, v24
vwredsum.vs v26, v23, v24
# CHECK: vwredsum.vs v30, v27, v28, v0.t
vwredsum.vs v30, v27, v28, v0.t

# CHECK: vredsum.vs v1, v31, v0
vredsum.vs v1, v31, v0
# CHECK: vredsum.vs v4, v2, v3, v0.t
vredsum.vs v4, v2, v3, v0.t

# CHECK: vredand.vs v7, v5, v6
vredand.vs v7, v5, v6
# CHECK: vredand.vs v10, v8, v9, v0.t
vredand.vs v10, v8, v9, v0.t

# CHECK: vredor.vs v13, v11, v12
vredor.vs v13, v11, v12
# CHECK: vredor.vs v16, v14, v15, v0.t
vredor.vs v16, v14, v15, v0.t

# CHECK: vredxor.vs v19, v17, v18
vredxor.vs v19, v17, v18
# CHECK: vredxor.vs v22, v20, v21, v0.t
vredxor.vs v22, v20, v21, v0.t

# CHECK: vredminu.vs v25, v23, v24
vredminu.vs v25, v23, v24
# CHECK: vredminu.vs v28, v26, v27, v0.t
vredminu.vs v28, v26, v27, v0.t

# CHECK: vredmin.vs v31, v29, v30
vredmin.vs v31, v29, v30
# CHECK: vredmin.vs v2, v0, v1, v0.t
vredmin.vs v2, v0, v1, v0.t

# CHECK: vredmaxu.vs v5, v3, v4
vredmaxu.vs v5, v3, v4
# CHECK: vredmaxu.vs v8, v6, v7, v0.t
vredmaxu.vs v8, v6, v7, v0.t

# CHECK: vredmax.vs v11, v9, v10
vredmax.vs v11, v9, v10
# CHECK: vredmax.vs v14, v12, v13, v0.t
vredmax.vs v14, v12, v13, v0.t

# CHECK: vmv.s.x v15, sp
vmv.s.x v15, sp

# CHECK: vslide1up.vx v17, v16, tp
vslide1up.vx v17, v16, tp
# CHECK: vslide1up.vx v19, v18, t1, v0.t
vslide1up.vx v19, v18, t1, v0.t

# CHECK: vslide1down.vx v21, v20, s0
vslide1down.vx v21, v20, s0
# CHECK: vslide1down.vx v23, v22, a0, v0.t
vslide1down.vx v23, v22, a0, v0.t

# CHECK: vmv.x.s a2, v24
vmv.x.s a2, v24

# CHECK: vpopc.m a4, v25
vpopc.m a4, v25
# CHECK: vpopc.m a6, v26, v0.t
vpopc.m a6, v26, v0.t

# CHECK: vfirst.m s2, v27
vfirst.m s2, v27
# CHECK: vfirst.m s4, v28, v0.t
vfirst.m s4, v28, v0.t

# CHECK: vcompress.vm v31, v29, v30
vcompress.vm v31, v29, v30

# CHECK: vmandnot.mm v2, v0, v1
vmandnot.mm v2, v0, v1

# CHECK: vmand.mm v5, v3, v4
vmand.mm v5, v3, v4

# CHECK: vmor.mm v8, v6, v7
vmor.mm v8, v6, v7

# CHECK: vmxor.mm v11, v9, v10
vmxor.mm v11, v9, v10

# CHECK: vmornot.mm v14, v12, v13
vmornot.mm v14, v12, v13

# CHECK: vmnand.mm v17, v15, v16
vmnand.mm v17, v15, v16

# CHECK: vmnor.mm v20, v18, v19
vmnor.mm v20, v18, v19

# CHECK: vmxnor.mm v23, v21, v22
vmxnor.mm v23, v21, v22

# CHECK: vdivu.vv v26, v24, v25
vdivu.vv v26, v24, v25
# CHECK: vdivu.vv v29, v27, v28, v0.t
vdivu.vv v29, v27, v28, v0.t

# CHECK: vdivu.vx v31, v30, s6
vdivu.vx v31, v30, s6
# CHECK: vdivu.vx v1, v0, s8, v0.t
vdivu.vx v1, v0, s8, v0.t

# CHECK: vdiv.vv v4, v2, v3
vdiv.vv v4, v2, v3
# CHECK: vdiv.vv v7, v5, v6, v0.t
vdiv.vv v7, v5, v6, v0.t

# CHECK: vdiv.vx v9, v8, s10
vdiv.vx v9, v8, s10
# CHECK: vdiv.vx v11, v10, t3, v0.t
vdiv.vx v11, v10, t3, v0.t

# CHECK: vremu.vv v14, v12, v13
vremu.vv v14, v12, v13
# CHECK: vremu.vv v17, v15, v16, v0.t
vremu.vv v17, v15, v16, v0.t

# CHECK: vremu.vx v19, v18, t5
vremu.vx v19, v18, t5
# CHECK: vremu.vx v21, v20, ra, v0.t
vremu.vx v21, v20, ra, v0.t

# CHECK: vrem.vv v24, v22, v23
vrem.vv v24, v22, v23
# CHECK: vrem.vv v27, v25, v26, v0.t
vrem.vv v27, v25, v26, v0.t

# CHECK: vrem.vx v29, v28, gp
vrem.vx v29, v28, gp
# CHECK: vrem.vx v31, v30, t0, v0.t
vrem.vx v31, v30, t0, v0.t

# CHECK: vmulhu.vv v2, v0, v1
vmulhu.vv v2, v0, v1
# CHECK: vmulhu.vv v5, v3, v4, v0.t
vmulhu.vv v5, v3, v4, v0.t

# CHECK: vmulhu.vx v7, v6, t2
vmulhu.vx v7, v6, t2
# CHECK: vmulhu.vx v9, v8, s1, v0.t
vmulhu.vx v9, v8, s1, v0.t

# CHECK: vmul.vv v12, v10, v11
vmul.vv v12, v10, v11
# CHECK: vmul.vv v15, v13, v14, v0.t
vmul.vv v15, v13, v14, v0.t

# CHECK: vmul.vx v17, v16, a1
vmul.vx v17, v16, a1
# CHECK: vmul.vx v19, v18, a3, v0.t
vmul.vx v19, v18, a3, v0.t

# CHECK: vmulhsu.vv v22, v20, v21
vmulhsu.vv v22, v20, v21
# CHECK: vmulhsu.vv v25, v23, v24, v0.t
vmulhsu.vv v25, v23, v24, v0.t

# CHECK: vmulhsu.vx v27, v26, a5
vmulhsu.vx v27, v26, a5
# CHECK: vmulhsu.vx v29, v28, a7, v0.t
vmulhsu.vx v29, v28, a7, v0.t

# CHECK: vmulh.vv v0, v30, v31
vmulh.vv v0, v30, v31
# CHECK: vmulh.vv v3, v1, v2, v0.t
vmulh.vv v3, v1, v2, v0.t

# CHECK: vmulh.vx v5, v4, s3
vmulh.vx v5, v4, s3
# CHECK: vmulh.vx v7, v6, s5, v0.t
vmulh.vx v7, v6, s5, v0.t

# CHECK: vmadd.vv v10, v9, v8
vmadd.vv v10, v9, v8
# CHECK: vmadd.vv v13, v12, v11, v0.t
vmadd.vv v13, v12, v11, v0.t

# CHECK: vmadd.vx v15, s7, v14
vmadd.vx v15, s7, v14
# CHECK: vmadd.vx v17, s9, v16, v0.t
vmadd.vx v17, s9, v16, v0.t

# CHECK: vnmsub.vv v20, v19, v18
vnmsub.vv v20, v19, v18
# CHECK: vnmsub.vv v23, v22, v21, v0.t
vnmsub.vv v23, v22, v21, v0.t

# CHECK: vnmsub.vx v25, s11, v24
vnmsub.vx v25, s11, v24
# CHECK: vnmsub.vx v27, t4, v26, v0.t
vnmsub.vx v27, t4, v26, v0.t

# CHECK: vmacc.vv v30, v29, v28
vmacc.vv v30, v29, v28
# CHECK: vmacc.vv v1, v0, v31, v0.t
vmacc.vv v1, v0, v31, v0.t

# CHECK: vmacc.vx v3, t6, v2
vmacc.vx v3, t6, v2
# CHECK: vmacc.vx v5, sp, v4, v0.t
vmacc.vx v5, sp, v4, v0.t

# CHECK: vnmsac.vv v8, v7, v6
vnmsac.vv v8, v7, v6
# CHECK: vnmsac.vv v11, v10, v9, v0.t
vnmsac.vv v11, v10, v9, v0.t

# CHECK: vnmsac.vx v13, tp, v12
vnmsac.vx v13, tp, v12
# CHECK: vnmsac.vx v15, t1, v14, v0.t
vnmsac.vx v15, t1, v14, v0.t

# CHECK: vwaddu.vv v18, v16, v17
vwaddu.vv v18, v16, v17
# CHECK: vwaddu.vv v22, v19, v20, v0.t
vwaddu.vv v22, v19, v20, v0.t

# CHECK: vwaddu.vx v24, v23, s0
vwaddu.vx v24, v23, s0
# CHECK: vwaddu.vx v26, v25, a0, v0.t
vwaddu.vx v26, v25, a0, v0.t

# CHECK: vwadd.vv v30, v27, v28
vwadd.vv v30, v27, v28
# CHECK: vwadd.vv v2, v31, v1, v0.t
vwadd.vv v2, v31, v1, v0.t

# CHECK: vwadd.vx v4, v3, a2
vwadd.vx v4, v3, a2
# CHECK: vwadd.vx v6, v5, a4, v0.t
vwadd.vx v6, v5, a4, v0.t

# CHECK: vwsubu.vv v10, v7, v8
vwsubu.vv v10, v7, v8
# CHECK: vwsubu.vv v14, v11, v12, v0.t
vwsubu.vv v14, v11, v12, v0.t

# CHECK: vwsubu.vx v16, v15, a6
vwsubu.vx v16, v15, a6
# CHECK: vwsubu.vx v18, v17, s2, v0.t
vwsubu.vx v18, v17, s2, v0.t

# CHECK: vwsub.vv v22, v19, v20
vwsub.vv v22, v19, v20
# CHECK: vwsub.vv v26, v23, v24, v0.t
vwsub.vv v26, v23, v24, v0.t

# CHECK: vwsub.vx v28, v27, s4
vwsub.vx v28, v27, s4
# CHECK: vwsub.vx v30, v29, s6, v0.t
vwsub.vx v30, v29, s6, v0.t

# CHECK: vwaddu.wv v2, v31, v1
vwaddu.wv v2, v31, v1
# CHECK: vwaddu.wv v6, v3, v4, v0.t
vwaddu.wv v6, v3, v4, v0.t

# CHECK: vwaddu.wx v8, v7, s8
vwaddu.wx v8, v7, s8
# CHECK: vwaddu.wx v10, v9, s10, v0.t
vwaddu.wx v10, v9, s10, v0.t

# CHECK: vwadd.wv v14, v11, v12
vwadd.wv v14, v11, v12
# CHECK: vwadd.wv v18, v15, v16, v0.t
vwadd.wv v18, v15, v16, v0.t

# CHECK: vwadd.wx v20, v19, t3
vwadd.wx v20, v19, t3
# CHECK: vwadd.wx v22, v21, t5, v0.t
vwadd.wx v22, v21, t5, v0.t

# CHECK: vwsubu.wv v26, v23, v24
vwsubu.wv v26, v23, v24
# CHECK: vwsubu.wv v30, v27, v28, v0.t
vwsubu.wv v30, v27, v28, v0.t

# CHECK: vwsubu.wx v2, v31, ra
vwsubu.wx v2, v31, ra
# CHECK: vwsubu.wx v4, v3, gp, v0.t
vwsubu.wx v4, v3, gp, v0.t

# CHECK: vwsub.wv v8, v5, v6
vwsub.wv v8, v5, v6
# CHECK: vwsub.wv v12, v9, v10, v0.t
vwsub.wv v12, v9, v10, v0.t

# CHECK: vwsub.wx v14, v13, t0
vwsub.wx v14, v13, t0
# CHECK: vwsub.wx v16, v15, t2, v0.t
vwsub.wx v16, v15, t2, v0.t

# CHECK: vwmulu.vv v20, v17, v18
vwmulu.vv v20, v17, v18
# CHECK: vwmulu.vv v24, v21, v22, v0.t
vwmulu.vv v24, v21, v22, v0.t

# CHECK: vwmulu.vx v26, v25, s1
vwmulu.vx v26, v25, s1
# CHECK: vwmulu.vx v28, v27, a1, v0.t
vwmulu.vx v28, v27, a1, v0.t

# CHECK: vwmulsu.vv v2, v29, v30
vwmulsu.vv v2, v29, v30
# CHECK: vwmulsu.vv v6, v3, v4, v0.t
vwmulsu.vv v6, v3, v4, v0.t

# CHECK: vwmulsu.vx v8, v7, a3
vwmulsu.vx v8, v7, a3
# CHECK: vwmulsu.vx v10, v9, a5, v0.t
vwmulsu.vx v10, v9, a5, v0.t

# CHECK: vwmul.vv v14, v11, v12
vwmul.vv v14, v11, v12
# CHECK: vwmul.vv v18, v15, v16, v0.t
vwmul.vv v18, v15, v16, v0.t

# CHECK: vwmul.vx v20, v19, a7
vwmul.vx v20, v19, a7
# CHECK: vwmul.vx v22, v21, s3, v0.t
vwmul.vx v22, v21, s3, v0.t

# CHECK: vwmaccu.vv v26, v24, v23
vwmaccu.vv v26, v24, v23
# CHECK: vwmaccu.vv v30, v28, v27, v0.t
vwmaccu.vv v30, v28, v27, v0.t

# CHECK: vwmaccu.vx v2, s5, v31
vwmaccu.vx v2, s5, v31
# CHECK: vwmaccu.vx v4, s7, v3, v0.t
vwmaccu.vx v4, s7, v3, v0.t

# CHECK: vwmacc.vv v8, v6, v5
vwmacc.vv v8, v6, v5
# CHECK: vwmacc.vv v12, v10, v9, v0.t
vwmacc.vv v12, v10, v9, v0.t

# CHECK: vwmacc.vx v14, s9, v13
vwmacc.vx v14, s9, v13
# CHECK: vwmacc.vx v16, s11, v15, v0.t
vwmacc.vx v16, s11, v15, v0.t

# CHECK: vwmaccus.vx v18, t4, v17
vwmaccus.vx v18, t4, v17
# CHECK: vwmaccus.vx v20, t6, v19, v0.t
vwmaccus.vx v20, t6, v19, v0.t

# CHECK: vwmaccsu.vv v24, v22, v21
vwmaccsu.vv v24, v22, v21
# CHECK: vwmaccsu.vv v28, v26, v25, v0.t
vwmaccsu.vv v28, v26, v25, v0.t

# CHECK: vwmaccsu.vx v30, sp, v29
vwmaccsu.vx v30, sp, v29
# CHECK: vwmaccsu.vx v2, tp, v31, v0.t
vwmaccsu.vx v2, tp, v31, v0.t

# CHECK: vfadd.vv v5, v3, v4
vfadd.vv v5, v3, v4
# CHECK: vfadd.vv v8, v6, v7, v0.t
vfadd.vv v8, v6, v7, v0.t

# CHECK: vfadd.vf v10, v9, ft0
vfadd.vf v10, v9, ft0
# CHECK: vfadd.vf v12, v11, ft1, v0.t
vfadd.vf v12, v11, ft1, v0.t

# CHECK: vfredsum.vs v15, v13, v14
vfredsum.vs v15, v13, v14
# CHECK: vfredsum.vs v18, v16, v17, v0.t
vfredsum.vs v18, v16, v17, v0.t

# CHECK: vfsub.vv v21, v19, v20
vfsub.vv v21, v19, v20
# CHECK: vfsub.vv v24, v22, v23, v0.t
vfsub.vv v24, v22, v23, v0.t

# CHECK: vfsub.vf v26, v25, ft2
vfsub.vf v26, v25, ft2
# CHECK: vfsub.vf v28, v27, ft3, v0.t
vfsub.vf v28, v27, ft3, v0.t

# CHECK: vfredosum.vs v31, v29, v30
vfredosum.vs v31, v29, v30
# CHECK: vfredosum.vs v2, v0, v1, v0.t
vfredosum.vs v2, v0, v1, v0.t

# CHECK: vfmin.vv v5, v3, v4
vfmin.vv v5, v3, v4
# CHECK: vfmin.vv v8, v6, v7, v0.t
vfmin.vv v8, v6, v7, v0.t

# CHECK: vfmin.vf v10, v9, ft4
vfmin.vf v10, v9, ft4
# CHECK: vfmin.vf v12, v11, ft5, v0.t
vfmin.vf v12, v11, ft5, v0.t

# CHECK: vfredmin.vs v15, v13, v14
vfredmin.vs v15, v13, v14
# CHECK: vfredmin.vs v18, v16, v17, v0.t
vfredmin.vs v18, v16, v17, v0.t

# CHECK: vfmax.vv v21, v19, v20
vfmax.vv v21, v19, v20
# CHECK: vfmax.vv v24, v22, v23, v0.t
vfmax.vv v24, v22, v23, v0.t

# CHECK: vfmax.vf v26, v25, ft6
vfmax.vf v26, v25, ft6
# CHECK: vfmax.vf v28, v27, ft7, v0.t
vfmax.vf v28, v27, ft7, v0.t

# CHECK: vfredmax.vs v31, v29, v30
vfredmax.vs v31, v29, v30
# CHECK: vfredmax.vs v2, v0, v1, v0.t
vfredmax.vs v2, v0, v1, v0.t

# CHECK: vfsgnj.vv v5, v3, v4
vfsgnj.vv v5, v3, v4
# CHECK: vfsgnj.vv v8, v6, v7, v0.t
vfsgnj.vv v8, v6, v7, v0.t

# CHECK: vfsgnj.vf v10, v9, fs0
vfsgnj.vf v10, v9, fs0
# CHECK: vfsgnj.vf v12, v11, fs1, v0.t
vfsgnj.vf v12, v11, fs1, v0.t

# CHECK: vfsgnjn.vv v15, v13, v14
vfsgnjn.vv v15, v13, v14
# CHECK: vfsgnjn.vv v18, v16, v17, v0.t
vfsgnjn.vv v18, v16, v17, v0.t

# CHECK: vfsgnjn.vf v20, v19, fa0
vfsgnjn.vf v20, v19, fa0
# CHECK: vfsgnjn.vf v22, v21, fa1, v0.t
vfsgnjn.vf v22, v21, fa1, v0.t

# CHECK: vfsgnjx.vv v25, v23, v24
vfsgnjx.vv v25, v23, v24
# CHECK: vfsgnjx.vv v28, v26, v27, v0.t
vfsgnjx.vv v28, v26, v27, v0.t

# CHECK: vfsgnjx.vf v30, v29, fa2
vfsgnjx.vf v30, v29, fa2
# CHECK: vfsgnjx.vf v0, v31, fa3, v0.t
vfsgnjx.vf v0, v31, fa3, v0.t

# CHECK: vfmv.v.f v1, fa4
vfmv.v.f v1, fa4

# CHECK: vfmv.f.s fa5, v2
vfmv.f.s fa5, v2

# CHECK: vfmv.s.f v3, fa6
vfmv.s.f v3, fa6

# CHECK: vfmerge.vfm v5, v4, fa7, v0
vfmerge.vfm v5, v4, fa7, v0

# CHECK: vmfeq.vv v8, v6, v7
vmfeq.vv v8, v6, v7
# CHECK: vmfeq.vv v11, v9, v10, v0.t
vmfeq.vv v11, v9, v10, v0.t

# CHECK: vmfeq.vf v13, v12, fs2
vmfeq.vf v13, v12, fs2
# CHECK: vmfeq.vf v15, v14, fs3, v0.t
vmfeq.vf v15, v14, fs3, v0.t

# CHECK: vmfle.vv v18, v16, v17
vmfle.vv v18, v16, v17
# CHECK: vmfle.vv v21, v19, v20, v0.t
vmfle.vv v21, v19, v20, v0.t

# CHECK: vmfle.vf v23, v22, fs4
vmfle.vf v23, v22, fs4
# CHECK: vmfle.vf v25, v24, fs5, v0.t
vmfle.vf v25, v24, fs5, v0.t

# CHECK: vmflt.vv v28, v26, v27
vmflt.vv v28, v26, v27
# CHECK: vmflt.vv v31, v29, v30, v0.t
vmflt.vv v31, v29, v30, v0.t

# CHECK: vmflt.vf v1, v0, fs6
vmflt.vf v1, v0, fs6
# CHECK: vmflt.vf v3, v2, fs7, v0.t
vmflt.vf v3, v2, fs7, v0.t

# CHECK: vmfne.vv v6, v4, v5
vmfne.vv v6, v4, v5
# CHECK: vmfne.vv v9, v7, v8, v0.t
vmfne.vv v9, v7, v8, v0.t

# CHECK: vmfne.vf v11, v10, fs8
vmfne.vf v11, v10, fs8
# CHECK: vmfne.vf v13, v12, fs9, v0.t
vmfne.vf v13, v12, fs9, v0.t

# CHECK: vmfgt.vf v15, v14, fs10
vmfgt.vf v15, v14, fs10
# CHECK: vmfgt.vf v17, v16, fs11, v0.t
vmfgt.vf v17, v16, fs11, v0.t

# CHECK: vmfge.vf v19, v18, ft8
vmfge.vf v19, v18, ft8
# CHECK: vmfge.vf v21, v20, ft9, v0.t
vmfge.vf v21, v20, ft9, v0.t

# CHECK: vfdiv.vv v24, v22, v23
vfdiv.vv v24, v22, v23
# CHECK: vfdiv.vv v27, v25, v26, v0.t
vfdiv.vv v27, v25, v26, v0.t

# CHECK: vfdiv.vf v29, v28, ft10
vfdiv.vf v29, v28, ft10
# CHECK: vfdiv.vf v31, v30, ft0, v0.t
vfdiv.vf v31, v30, ft0, v0.t

# CHECK: vfrdiv.vf v1, v0, ft1
vfrdiv.vf v1, v0, ft1
# CHECK: vfrdiv.vf v3, v2, ft2, v0.t
vfrdiv.vf v3, v2, ft2, v0.t

# CHECK: vfmul.vv v6, v4, v5
vfmul.vv v6, v4, v5
# CHECK: vfmul.vv v9, v7, v8, v0.t
vfmul.vv v9, v7, v8, v0.t

# CHECK: vfmul.vf v11, v10, ft3
vfmul.vf v11, v10, ft3
# CHECK: vfmul.vf v13, v12, ft4, v0.t
vfmul.vf v13, v12, ft4, v0.t

# CHECK: vfmadd.vv v16, v15, v14
vfmadd.vv v16, v15, v14
# CHECK: vfmadd.vv v19, v18, v17, v0.t
vfmadd.vv v19, v18, v17, v0.t

# CHECK: vfmadd.vf v21, ft5, v20
vfmadd.vf v21, ft5, v20
# CHECK: vfmadd.vf v23, ft6, v22, v0.t
vfmadd.vf v23, ft6, v22, v0.t

# CHECK: vfnmadd.vv v26, v25, v24
vfnmadd.vv v26, v25, v24
# CHECK: vfnmadd.vv v29, v28, v27, v0.t
vfnmadd.vv v29, v28, v27, v0.t

# CHECK: vfnmadd.vf v31, ft7, v30
vfnmadd.vf v31, ft7, v30
# CHECK: vfnmadd.vf v1, fs0, v0, v0.t
vfnmadd.vf v1, fs0, v0, v0.t

# CHECK: vfmsub.vv v4, v3, v2
vfmsub.vv v4, v3, v2
# CHECK: vfmsub.vv v7, v6, v5, v0.t
vfmsub.vv v7, v6, v5, v0.t

# CHECK: vfmsub.vf v9, fs1, v8
vfmsub.vf v9, fs1, v8
# CHECK: vfmsub.vf v11, fa0, v10, v0.t
vfmsub.vf v11, fa0, v10, v0.t

# CHECK: vfnmsub.vv v14, v13, v12
vfnmsub.vv v14, v13, v12
# CHECK: vfnmsub.vv v17, v16, v15, v0.t
vfnmsub.vv v17, v16, v15, v0.t

# CHECK: vfnmsub.vf v19, fa1, v18
vfnmsub.vf v19, fa1, v18
# CHECK: vfnmsub.vf v21, fa2, v20, v0.t
vfnmsub.vf v21, fa2, v20, v0.t

# CHECK: vfmacc.vv v24, v23, v22
vfmacc.vv v24, v23, v22
# CHECK: vfmacc.vv v27, v26, v25, v0.t
vfmacc.vv v27, v26, v25, v0.t

# CHECK: vfmacc.vf v29, fa3, v28
vfmacc.vf v29, fa3, v28
# CHECK: vfmacc.vf v31, fa4, v30, v0.t
vfmacc.vf v31, fa4, v30, v0.t

# CHECK: vfnmacc.vv v2, v1, v0
vfnmacc.vv v2, v1, v0
# CHECK: vfnmacc.vv v5, v4, v3, v0.t
vfnmacc.vv v5, v4, v3, v0.t

# CHECK: vfnmacc.vf v7, fa5, v6
vfnmacc.vf v7, fa5, v6
# CHECK: vfnmacc.vf v9, fa6, v8, v0.t
vfnmacc.vf v9, fa6, v8, v0.t

# CHECK: vfmsac.vv v12, v11, v10
vfmsac.vv v12, v11, v10
# CHECK: vfmsac.vv v15, v14, v13, v0.t
vfmsac.vv v15, v14, v13, v0.t

# CHECK: vfmsac.vf v17, fa7, v16
vfmsac.vf v17, fa7, v16
# CHECK: vfmsac.vf v19, fs2, v18, v0.t
vfmsac.vf v19, fs2, v18, v0.t

# CHECK: vfnmsac.vv v22, v21, v20
vfnmsac.vv v22, v21, v20
# CHECK: vfnmsac.vv v25, v24, v23, v0.t
vfnmsac.vv v25, v24, v23, v0.t

# CHECK: vfnmsac.vf v27, fs3, v26
vfnmsac.vf v27, fs3, v26
# CHECK: vfnmsac.vf v29, fs4, v28, v0.t
vfnmsac.vf v29, fs4, v28, v0.t

# CHECK: vfwadd.vv v2, v30, v31
vfwadd.vv v2, v30, v31
# CHECK: vfwadd.vv v6, v3, v4, v0.t
vfwadd.vv v6, v3, v4, v0.t

# CHECK: vfwadd.vf v8, v7, fs5
vfwadd.vf v8, v7, fs5
# CHECK: vfwadd.vf v10, v9, fs6, v0.t
vfwadd.vf v10, v9, fs6, v0.t

# CHECK: vfwredsum.vs v14, v11, v12
vfwredsum.vs v14, v11, v12
# CHECK: vfwredsum.vs v18, v15, v16, v0.t
vfwredsum.vs v18, v15, v16, v0.t

# CHECK: vfwsub.vv v22, v19, v20
vfwsub.vv v22, v19, v20
# CHECK: vfwsub.vv v26, v23, v24, v0.t
vfwsub.vv v26, v23, v24, v0.t

# CHECK: vfwsub.vf v28, v27, fs7
vfwsub.vf v28, v27, fs7
# CHECK: vfwsub.vf v30, v29, fs8, v0.t
vfwsub.vf v30, v29, fs8, v0.t

# CHECK: vfwredosum.vs v2, v31, v1
vfwredosum.vs v2, v31, v1
# CHECK: vfwredosum.vs v6, v3, v4, v0.t
vfwredosum.vs v6, v3, v4, v0.t

# CHECK: vfwadd.wv v10, v7, v8
vfwadd.wv v10, v7, v8
# CHECK: vfwadd.wv v14, v11, v12, v0.t
vfwadd.wv v14, v11, v12, v0.t

# CHECK: vfwadd.wf v16, v15, fs9
vfwadd.wf v16, v15, fs9
# CHECK: vfwadd.wf v18, v17, fs10, v0.t
vfwadd.wf v18, v17, fs10, v0.t

# CHECK: vfwsub.wv v22, v19, v20
vfwsub.wv v22, v19, v20
# CHECK: vfwsub.wv v26, v23, v24, v0.t
vfwsub.wv v26, v23, v24, v0.t

# CHECK: vfwsub.wf v28, v27, fs11
vfwsub.wf v28, v27, fs11
# CHECK: vfwsub.wf v30, v29, ft8, v0.t
vfwsub.wf v30, v29, ft8, v0.t

# CHECK: vfwmul.vv v2, v31, v1
vfwmul.vv v2, v31, v1
# CHECK: vfwmul.vv v6, v3, v4, v0.t
vfwmul.vv v6, v3, v4, v0.t

# CHECK: vfwmul.vf v8, v7, ft9
vfwmul.vf v8, v7, ft9
# CHECK: vfwmul.vf v10, v9, ft10, v0.t
vfwmul.vf v10, v9, ft10, v0.t

# CHECK: vfwmacc.vv v14, v12, v11
vfwmacc.vv v14, v12, v11
# CHECK: vfwmacc.vv v18, v16, v15, v0.t
vfwmacc.vv v18, v16, v15, v0.t

# CHECK: vfwmacc.vf v20, ft0, v19
vfwmacc.vf v20, ft0, v19
# CHECK: vfwmacc.vf v22, ft1, v21, v0.t
vfwmacc.vf v22, ft1, v21, v0.t

# CHECK: vfwnmacc.vv v26, v24, v23
vfwnmacc.vv v26, v24, v23
# CHECK: vfwnmacc.vv v30, v28, v27, v0.t
vfwnmacc.vv v30, v28, v27, v0.t

# CHECK: vfwnmacc.vf v2, ft2, v31
vfwnmacc.vf v2, ft2, v31
# CHECK: vfwnmacc.vf v4, ft3, v3, v0.t
vfwnmacc.vf v4, ft3, v3, v0.t

# CHECK: vfwmsac.vv v8, v6, v5
vfwmsac.vv v8, v6, v5
# CHECK: vfwmsac.vv v12, v10, v9, v0.t
vfwmsac.vv v12, v10, v9, v0.t

# CHECK: vfwmsac.vf v14, ft4, v13
vfwmsac.vf v14, ft4, v13
# CHECK: vfwmsac.vf v16, ft5, v15, v0.t
vfwmsac.vf v16, ft5, v15, v0.t

# CHECK: vfwnmsac.vv v20, v18, v17
vfwnmsac.vv v20, v18, v17
# CHECK: vfwnmsac.vv v24, v22, v21, v0.t
vfwnmsac.vv v24, v22, v21, v0.t

# CHECK: vfwnmsac.vf v26, ft6, v25
vfwnmsac.vf v26, ft6, v25
# CHECK: vfwnmsac.vf v28, ft7, v27, v0.t
vfwnmsac.vf v28, ft7, v27, v0.t

# CHECK: vfsqrt.v v30, v29
vfsqrt.v v30, v29
# CHECK: vfsqrt.v v0, v31, v0.t
vfsqrt.v v0, v31, v0.t

# CHECK: vfclass.v v2, v1
vfclass.v v2, v1
# CHECK: vfclass.v v4, v3, v0.t
vfclass.v v4, v3, v0.t

# CHECK: vfcvt.xu.f.v v6, v5
vfcvt.xu.f.v v6, v5
# CHECK: vfcvt.xu.f.v v8, v7, v0.t
vfcvt.xu.f.v v8, v7, v0.t

# CHECK: vfcvt.x.f.v v10, v9
vfcvt.x.f.v v10, v9
# CHECK: vfcvt.x.f.v v12, v11, v0.t
vfcvt.x.f.v v12, v11, v0.t

# CHECK: vfcvt.f.xu.v v14, v13
vfcvt.f.xu.v v14, v13
# CHECK: vfcvt.f.xu.v v16, v15, v0.t
vfcvt.f.xu.v v16, v15, v0.t

# CHECK: vfcvt.f.x.v v18, v17
vfcvt.f.x.v v18, v17
# CHECK: vfcvt.f.x.v v20, v19, v0.t
vfcvt.f.x.v v20, v19, v0.t

# CHECK: vfwcvt.xu.f.v v22, v21
vfwcvt.xu.f.v v22, v21
# CHECK: vfwcvt.xu.f.v v24, v23, v0.t
vfwcvt.xu.f.v v24, v23, v0.t

# CHECK: vfwcvt.x.f.v v26, v25
vfwcvt.x.f.v v26, v25
# CHECK: vfwcvt.x.f.v v28, v27, v0.t
vfwcvt.x.f.v v28, v27, v0.t

# CHECK: vfwcvt.f.xu.v v30, v29
vfwcvt.f.xu.v v30, v29
# CHECK: vfwcvt.f.xu.v v2, v31, v0.t
vfwcvt.f.xu.v v2, v31, v0.t

# CHECK: vfwcvt.f.x.v v4, v3
vfwcvt.f.x.v v4, v3
# CHECK: vfwcvt.f.x.v v6, v5, v0.t
vfwcvt.f.x.v v6, v5, v0.t

# CHECK: vfwcvt.f.f.v v8, v7
vfwcvt.f.f.v v8, v7
# CHECK: vfwcvt.f.f.v v10, v9, v0.t
vfwcvt.f.f.v v10, v9, v0.t

# CHECK: vfncvt.xu.f.w v12, v11
vfncvt.xu.f.w v12, v11
# CHECK: vfncvt.xu.f.w v14, v13, v0.t
vfncvt.xu.f.w v14, v13, v0.t

# CHECK: vfncvt.x.f.w v16, v15
vfncvt.x.f.w v16, v15
# CHECK: vfncvt.x.f.w v18, v17, v0.t
vfncvt.x.f.w v18, v17, v0.t

# CHECK: vfncvt.f.xu.w v20, v19
vfncvt.f.xu.w v20, v19
# CHECK: vfncvt.f.xu.w v22, v21, v0.t
vfncvt.f.xu.w v22, v21, v0.t

# CHECK: vfncvt.f.x.w v24, v23
vfncvt.f.x.w v24, v23
# CHECK: vfncvt.f.x.w v26, v25, v0.t
vfncvt.f.x.w v26, v25, v0.t

# CHECK: vfncvt.f.f.w v28, v27
vfncvt.f.f.w v28, v27
# CHECK: vfncvt.f.f.w v30, v29, v0.t
vfncvt.f.f.w v30, v29, v0.t

# CHECK: vmsbf.m v0, v31
vmsbf.m v0, v31
# CHECK: vmsbf.m v2, v1, v0.t
vmsbf.m v2, v1, v0.t

# CHECK: vmsof.m v4, v3
vmsof.m v4, v3
# CHECK: vmsof.m v6, v5, v0.t
vmsof.m v6, v5, v0.t

# CHECK: vmsif.m v8, v7
vmsif.m v8, v7
# CHECK: vmsif.m v10, v9, v0.t
vmsif.m v10, v9, v0.t

# CHECK: viota.m v12, v11
viota.m v12, v11
# CHECK: viota.m v14, v13, v0.t
viota.m v14, v13, v0.t

# CHECK: vid.v v15
vid.v v15
# CHECK: vid.v v16, v0.t
vid.v v16, v0.t

# CHECK: vlb.v v17, (t1)
vlb.v v17, (t1)
# CHECK: vlb.v v18, (s0), v0.t
vlb.v v18, (s0), v0.t

# CHECK: vlh.v v19, (a0)
vlh.v v19, (a0)
# CHECK: vlh.v v20, (a2), v0.t
vlh.v v20, (a2), v0.t

# CHECK: vlw.v v21, (a4)
vlw.v v21, (a4)
# CHECK: vlw.v v22, (a6), v0.t
vlw.v v22, (a6), v0.t

# CHECK: vlbu.v v23, (s2)
vlbu.v v23, (s2)
# CHECK: vlbu.v v24, (s4), v0.t
vlbu.v v24, (s4), v0.t

# CHECK: vlhu.v v25, (s6)
vlhu.v v25, (s6)
# CHECK: vlhu.v v26, (s8), v0.t
vlhu.v v26, (s8), v0.t

# CHECK: vlwu.v v27, (s10)
vlwu.v v27, (s10)
# CHECK: vlwu.v v28, (t3), v0.t
vlwu.v v28, (t3), v0.t

# CHECK: vle.v v29, (t5)
vle.v v29, (t5)
# CHECK: vle.v v30, (ra), v0.t
vle.v v30, (ra), v0.t

# CHECK: vsb.v v31, (gp)
vsb.v v31, (gp)
# CHECK: vsb.v v0, (t0), v0.t
vsb.v v0, (t0), v0.t

# CHECK: vsh.v v1, (t2)
vsh.v v1, (t2)
# CHECK: vsh.v v2, (s1), v0.t
vsh.v v2, (s1), v0.t

# CHECK: vsw.v v3, (a1)
vsw.v v3, (a1)
# CHECK: vsw.v v4, (a3), v0.t
vsw.v v4, (a3), v0.t

# CHECK: vse.v v5, (a5)
vse.v v5, (a5)
# CHECK: vse.v v6, (a7), v0.t
vse.v v6, (a7), v0.t

# CHECK: vlsb.v v7, (s5), s3
vlsb.v v7, (s5), s3
# CHECK: vlsb.v v8, (s9), s7, v0.t
vlsb.v v8, (s9), s7, v0.t

# CHECK: vlsh.v v9, (t4), s11
vlsh.v v9, (t4), s11
# CHECK: vlsh.v v10, (sp), t6, v0.t
vlsh.v v10, (sp), t6, v0.t

# CHECK: vlsw.v v11, (t1), tp
vlsw.v v11, (t1), tp
# CHECK: vlsw.v v12, (a0), s0, v0.t
vlsw.v v12, (a0), s0, v0.t

# CHECK: vlsbu.v v13, (a4), a2
vlsbu.v v13, (a4), a2
# CHECK: vlsbu.v v14, (s2), a6, v0.t
vlsbu.v v14, (s2), a6, v0.t

# CHECK: vlshu.v v15, (s6), s4
vlshu.v v15, (s6), s4
# CHECK: vlshu.v v16, (s10), s8, v0.t
vlshu.v v16, (s10), s8, v0.t

# CHECK: vlswu.v v17, (t5), t3
vlswu.v v17, (t5), t3
# CHECK: vlswu.v v18, (gp), ra, v0.t
vlswu.v v18, (gp), ra, v0.t

# CHECK: vlse.v v19, (t2), t0
vlse.v v19, (t2), t0
# CHECK: vlse.v v20, (a1), s1, v0.t
vlse.v v20, (a1), s1, v0.t

# CHECK: vssb.v v21, (a5), a3
vssb.v v21, (a5), a3
# CHECK: vssb.v v22, (s3), a7, v0.t
vssb.v v22, (s3), a7, v0.t

# CHECK: vssh.v v23, (s7), s5
vssh.v v23, (s7), s5
# CHECK: vssh.v v24, (s11), s9, v0.t
vssh.v v24, (s11), s9, v0.t

# CHECK: vssw.v v25, (t6), t4
vssw.v v25, (t6), t4
# CHECK: vssw.v v26, (tp), sp, v0.t
vssw.v v26, (tp), sp, v0.t

# CHECK: vsse.v v27, (s0), t1
vsse.v v27, (s0), t1
# CHECK: vsse.v v28, (a2), a0, v0.t
vsse.v v28, (a2), a0, v0.t

# CHECK: vlxb.v v30, (a4), v29
vlxb.v v30, (a4), v29
# CHECK: vlxb.v v0, (a6), v31, v0.t
vlxb.v v0, (a6), v31, v0.t

# CHECK: vlxh.v v2, (s2), v1
vlxh.v v2, (s2), v1
# CHECK: vlxh.v v4, (s4), v3, v0.t
vlxh.v v4, (s4), v3, v0.t

# CHECK: vlxw.v v6, (s6), v5
vlxw.v v6, (s6), v5
# CHECK: vlxw.v v8, (s8), v7, v0.t
vlxw.v v8, (s8), v7, v0.t

# CHECK: vlxbu.v v10, (s10), v9
vlxbu.v v10, (s10), v9
# CHECK: vlxbu.v v12, (t3), v11, v0.t
vlxbu.v v12, (t3), v11, v0.t

# CHECK: vlxhu.v v14, (t5), v13
vlxhu.v v14, (t5), v13
# CHECK: vlxhu.v v16, (ra), v15, v0.t
vlxhu.v v16, (ra), v15, v0.t

# CHECK: vlxwu.v v18, (gp), v17
vlxwu.v v18, (gp), v17
# CHECK: vlxwu.v v20, (t0), v19, v0.t
vlxwu.v v20, (t0), v19, v0.t

# CHECK: vlxe.v v22, (t2), v21
vlxe.v v22, (t2), v21
# CHECK: vlxe.v v24, (s1), v23, v0.t
vlxe.v v24, (s1), v23, v0.t

# CHECK: vsxb.v v26, (a1), v25
vsxb.v v26, (a1), v25
# CHECK: vsxb.v v28, (a3), v27, v0.t
vsxb.v v28, (a3), v27, v0.t

# CHECK: vsxh.v v30, (a5), v29
vsxh.v v30, (a5), v29
# CHECK: vsxh.v v0, (a7), v31, v0.t
vsxh.v v0, (a7), v31, v0.t

# CHECK: vsxw.v v2, (s3), v1
vsxw.v v2, (s3), v1
# CHECK: vsxw.v v4, (s5), v3, v0.t
vsxw.v v4, (s5), v3, v0.t

# CHECK: vsxe.v v6, (s7), v5
vsxe.v v6, (s7), v5
# CHECK: vsxe.v v8, (s9), v7, v0.t
vsxe.v v8, (s9), v7, v0.t

# CHECK: vsuxb.v v10, (s11), v9
vsuxb.v v10, (s11), v9
# CHECK: vsuxb.v v12, (t4), v11, v0.t
vsuxb.v v12, (t4), v11, v0.t

# CHECK: vsuxh.v v14, (t6), v13
vsuxh.v v14, (t6), v13
# CHECK: vsuxh.v v16, (sp), v15, v0.t
vsuxh.v v16, (sp), v15, v0.t

# CHECK: vsuxw.v v18, (tp), v17
vsuxw.v v18, (tp), v17
# CHECK: vsuxw.v v20, (t1), v19, v0.t
vsuxw.v v20, (t1), v19, v0.t

# CHECK: vsuxe.v v22, (s0), v21
vsuxe.v v22, (s0), v21
# CHECK: vsuxe.v v24, (a0), v23, v0.t
vsuxe.v v24, (a0), v23, v0.t

