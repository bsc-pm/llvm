// UNSUPPORTED: riscv
// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -emit-llvm -w -o- %s \
// RUN:   -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +epi  -Werror=implicit-function-declaration \
// RUN:   | FileCheck %s

void builtins_config() {
  unsigned long long l;

  l = __builtin_epi_vl();
  // CHECK: call i64 @llvm.epi.vl()

  l = __builtin_epi_maxvl();
  // CHECK: call i64 @llvm.epi.maxvl()

  l = __builtin_epi_setvl(l);
  // CHECK: call i64 @llvm.epi.setvl(i64 %{{.*}})

  __builtin_epi_vconfig(1, 2, 3, 4);
  // CHECK: call void @llvm.epi.vconfig(i64 {{.*}})
}

void builtins_i1() {
  __epi_bool v1, v2, v3, v4;
  unsigned long l;
  _Bool b;

  // Binary
  v1 = __builtin_epi_vand_i1(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vand.i1(<vscale x 1 x i1> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vor_i1(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vor.i1(<vscale x 1 x i1> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vxor_i1(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vxor.i1(<vscale x 1 x i1> {{.*}}, <vscale x 1 x i1> {{.*}})

  // Immediates
  v1 = __builtin_epi_vandi_i1(v2, 1);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vandi.i1(<vscale x 1 x i1> {{.*}}, i64 1)

  v1 = __builtin_epi_vori_i1(v2, 1);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vori.i1(<vscale x 1 x i1> {{.*}}, i64 1)

  v1 = __builtin_epi_vxori_i1(v2, 1);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vxori.i1(<vscale x 1 x i1> {{.*}}, i64 1)

  // Other

  v1 = __builtin_epi_vinsert_i1(v2, 1, 4);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vinsert.i1(<vscale x 1 x i1> {{.*}}, i64 -1, i64 4)

  b = __builtin_epi_vextract_i1(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i1(<vscale x 1 x i1> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_i1(v2, v3, v4);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vmerge.i1(<vscale x 1 x i1> {{.*}}, <vscale x 1 x i1> {{.*}}, <vscale x 1 x i1> {{.*}})

  l = __builtin_epi_vmpopc_i1(v1);
  // CHECK: call i64 @llvm.epi.vmpopc.i1(<vscale x 1 x i1> {{.*}})
  l = __builtin_epi_vmfirst_i1(v1);
  // CHECK: call i64 @llvm.epi.vmfirst.i1(<vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vslideup_i1(v2, 4);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vslideup.i1(<vscale x 1 x i1> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_i1(v2, 4);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vslidedown.i1(<vscale x 1 x i1> {{.*}}, i64 4)
}

void builtins_i8() {
  __epi_byte v1, v2, v3;
  __epi_bool vm;
  signed char s;
  unsigned char u;
  unsigned long l;

  v1 = __builtin_epi_vadd_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vadd.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vsub_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vsub.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vsll_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vsll.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vsrl_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vsrl.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vsra_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vsra.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vand_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vand.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vor_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vor.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vxor_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vxor.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vdiv_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vdiv.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vdivu_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vdivu.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vrem_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vrem.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vremu_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vremu.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vmul_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vmul.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vmulh_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vmulh.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vmulhu_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vmulhu.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vmulhsu_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vmulhsu.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  // Immediates

  v1 = __builtin_epi_vaddi_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vaddi.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  v1 = __builtin_epi_vslli_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vslli.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrli_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vsrli.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrai_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vsrai.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  v1 = __builtin_epi_vandi_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vandi.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  v1 = __builtin_epi_vori_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vori.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  v1 = __builtin_epi_vxori_i8(v2, 42);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vxori.i8(<vscale x 1 x i8> {{.*}}, i64 42)

  // Relationals

  vm = __builtin_epi_vseq_i8(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vseq.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  vm = __builtin_epi_vsne_i8(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsne.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  vm = __builtin_epi_vslt_i8(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vslt.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  vm = __builtin_epi_vsge_i8(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsge.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  vm = __builtin_epi_vsltu_i8(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsltu.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  vm = __builtin_epi_vsgeu_i8(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsgeu.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  // Unary
  v1 = __builtin_epi_vneg_i8(v2);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vneg.i8(<vscale x 1 x i8> {{.*}})

  // Other

  v1 = __builtin_epi_vinsert_i8(v2, 42, 4);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vinsert.i8(<vscale x 1 x i8> {{.*}}, i64 42, i64 4)

  v1 = __builtin_epi_vinsert_unsigned_i8(v2, 42, 4);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vinsert.i8(<vscale x 1 x i8> {{.*}}, i64 42, i64 4)

  s = __builtin_epi_vextract_i8(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i8(<vscale x 1 x i8> {{.*}}, i64 4)

  u = __builtin_epi_vextract_unsigned_i8(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i8(<vscale x 1 x i8> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_i8(v2, v3, vm);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vmerge.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vredsum_i8(v2);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vredsum.i8(<vscale x 1 x i8> {{.*}})
  v1 = __builtin_epi_vredmax_i8(v2);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vredmax.i8(<vscale x 1 x i8> {{.*}})
  v1 = __builtin_epi_vredmin_i8(v2);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vredmin.i8(<vscale x 1 x i8> {{.*}})

  l = __builtin_epi_vmpopc_i8(v1);
  // CHECK: call i64 @llvm.epi.vmpopc.i8(<vscale x 1 x i8> {{.*}})
  l = __builtin_epi_vmfirst_i8(v1);
  // CHECK: call i64 @llvm.epi.vmfirst.i8(<vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vselect_i8();
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vselect.i8()
  v1 = __builtin_epi_vselect_i8_true(vm);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vselect.i8.true(<vscale x 1 x i1> {{.*}})
  v1 = __builtin_epi_vselect_i8_false(vm);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vselect.i8.false(<vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vrgather_i8_i8(v2, v3);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vrgather.i8.i8(<vscale x 1 x i8> {{.*}}, <vscale x 1 x i8> {{.*}})

  v1 = __builtin_epi_vslideup_i8(v2, 4);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vslideup.i8(<vscale x 1 x i8> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_i8(v2, 4);
  // CHECK: call <vscale x 1 x i8> @llvm.epi.vslidedown.i8(<vscale x 1 x i8> {{.*}}, i64 4)
}

void builtins_i16() {
  __epi_short v1, v2, v3;
  __epi_bool vm;
  signed short s;
  unsigned short u;
  unsigned long l;

  v1 = __builtin_epi_vadd_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vadd.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vsub_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vsub.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vsll_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vsll.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vsrl_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vsrl.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vsra_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vsra.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vand_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vand.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vor_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vor.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vxor_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vxor.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vdiv_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vdiv.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vdivu_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vdivu.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vrem_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vrem.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vremu_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vremu.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vmul_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vmul.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vmulh_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vmulh.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vmulhu_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vmulhu.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vmulhsu_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vmulhsu.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  // Immediates

  v1 = __builtin_epi_vaddi_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vaddi.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  v1 = __builtin_epi_vslli_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vslli.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrli_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vsrli.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrai_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vsrai.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  v1 = __builtin_epi_vandi_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vandi.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  v1 = __builtin_epi_vori_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vori.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  v1 = __builtin_epi_vxori_i16(v2, 42);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vxori.i16(<vscale x 1 x i16> {{.*}}, i64 42)

  // Relationals

  vm = __builtin_epi_vseq_i16(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vseq.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  vm = __builtin_epi_vsne_i16(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsne.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  vm = __builtin_epi_vslt_i16(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vslt.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  vm = __builtin_epi_vsge_i16(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsge.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  vm = __builtin_epi_vsltu_i16(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsltu.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  vm = __builtin_epi_vsgeu_i16(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsgeu.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  // Unary
  v1 = __builtin_epi_vneg_i16(v2);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vneg.i16(<vscale x 1 x i16> {{.*}})

  // Other

  v1 = __builtin_epi_vinsert_i16(v2, 42, 4);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vinsert.i16(<vscale x 1 x i16> {{.*}}, i64 42, i64 4)

  v1 = __builtin_epi_vinsert_unsigned_i16(v2, 42, 4);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vinsert.i16(<vscale x 1 x i16> {{.*}}, i64 42, i64 4)

  s = __builtin_epi_vextract_i16(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i16(<vscale x 1 x i16> {{.*}}, i64 4)

  u = __builtin_epi_vextract_unsigned_i16(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i16(<vscale x 1 x i16> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_i16(v2, v3, vm);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vmerge.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vredsum_i16(v2);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vredsum.i16(<vscale x 1 x i16> {{.*}})
  v1 = __builtin_epi_vredmax_i16(v2);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vredmax.i16(<vscale x 1 x i16> {{.*}})
  v1 = __builtin_epi_vredmin_i16(v2);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vredmin.i16(<vscale x 1 x i16> {{.*}})

  l = __builtin_epi_vmpopc_i16(v1);
  // CHECK: call i64 @llvm.epi.vmpopc.i16(<vscale x 1 x i16> {{.*}})
  l = __builtin_epi_vmfirst_i16(v1);
  // CHECK: call i64 @llvm.epi.vmfirst.i16(<vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vselect_i16();
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vselect.i16()
  v1 = __builtin_epi_vselect_i16_true(vm);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vselect.i16.true(<vscale x 1 x i1> {{.*}})
  v1 = __builtin_epi_vselect_i16_false(vm);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vselect.i16.false(<vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vrgather_i16_i16(v2, v3);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vrgather.i16.i16(<vscale x 1 x i16> {{.*}}, <vscale x 1 x i16> {{.*}})

  v1 = __builtin_epi_vslideup_i16(v2, 4);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vslideup.i16(<vscale x 1 x i16> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_i16(v2, 4);
  // CHECK: call <vscale x 1 x i16> @llvm.epi.vslidedown.i16(<vscale x 1 x i16> {{.*}}, i64 4)
}

void builtins_i32() {
  __epi_int v1, v2, v3;
  __epi_bool vm;
  signed int s;
  unsigned int u;
  unsigned long l;

  v1 = __builtin_epi_vadd_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vadd.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vsub_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vsub.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vsll_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vsll.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vsrl_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vsrl.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vsra_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vsra.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vand_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vand.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vor_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vor.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vxor_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vxor.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vdiv_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vdiv.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vdivu_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vdivu.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vrem_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vrem.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vremu_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vremu.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vmul_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vmul.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vmulh_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vmulh.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vmulhu_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vmulhu.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vmulhsu_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vmulhsu.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  // Immediates

  v1 = __builtin_epi_vaddi_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vaddi.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  v1 = __builtin_epi_vslli_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vslli.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrli_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vsrli.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrai_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vsrai.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  v1 = __builtin_epi_vandi_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vandi.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  v1 = __builtin_epi_vori_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vori.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  v1 = __builtin_epi_vxori_i32(v2, 42);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vxori.i32(<vscale x 1 x i32> {{.*}}, i64 42)

  // Relationals

  vm = __builtin_epi_vseq_i32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vseq.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  vm = __builtin_epi_vsne_i32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsne.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  vm = __builtin_epi_vslt_i32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vslt.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  vm = __builtin_epi_vsge_i32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsge.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  vm = __builtin_epi_vsltu_i32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsltu.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  vm = __builtin_epi_vsgeu_i32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsgeu.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  // Unary
  v1 = __builtin_epi_vneg_i32(v2);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vneg.i32(<vscale x 1 x i32> {{.*}})

  // Other

  v1 = __builtin_epi_vinsert_i32(v2, 42, 4);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vinsert.i32(<vscale x 1 x i32> {{.*}}, i64 42, i64 4)

  v1 = __builtin_epi_vinsert_unsigned_i32(v2, 42, 4);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vinsert.i32(<vscale x 1 x i32> {{.*}}, i64 42, i64 4)

  s = __builtin_epi_vextract_i32(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i32(<vscale x 1 x i32> {{.*}}, i64 4)

  u = __builtin_epi_vextract_unsigned_i32(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i32(<vscale x 1 x i32> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_i32(v2, v3, vm);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vmerge.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vredsum_i32(v2);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vredsum.i32(<vscale x 1 x i32> {{.*}})
  v1 = __builtin_epi_vredmax_i32(v2);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vredmax.i32(<vscale x 1 x i32> {{.*}})
  v1 = __builtin_epi_vredmin_i32(v2);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vredmin.i32(<vscale x 1 x i32> {{.*}})

  l = __builtin_epi_vmpopc_i32(v1);
  // CHECK: call i64 @llvm.epi.vmpopc.i32(<vscale x 1 x i32> {{.*}})
  l = __builtin_epi_vmfirst_i32(v1);
  // CHECK: call i64 @llvm.epi.vmfirst.i32(<vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vselect_i32();
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vselect.i32()
  v1 = __builtin_epi_vselect_i32_true(vm);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vselect.i32.true(<vscale x 1 x i1> {{.*}})
  v1 = __builtin_epi_vselect_i32_false(vm);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vselect.i32.false(<vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vrgather_i32_i32(v2, v3);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vrgather.i32.i32(<vscale x 1 x i32> {{.*}}, <vscale x 1 x i32> {{.*}})

  v1 = __builtin_epi_vslideup_i32(v2, 4);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vslideup.i32(<vscale x 1 x i32> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_i32(v2, 4);
  // CHECK: call <vscale x 1 x i32> @llvm.epi.vslidedown.i32(<vscale x 1 x i32> {{.*}}, i64 4)
}

void builtins_i64() {
  __epi_long v1, v2, v3;
  __epi_bool vm;
  unsigned long u, l;
  signed long s;

  v1 = __builtin_epi_vadd_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vadd.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vsub_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vsub.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vsll_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vsll.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vsrl_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vsrl.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vsra_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vsra.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vand_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vand.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vor_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vor.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vxor_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vxor.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vdiv_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vdiv.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vdivu_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vdivu.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vrem_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vrem.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vremu_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vremu.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vmul_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vmul.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vmulh_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vmulh.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vmulhu_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vmulhu.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vmulhsu_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vmulhsu.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  // Immediates

  v1 = __builtin_epi_vaddi_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vaddi.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  v1 = __builtin_epi_vslli_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vslli.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrli_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vsrli.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  v1 = __builtin_epi_vsrai_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vsrai.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  v1 = __builtin_epi_vandi_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vandi.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  v1 = __builtin_epi_vori_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vori.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  v1 = __builtin_epi_vxori_i64(v2, 42);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vxori.i64(<vscale x 1 x i64> {{.*}}, i64 42)

  // Relationals

  vm = __builtin_epi_vseq_i64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vseq.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  vm = __builtin_epi_vsne_i64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsne.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  vm = __builtin_epi_vslt_i64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vslt.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  vm = __builtin_epi_vsge_i64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsge.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  vm = __builtin_epi_vsltu_i64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsltu.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  vm = __builtin_epi_vsgeu_i64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vsgeu.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  // Unary
  v1 = __builtin_epi_vneg_i64(v2);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vneg.i64(<vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vinsert_i64(v2, 42, 4);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vinsert.i64(<vscale x 1 x i64> {{.*}}, i64 42, i64 4)

  // Other

  v1 = __builtin_epi_vinsert_i64(v2, 42, 4);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vinsert.i64(<vscale x 1 x i64> {{.*}}, i64 42, i64 4)

  v1 = __builtin_epi_vinsert_unsigned_i64(v2, 42, 4);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vinsert.i64(<vscale x 1 x i64> {{.*}}, i64 42, i64 4)

  s = __builtin_epi_vextract_i64(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i64(<vscale x 1 x i64> {{.*}}, i64 4)

  u = __builtin_epi_vextract_unsigned_i64(v2, 4);
  // CHECK: call i64 @llvm.epi.vextract.i64(<vscale x 1 x i64> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_i64(v2, v3, vm);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vmerge.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vredsum_i64(v2);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vredsum.i64(<vscale x 1 x i64> {{.*}})
  v1 = __builtin_epi_vredmax_i64(v2);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vredmax.i64(<vscale x 1 x i64> {{.*}})
  v1 = __builtin_epi_vredmin_i64(v2);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vredmin.i64(<vscale x 1 x i64> {{.*}})

  l = __builtin_epi_vmpopc_i64(v1);
  // CHECK: call i64 @llvm.epi.vmpopc.i64(<vscale x 1 x i64> {{.*}})
  l = __builtin_epi_vmfirst_i64(v1);
  // CHECK: call i64 @llvm.epi.vmfirst.i64(<vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vselect_i64();
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vselect.i64()
  v1 = __builtin_epi_vselect_i64_true(vm);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vselect.i64.true(<vscale x 1 x i1> {{.*}})
  v1 = __builtin_epi_vselect_i64_false(vm);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vselect.i64.false(<vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vrgather_i64_i64(v2, v3);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vrgather.i64.i64(<vscale x 1 x i64> {{.*}}, <vscale x 1 x i64> {{.*}})

  v1 = __builtin_epi_vslideup_i64(v2, 4);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vslideup.i64(<vscale x 1 x i64> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_i64(v2, 4);
  // CHECK: call <vscale x 1 x i64> @llvm.epi.vslidedown.i64(<vscale x 1 x i64> {{.*}}, i64 4)
}

void builtins_f32() {
  __epi_float v1, v2, v3, v4;
  __epi_bool vm;
  float f;

  v1 = __builtin_epi_vfadd_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfadd.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfsub_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfsub.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfmul_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfmul.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfdiv_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfdiv.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfsgnj_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfsgnj.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfsgnjn_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfsgnjn.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfsgnjx_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfsgnjx.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfmin_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfmin.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfmax_f32(v2, v3);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfmax.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  // Relational

  vm = __builtin_epi_vfeq_f32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vfeq.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  vm = __builtin_epi_vfne_f32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vfne.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  vm = __builtin_epi_vflt_f32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vflt.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  vm = __builtin_epi_vfle_f32(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vfle.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  // Unary

  v1 = __builtin_epi_vfsqrt_f32(v2);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfsqrt.f32(<vscale x 1 x float> {{.*}})


  // Float-multiply

  v1 = __builtin_epi_vfmadd_f32(v2, v3, v4);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfmadd.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vfmsub_f32(v2, v3, v4);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfmsub.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}})


  // Other

  v1 = __builtin_epi_vfinsert_f32(v2, 42.0f, 4);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfinsert.f32(<vscale x 1 x float> {{.*}}, float {{.*}}, i64 4)

  f = __builtin_epi_vfextract_f32(v2, 4);
  // CHECK: call float @llvm.epi.vfextract.f32(<vscale x 1 x float> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_f32(v2, v3, vm);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vmerge.f32(<vscale x 1 x float> {{.*}}, <vscale x 1 x float> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vfredsum_f32(v2);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfredsum.f32(<vscale x 1 x float> {{.*}})
  v1 = __builtin_epi_vfredmax_f32(v2);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfredmax.f32(<vscale x 1 x float> {{.*}})
  v1 = __builtin_epi_vfredmin_f32(v2);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vfredmin.f32(<vscale x 1 x float> {{.*}})

  v1 = __builtin_epi_vslideup_f32(v2, 4);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vslideup.f32(<vscale x 1 x float> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_f32(v2, 4);
  // CHECK: call <vscale x 1 x float> @llvm.epi.vslidedown.f32(<vscale x 1 x float> {{.*}}, i64 4)
}

void builtins_f64() {
  __epi_double v1, v2, v3, v4;
  __epi_bool vm;
  double d;

  v1 = __builtin_epi_vfadd_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfadd.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfsub_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfsub.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfmul_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfmul.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfdiv_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfdiv.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfsgnj_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfsgnj.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfsgnjn_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfsgnjn.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfsgnjx_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfsgnjx.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfmin_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfmin.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfmax_f64(v2, v3);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfmax.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  // Relational

  vm = __builtin_epi_vfeq_f64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vfeq.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  vm = __builtin_epi_vfne_f64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vfne.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  vm = __builtin_epi_vflt_f64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vflt.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  vm = __builtin_epi_vfle_f64(v2, v3);
  // CHECK: call <vscale x 1 x i1> @llvm.epi.vfle.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  // Unary

  v1 = __builtin_epi_vfsqrt_f64(v2);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfsqrt.f64(<vscale x 1 x double> {{.*}})

  // Float-multiply

  v1 = __builtin_epi_vfmadd_f64(v2, v3, v4);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfmadd.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vfmsub_f64(v2, v3, v4);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfmsub.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}})

  // Other

  v1 = __builtin_epi_vfinsert_f64(v2, 42.0, 4);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfinsert.f64(<vscale x 1 x double> {{.*}}, double {{.*}}, i64 4)

  d = __builtin_epi_vfextract_f64(v2, 4);
  // CHECK: call double @llvm.epi.vfextract.f64(<vscale x 1 x double> {{.*}}, i64 4)

  v1 = __builtin_epi_vmerge_f64(v2, v3, vm);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vmerge.f64(<vscale x 1 x double> {{.*}}, <vscale x 1 x double> {{.*}}, <vscale x 1 x i1> {{.*}})

  v1 = __builtin_epi_vfredsum_f64(v2);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfredsum.f64(<vscale x 1 x double> {{.*}})
  v1 = __builtin_epi_vfredmax_f64(v2);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfredmax.f64(<vscale x 1 x double> {{.*}})
  v1 = __builtin_epi_vfredmin_f64(v2);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vfredmin.f64(<vscale x 1 x double> {{.*}})

  v1 = __builtin_epi_vslideup_f64(v2, 4);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vslideup.f64(<vscale x 1 x double> {{.*}}, i64 4)
  v1 = __builtin_epi_vslidedown_f64(v2, 4);
  // CHECK: call <vscale x 1 x double> @llvm.epi.vslidedown.f64(<vscale x 1 x double> {{.*}}, i64 4)
}
