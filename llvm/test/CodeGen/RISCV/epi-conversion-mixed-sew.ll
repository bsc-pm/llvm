; RUN: llc -mtriple=riscv64 -mattr=+epi,+a,+f,+d,+c,+m -o - %s | FileCheck %s

; Note: This test uses the vector calling convention which is subject to change.

; Widening float->uint
declare <vscale x 2 x i64> @llvm.epi.vfwcvt.xu.f.nxv2i64.nxv2f32(<vscale x 2 x float>, i64);

define <vscale x 2 x i64> @test_widen_float_to_uint(<vscale x 2 x float> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_float_to_uint
; CHECK:     vfwcvt.xu.f.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x i64> @llvm.epi.vfwcvt.xu.f.nxv2i64.nxv2f32(
    <vscale x 2 x float> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i64> %a
}

; Widening float->int
declare <vscale x 2 x i64> @llvm.epi.vfwcvt.x.f.nxv2i64.nxv2f32(<vscale x 2 x float>, i64);

define <vscale x 2 x i64> @test_widen_float_to_int(<vscale x 2 x float> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_float_to_int
; CHECK:     vfwcvt.x.f.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x i64> @llvm.epi.vfwcvt.x.f.nxv2i64.nxv2f32(
    <vscale x 2 x float> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i64> %a
}

; Widening uint->float
declare <vscale x 2 x double> @llvm.epi.vfwcvt.f.xu.nxv2f64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x double> @test_widen_uint_to_float(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_uint_to_float
; CHECK:     vfwcvt.f.xu.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x double> @llvm.epi.vfwcvt.f.xu.nxv2f64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x double> %a
}

; Widening int->float
declare <vscale x 2 x double> @llvm.epi.vfwcvt.f.x.nxv2f64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x double> @test_widen_int_to_float(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_int_to_float
; CHECK:     vfwcvt.f.x.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x double> @llvm.epi.vfwcvt.f.x.nxv2f64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x double> %a
}

; Widening float->float
declare <vscale x 2 x double> @llvm.epi.vfwcvt.f.f.nxv2f64.nxv2f32( <vscale x 2 x float>, i64);

define <vscale x 2 x double> @test_widen_float_to_float(<vscale x 2 x float> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_float_to_float
; CHECK:     vfwcvt.f.f.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x double> @llvm.epi.vfwcvt.f.f.nxv2f64.nxv2f32(
    <vscale x 2 x float> %parm0,
    i64 %gvl)

  ret <vscale x 2 x double> %a
}

; Widening int->uint
declare <vscale x 2 x i64> @llvm.epi.vwcvt.xu.x.nxv2i64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x i64> @test_widen_int_to_uint(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_int_to_uint
; CHECK:     vwaddu.vx
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]], zero
  %a = call <vscale x 2 x i64> @llvm.epi.vwcvt.xu.x.nxv2i64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x i64> %a
}

; Widening int->int
declare <vscale x 2 x i64> @llvm.epi.vwcvt.x.x.nxv2i64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x i64> @test_widen_int_to_int(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_widen_int_to_int
; CHECK:     vwadd.vx
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]], zero
  %a = call <vscale x 2 x i64> @llvm.epi.vwcvt.x.x.nxv2i64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x i64> %a
}

; Narrowing float->uint
declare <vscale x 2 x i32> @llvm.epi.vfncvt.xu.f.nxv2i32.nxv2f64(<vscale x 2 x double>, i64);

define <vscale x 2 x i32> @test_narrow_float_to_uint(<vscale x 2 x double> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_narrow_float_to_uint
; CHECK:     vfncvt.xu.f.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x i32> @llvm.epi.vfncvt.xu.f.nxv2i32.nxv2f64(
    <vscale x 2 x double> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i32> %a
}

; Narrowing float->int
declare <vscale x 2 x i32> @llvm.epi.vfncvt.x.f.nxv2i32.nxv2f64(<vscale x 2 x double>, i64);

define <vscale x 2 x i32> @test_narrow_float_to_int(<vscale x 2 x double> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_narrow_float_to_int
; CHECK:     vfncvt.x.f.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x i32> @llvm.epi.vfncvt.x.f.nxv2i32.nxv2f64(
    <vscale x 2 x double> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i32> %a
}

; Narrowing uint->float
declare <vscale x 2 x float> @llvm.epi.vfncvt.f.xu.nxv2f32.nxv2i64( <vscale x 2 x i64>, i64);

define <vscale x 2 x float> @test_narrow_uint_to_float(<vscale x 2 x i64> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_narrow_uint_to_float
; CHECK:     vfncvt.f.xu.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x float> @llvm.epi.vfncvt.f.xu.nxv2f32.nxv2i64(
    <vscale x 2 x i64> %parm0,
    i64 %gvl)

  ret <vscale x 2 x float> %a
}

; Narrowing int->float
declare <vscale x 2 x float> @llvm.epi.vfncvt.f.x.nxv2f32.nxv2i64( <vscale x 2 x i64>, i64);

define <vscale x 2 x float> @test_narrow_int_to_float(<vscale x 2 x i64> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_narrow_int_to_float
; CHECK:     vfncvt.f.x.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x float> @llvm.epi.vfncvt.f.x.nxv2f32.nxv2i64(
    <vscale x 2 x i64> %parm0,
    i64 %gvl)

  ret <vscale x 2 x float> %a
}

; Narrowing float->float
declare <vscale x 2 x float> @llvm.epi.vfncvt.f.f.nxv2f32.nxv2f64( <vscale x 2 x double>, i64);

define <vscale x 2 x float> @test_narrow_float_to_float(<vscale x 2 x double> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_narrow_float_to_float
; CHECK:     vfncvt.f.f.v
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]], zero
  %a = call <vscale x 2 x float> @llvm.epi.vfncvt.f.f.nxv2f32.nxv2f64(
    <vscale x 2 x double> %parm0,
    i64 %gvl)

  ret <vscale x 2 x float> %a
}

; Narrowing int->int
declare <vscale x 2 x i32> @llvm.epi.vncvt.x.x.nxv2i32.nxv2i64( <vscale x 2 x i64>, i64);

define <vscale x 2 x i32> @test_narrow_int_to_int(<vscale x 2 x i64> %parm0, i64 %gvl) nounwind {
entry:
; CHECK-LABEL: test_narrow_int_to_int
; CHECK:     vnsrl.vx
; CHECK-NOT: [[DEST:v[0-9]+]], [[DEST]]
  %a = call <vscale x 2 x i32> @llvm.epi.vncvt.x.x.nxv2i32.nxv2i64(
    <vscale x 2 x i64> %parm0,
    i64 %gvl)

  ret <vscale x 2 x i32> %a
}
