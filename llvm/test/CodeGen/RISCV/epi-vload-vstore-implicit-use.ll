; RUN: llc -mtriple riscv64 -mattr=+epi -o /dev/null %s \
; RUN:     -print-after=riscv-expand-pseudo 2>&1 | FileCheck %s

define dso_local void @test_vload_8xi8_vstore_8xi8(i8* nocapture %addr, i64 %gvl) {
; CHECK:   $v0 = VLE_V $x10, implicit $vl, implicit $vtype
; CHECK:   VSE_V $v0, $x10, implicit $vl, implicit $vtype
entry:
  %0 = bitcast i8* %addr to <vscale x 8 x i8>*
  %1 = tail call <vscale x 8 x i8> @llvm.epi.vload.v8i8(<vscale x 8 x i8>* %0, i64 %gvl)
  tail call void @llvm.epi.vstore.v8i8(<vscale x 8 x i8> %1, <vscale x 8 x i8>* %0, i64 %gvl)
  ret void
}

declare <vscale x 8 x i8> @llvm.epi.vload.v8i8(<vscale x 8 x i8>* nocapture, i64) #4
declare void @llvm.epi.vstore.v8i8(<vscale x 8 x i8>, <vscale x 8 x i8>* nocapture, i64) #5
