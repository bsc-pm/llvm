; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+hard-float-double < %s | FileCheck %s

; -- float

define float @fma32(float %a, float %b, float %c) nounwind {
entry:
; CHECK-LABEL: fma32
; CHECK: fmadd.s  fa0, fa0, fa1, fa2
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %0
}

define float @fnma32(float %a, float %b, float %c) nounwind {
entry:
; CHECK-LABEL: fnma32
; CHECK: fnmadd.s  fa0, fa0, fa1, fa2
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c)
  %mul = fmul float %0, -1.000000e+00
  ret float %mul
}

define float @fms32(float %a, float %b, float %c) nounwind {
entry:
; CHECK-LABEL: fms32
; CHECK: fmsub.s  fa0, fa0, fa1, fa2
  %neg = fmul float %c, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %neg)
  ret float %0
}

define float @fnms32_1(float %a, float %b, float %c) nounwind {
entry:
; CHECK-LABEL: fnms32_1
; CHECK: fnmsub.s  fa0, fa0, fa1, fa2
  %neg = fmul float %a, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %neg, float %b, float %c)
  ret float %0
}

define float @fnms32_2(float %a, float %b, float %c) nounwind {
entry:
; CHECK-LABEL: fnms32_2
; CHECK: fnmsub.s  fa0, fa0, fa1, fa2
  %neg = fmul float %b, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %a, float %neg, float %c)
  ret float %0
}

; -- double

define double @fma64(double %a, double %b, double %c) nounwind {
entry:
; CHECK-LABEL: fma64
; CHECK: fmadd.d  fa0, fa0, fa1, fa2
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %0
}

define double @fnma64(double %a, double %b, double %c) nounwind {
entry:
; CHECK-LABEL: fnma64
; CHECK: fnmadd.d  fa0, fa0, fa1, fa2
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %c)
  %mul = fmul double %0, -1.000000e+00
  ret double %mul
}

define double @fms64(double %a, double %b, double %c) nounwind {
entry:
; CHECK-LABEL: fms64
; CHECK: fmsub.d  fa0, fa0, fa1, fa2
  %neg = fmul double %c, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %neg)
  ret double %0
}

define double @fnms64_1(double %a, double %b, double %c) nounwind {
entry:
; CHECK-LABEL: fnms64_1
; CHECK: fnmsub.d  fa0, fa0, fa1, fa2
  %neg = fmul double %a, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %neg, double %b, double %c)
  ret double %0
}

define double @fnms64_2(double %a, double %b, double %c) nounwind {
entry:
; CHECK-LABEL: fnms64_2
; CHECK: fnmsub.d  fa0, fa0, fa1, fa2
  %neg = fmul double %b, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %a, double %neg, double %c)
  ret double %0
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone
