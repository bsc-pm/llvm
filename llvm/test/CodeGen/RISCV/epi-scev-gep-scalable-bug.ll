;RUN: opt %s -o %t -loop-reduce -mtriple riscv64 -mattr +m,+a,+f,+d,+epi

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"

define dso_local void @t1sv_2(double* %ri, double* %ii, double* %W, i64 %rs, i64 %mb, i64 %me, i64 %ms) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %W.addr.097 = phi double* [ undef, %entry ], [ %add.ptr35, %for.body ]
  %0 = bitcast double* %W.addr.097 to <vscale x 1 x double>*
  %1 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %0, i64 8) #4
  %2 = tail call <vscale x 1 x double> @llvm.epi.vfmul.nxv1f64.nxv1f64(<vscale x 1 x double> %1, <vscale x 1 x double> zeroinitializer, i64 8)
  %arrayidx17 = getelementptr inbounds double, double* %W.addr.097, i64 8
  %3 = bitcast double* %arrayidx17 to <vscale x 1 x double>*
  %4 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* nonnull %3, i64 8) #4
  %5 = tail call <vscale x 1 x double> @llvm.epi.vfmacc.nxv1f64.nxv1f64(<vscale x 1 x double> %2, <vscale x 1 x double> %4, <vscale x 1 x double> undef, i64 8)
  %6 = tail call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> undef, <vscale x 1 x double> %5, i64 8)
  tail call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %6, <vscale x 1 x double>* undef, i64 8) #4
  %add.ptr35 = getelementptr inbounds double, double* %W.addr.097, i64 16
  br label %for.body
}

; Function Attrs: nounwind readnone
declare <vscale x 1 x double> @llvm.epi.vfmul.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64) #1

; Function Attrs: nounwind readnone
declare <vscale x 1 x double> @llvm.epi.vfmacc.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, <vscale x 1 x double>, i64) #1

; Function Attrs: nounwind readnone
declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64) #1

; Function Attrs: nounwind readonly
declare <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* nocapture, i64) #2

; Function Attrs: nounwind writeonly
declare void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>* nocapture, i64) #3
