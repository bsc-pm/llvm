; RUN: opt < %s -S -o - -mtriple riscv64 -mattr +m,+a,+f,+d,+v -gvn | \
; RUN:     FileCheck %s
; REQUIRES: riscv-registered-target

; CHECK-NOT: bitcast <vscale x 1 x i64> {{.*}} to i64

@test_double.a = internal global [64 x double] zeroinitializer, align 8
@test_double.b = internal global [64 x double] zeroinitializer, align 8
@test_double.mask_eq = internal unnamed_addr global [64 x i64] zeroinitializer, align 8
@test_double.mask_ne = internal unnamed_addr global [64 x i64] zeroinitializer, align 8
@test_double.mask_lt = internal unnamed_addr global [64 x i64] zeroinitializer, align 8
@test_double.mask_le = internal unnamed_addr global [64 x i64] zeroinitializer, align 8
@test_double.test_mask = internal unnamed_addr global [64 x i64] zeroinitializer, align 8
@.str = private unnamed_addr constant [10 x i8] c"gvl == VL\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"t.c\00", align 1
@__PRETTY_FUNCTION__.test_double = private unnamed_addr constant [23 x i8] c"void test_double(void)\00", align 1
@.str.2 = private unnamed_addr constant [27 x i8] c"test_mask[i] == mask_eq[i]\00", align 1
@.str.3 = private unnamed_addr constant [27 x i8] c"test_mask[i] == mask_ne[i]\00", align 1
@.str.4 = private unnamed_addr constant [27 x i8] c"test_mask[i] == mask_lt[i]\00", align 1
@.str.5 = private unnamed_addr constant [27 x i8] c"test_mask[i] == mask_le[i]\00", align 1

; Function Attrs: nounwind
define void @test_double() nounwind {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %0 = tail call i64 @llvm.epi.vsetvlmax(i64 3, i64 0)
  %conv38 = trunc i64 %0 to i32
  %cmp39 = icmp eq i32 %conv38, 64
  br i1 %cmp39, label %if.end, label %if.else

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv182 = phi i64 [ 0, %entry ], [ %indvars.iv.next183, %for.body ]
  %d.0172 = phi double [ -4.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [64 x double], [64 x double]* @test_double.a, i64 0, i64 %indvars.iv182
  store double %d.0172, double* %arrayidx, align 8
  %sub = fsub double -0.000000e+00, %d.0172
  %arrayidx2 = getelementptr inbounds [64 x double], [64 x double]* @test_double.b, i64 0, i64 %indvars.iv182
  store double %sub, double* %arrayidx2, align 8
  %cmp7 = fcmp oeq double %d.0172, %sub
  %conv8 = zext i1 %cmp7 to i64
  %arrayidx10 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_eq, i64 0, i64 %indvars.iv182
  store i64 %conv8, i64* %arrayidx10, align 8
  %cmp15 = fcmp une double %d.0172, %sub
  %conv17 = zext i1 %cmp15 to i64
  %arrayidx19 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_ne, i64 0, i64 %indvars.iv182
  store i64 %conv17, i64* %arrayidx19, align 8
  %cmp24 = fcmp olt double %d.0172, %sub
  %conv26 = zext i1 %cmp24 to i64
  %arrayidx28 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_lt, i64 0, i64 %indvars.iv182
  store i64 %conv26, i64* %arrayidx28, align 8
  %cmp33 = fcmp ole double %d.0172, %sub
  %conv35 = zext i1 %cmp33 to i64
  %arrayidx37 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_le, i64 0, i64 %indvars.iv182
  store i64 %conv35, i64* %arrayidx37, align 8
  %indvars.iv.next183 = add nuw nsw i64 %indvars.iv182, 1
  %add = fadd double %d.0172, 1.250000e-01
  %exitcond184 = icmp eq i64 %indvars.iv.next183, 64
  br i1 %exitcond184, label %for.cond.cleanup, label %for.body

if.else:                                          ; preds = %for.cond.cleanup
  tail call void @__assert_fail(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 signext 32, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.test_double, i64 0, i64 0)) #5
  unreachable

if.end:                                           ; preds = %for.cond.cleanup
  %1 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* bitcast ([64 x double]* @test_double.a to <vscale x 1 x double>*), i64 64)
  %2 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* bitcast ([64 x double]* @test_double.b to <vscale x 1 x double>*), i64 64)
  %3 = tail call <vscale x 1 x i1> @llvm.epi.vmfeq.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double> %1, <vscale x 1 x double> %2, i64 64)
  %frommask = zext <vscale x 1 x i1> %3 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %frommask, <vscale x 1 x i64>* bitcast ([64 x i64]* @test_double.test_mask to <vscale x 1 x i64>*), align 8
  br label %for.body49

for.cond45:                                       ; preds = %for.body49
  %exitcond181 = icmp eq i64 %indvars.iv.next180, 64
  br i1 %exitcond181, label %for.cond.cleanup48, label %for.body49

for.cond.cleanup48:                               ; preds = %for.cond45
  %4 = tail call <vscale x 1 x i1> @llvm.epi.vmfne.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double> %1, <vscale x 1 x double> %2, i64 64)
  %frommask63 = zext <vscale x 1 x i1> %4 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %frommask63, <vscale x 1 x i64>* bitcast ([64 x i64]* @test_double.test_mask to <vscale x 1 x i64>*), align 8
  br label %for.body70

for.body49:                                       ; preds = %for.cond45, %if.end
  %indvars.iv179 = phi i64 [ 0, %if.end ], [ %indvars.iv.next180, %for.cond45 ]
  %arrayidx51 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.test_mask, i64 0, i64 %indvars.iv179
  %5 = load i64, i64* %arrayidx51, align 8
  %arrayidx53 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_eq, i64 0, i64 %indvars.iv179
  %6 = load i64, i64* %arrayidx53, align 8
  %cmp54 = icmp eq i64 %5, %6
  %indvars.iv.next180 = add nuw nsw i64 %indvars.iv179, 1
  br i1 %cmp54, label %for.cond45, label %if.else57

if.else57:                                        ; preds = %for.body49
  tail call void @__assert_fail(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 signext 42, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.test_double, i64 0, i64 0)) #5
  unreachable

for.cond66:                                       ; preds = %for.body70
  %exitcond178 = icmp eq i64 %indvars.iv.next177, 64
  br i1 %exitcond178, label %for.cond.cleanup69, label %for.body70

for.cond.cleanup69:                               ; preds = %for.cond66
  %7 = tail call <vscale x 1 x i1> @llvm.epi.vmflt.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double> %1, <vscale x 1 x double> %2, i64 64)
  %frommask84 = zext <vscale x 1 x i1> %7 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %frommask84, <vscale x 1 x i64>* bitcast ([64 x i64]* @test_double.test_mask to <vscale x 1 x i64>*), align 8
  br label %for.body91

for.body70:                                       ; preds = %for.cond66, %for.cond.cleanup48
  %indvars.iv176 = phi i64 [ 0, %for.cond.cleanup48 ], [ %indvars.iv.next177, %for.cond66 ]
  %arrayidx72 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.test_mask, i64 0, i64 %indvars.iv176
  %8 = load i64, i64* %arrayidx72, align 8
  %arrayidx74 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_ne, i64 0, i64 %indvars.iv176
  %9 = load i64, i64* %arrayidx74, align 8
  %cmp75 = icmp eq i64 %8, %9
  %indvars.iv.next177 = add nuw nsw i64 %indvars.iv176, 1
  br i1 %cmp75, label %for.cond66, label %if.else78

if.else78:                                        ; preds = %for.body70
  tail call void @__assert_fail(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 signext 49, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.test_double, i64 0, i64 0)) #5
  unreachable

for.cond87:                                       ; preds = %for.body91
  %exitcond175 = icmp eq i64 %indvars.iv.next174, 64
  br i1 %exitcond175, label %for.cond.cleanup90, label %for.body91

for.cond.cleanup90:                               ; preds = %for.cond87
  %10 = tail call <vscale x 1 x i1> @llvm.epi.vmfle.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double> %1, <vscale x 1 x double> %2, i64 64)
  %frommask105 = zext <vscale x 1 x i1> %10 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %frommask105, <vscale x 1 x i64>* bitcast ([64 x i64]* @test_double.test_mask to <vscale x 1 x i64>*), align 8
  br label %for.body112

for.body91:                                       ; preds = %for.cond87, %for.cond.cleanup69
  %indvars.iv173 = phi i64 [ 0, %for.cond.cleanup69 ], [ %indvars.iv.next174, %for.cond87 ]
  %arrayidx93 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.test_mask, i64 0, i64 %indvars.iv173
  %11 = load i64, i64* %arrayidx93, align 8
  %arrayidx95 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_lt, i64 0, i64 %indvars.iv173
  %12 = load i64, i64* %arrayidx95, align 8
  %cmp96 = icmp eq i64 %11, %12
  %indvars.iv.next174 = add nuw nsw i64 %indvars.iv173, 1
  br i1 %cmp96, label %for.cond87, label %if.else99

if.else99:                                        ; preds = %for.body91
  tail call void @__assert_fail(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 signext 56, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.test_double, i64 0, i64 0)) #5
  unreachable

for.cond108:                                      ; preds = %for.body112
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.cond.cleanup111, label %for.body112

for.cond.cleanup111:                              ; preds = %for.cond108
  ret void

for.body112:                                      ; preds = %for.cond108, %for.cond.cleanup90
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup90 ], [ %indvars.iv.next, %for.cond108 ]
  %arrayidx114 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.test_mask, i64 0, i64 %indvars.iv
  %13 = load i64, i64* %arrayidx114, align 8
  %arrayidx116 = getelementptr inbounds [64 x i64], [64 x i64]* @test_double.mask_le, i64 0, i64 %indvars.iv
  %14 = load i64, i64* %arrayidx116, align 8
  %cmp117 = icmp eq i64 %13, %14
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %cmp117, label %for.cond108, label %if.else120

if.else120:                                       ; preds = %for.body112
  tail call void @__assert_fail(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.5, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 signext 63, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.test_double, i64 0, i64 0)) #5
  unreachable
}

; Function Attrs: nounwind
declare i64 @llvm.epi.vsetvlmax(i64, i64) #1

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32 signext, i8*) local_unnamed_addr #2

; Function Attrs: nounwind readonly
declare <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* nocapture, i64) #3

; Function Attrs: nounwind readnone
declare <vscale x 1 x i1> @llvm.epi.vmfeq.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64) #4

; Function Attrs: nounwind readnone
declare <vscale x 1 x i1> @llvm.epi.vmfne.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64) #4

; Function Attrs: nounwind readnone
declare <vscale x 1 x i1> @llvm.epi.vmflt.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64) #4

; Function Attrs: nounwind readnone
declare <vscale x 1 x i1> @llvm.epi.vmfle.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64) #4
