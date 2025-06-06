// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py UTC_ARGS: --include-generated-funcs
// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// LT → signed(<)    → 0
// LE → signed(<=)   → 1
// GT → signed(>)    → 2
// GE → signed(>=)   → 3
int sum = 0;
void signed_loop(int lb, int ub, int step) {
  #pragma oss taskloop
  for (int i = lb; i < ub; i += step)
      sum += i;
  #pragma oss taskloop
  for (int i = lb; i <= ub; i += step)
      sum += i;
  #pragma oss taskloop
  for (int i = ub; i > lb; i -= step)
      sum += i;
  #pragma oss taskloop
  for (int i = ub; i >= lb; i -= step)
      sum += i;
}

void unsigned_loop(unsigned lb, unsigned ub, unsigned step) {
  #pragma oss taskloop
  for (unsigned i = lb; i < ub; i += step)
      sum += i;
  #pragma oss taskloop
  for (unsigned i = lb; i <= ub; i += step)
      sum += i;
  #pragma oss taskloop
  for (unsigned i = ub; i > lb; i -= step)
      sum += i;
  #pragma oss taskloop
  for (unsigned i = ub; i >= lb; i -= step)
      sum += i;
}



























// CHECK-LABEL: @_Z11signed_loopiii(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I3:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I5:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[LB:%.*]], ptr [[LB_ADDR]], align 4
// CHECK-NEXT:    store i32 [[UB:%.*]], ptr [[UB_ADDR]], align 4
// CHECK-NEXT:    store i32 [[STEP:%.*]], ptr [[STEP_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB_ADDR]], align 4, !dbg [[DBG9:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[I]], align 4, !dbg [[DBG10:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 1, i64 1, i64 1, i64 1) ], !dbg [[DBG11:![0-9]+]]
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[I]], align 4, !dbg [[DBG12:![0-9]+]]
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG13:![0-9]+]]
// CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP3]], [[TMP2]], !dbg [[DBG13]]
// CHECK-NEXT:    store i32 [[ADD]], ptr @sum, align 4, !dbg [[DBG13]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP1]]), !dbg [[DBG14:![0-9]+]]
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[LB_ADDR]], align 4, !dbg [[DBG15:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP4]], ptr [[I1]], align 4, !dbg [[DBG16:![0-9]+]]
// CHECK-NEXT:    [[TMP5:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I1]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I1]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.1, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.2, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.3, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1) ], !dbg [[DBG17:![0-9]+]]
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[I1]], align 4, !dbg [[DBG18:![0-9]+]]
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG19:![0-9]+]]
// CHECK-NEXT:    [[ADD2:%.*]] = add nsw i32 [[TMP7]], [[TMP6]], !dbg [[DBG19]]
// CHECK-NEXT:    store i32 [[ADD2]], ptr @sum, align 4, !dbg [[DBG19]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP5]]), !dbg [[DBG20:![0-9]+]]
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[UB_ADDR]], align 4, !dbg [[DBG21:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP8]], ptr [[I3]], align 4, !dbg [[DBG22:![0-9]+]]
// CHECK-NEXT:    [[TMP9:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I3]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I3]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.4, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.5, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.6, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 1, i64 1, i64 1, i64 1) ], !dbg [[DBG23:![0-9]+]]
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[I3]], align 4, !dbg [[DBG24:![0-9]+]]
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG25:![0-9]+]]
// CHECK-NEXT:    [[ADD4:%.*]] = add nsw i32 [[TMP11]], [[TMP10]], !dbg [[DBG25]]
// CHECK-NEXT:    store i32 [[ADD4]], ptr @sum, align 4, !dbg [[DBG25]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP9]]), !dbg [[DBG26:![0-9]+]]
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, ptr [[UB_ADDR]], align 4, !dbg [[DBG27:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP12]], ptr [[I5]], align 4, !dbg [[DBG28:![0-9]+]]
// CHECK-NEXT:    [[TMP13:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I5]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I5]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.7, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.8, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.9, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 1, i64 1, i64 1, i64 1) ], !dbg [[DBG29:![0-9]+]]
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, ptr [[I5]], align 4, !dbg [[DBG30:![0-9]+]]
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG31:![0-9]+]]
// CHECK-NEXT:    [[ADD6:%.*]] = add nsw i32 [[TMP15]], [[TMP14]], !dbg [[DBG31]]
// CHECK-NEXT:    store i32 [[ADD6]], ptr @sum, align 4, !dbg [[DBG31]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP13]]), !dbg [[DBG32:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG33:![0-9]+]]
//
//
// CHECK-LABEL: @compute_lb(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG35:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG35]]
//
//
// CHECK-LABEL: @compute_ub(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG38:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG38]]
//
//
// CHECK-LABEL: @compute_step(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG41:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG41]]
//
//
// CHECK-LABEL: @compute_lb.1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG44:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG44]]
//
//
// CHECK-LABEL: @compute_ub.2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG47:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG47]]
//
//
// CHECK-LABEL: @compute_step.3(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG50:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG50]]
//
//
// CHECK-LABEL: @compute_lb.4(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG53:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG53]]
//
//
// CHECK-LABEL: @compute_ub.5(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG56:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG56]]
//
//
// CHECK-LABEL: @compute_step.6(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG59:![0-9]+]]
// CHECK-NEXT:    [[SUB:%.*]] = sub nsw i32 0, [[TMP0]], !dbg [[DBG59]]
// CHECK-NEXT:    ret i32 [[SUB]], !dbg [[DBG59]]
//
//
// CHECK-LABEL: @compute_lb.7(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG62:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG62]]
//
//
// CHECK-LABEL: @compute_ub.8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG65:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG65]]
//
//
// CHECK-LABEL: @compute_step.9(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG68:![0-9]+]]
// CHECK-NEXT:    [[SUB:%.*]] = sub nsw i32 0, [[TMP0]], !dbg [[DBG68]]
// CHECK-NEXT:    ret i32 [[SUB]], !dbg [[DBG68]]
//
//
// CHECK-LABEL: @_Z13unsigned_loopjjj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I3:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I5:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[LB:%.*]], ptr [[LB_ADDR]], align 4
// CHECK-NEXT:    store i32 [[UB:%.*]], ptr [[UB_ADDR]], align 4
// CHECK-NEXT:    store i32 [[STEP:%.*]], ptr [[STEP_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB_ADDR]], align 4, !dbg [[DBG71:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[I]], align 4, !dbg [[DBG72:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.10, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.11, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.12, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 0, i64 0, i64 0, i64 0, i64 0) ], !dbg [[DBG73:![0-9]+]]
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[I]], align 4, !dbg [[DBG74:![0-9]+]]
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG75:![0-9]+]]
// CHECK-NEXT:    [[ADD:%.*]] = add i32 [[TMP3]], [[TMP2]], !dbg [[DBG75]]
// CHECK-NEXT:    store i32 [[ADD]], ptr @sum, align 4, !dbg [[DBG75]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP1]]), !dbg [[DBG76:![0-9]+]]
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[LB_ADDR]], align 4, !dbg [[DBG77:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP4]], ptr [[I1]], align 4, !dbg [[DBG78:![0-9]+]]
// CHECK-NEXT:    [[TMP5:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I1]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I1]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.13, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.14, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.15, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 0, i64 0, i64 0, i64 0) ], !dbg [[DBG79:![0-9]+]]
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[I1]], align 4, !dbg [[DBG80:![0-9]+]]
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG81:![0-9]+]]
// CHECK-NEXT:    [[ADD2:%.*]] = add i32 [[TMP7]], [[TMP6]], !dbg [[DBG81]]
// CHECK-NEXT:    store i32 [[ADD2]], ptr @sum, align 4, !dbg [[DBG81]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP5]]), !dbg [[DBG82:![0-9]+]]
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[UB_ADDR]], align 4, !dbg [[DBG83:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP8]], ptr [[I3]], align 4, !dbg [[DBG84:![0-9]+]]
// CHECK-NEXT:    [[TMP9:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I3]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I3]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.16, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.17, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.18, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 2, i64 0, i64 0, i64 0, i64 0) ], !dbg [[DBG85:![0-9]+]]
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[I3]], align 4, !dbg [[DBG86:![0-9]+]]
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG87:![0-9]+]]
// CHECK-NEXT:    [[ADD4:%.*]] = add i32 [[TMP11]], [[TMP10]], !dbg [[DBG87]]
// CHECK-NEXT:    store i32 [[ADD4]], ptr @sum, align 4, !dbg [[DBG87]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP9]]), !dbg [[DBG88:![0-9]+]]
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, ptr [[UB_ADDR]], align 4, !dbg [[DBG89:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP12]], ptr [[I5]], align 4, !dbg [[DBG90:![0-9]+]]
// CHECK-NEXT:    [[TMP13:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.SHARED"(ptr @sum, i32 undef), "QUAL.OSS.PRIVATE"(ptr [[I5]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[UB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[LB_ADDR]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[STEP_ADDR]], i32 undef), "QUAL.OSS.LOOP.IND.VAR"(ptr [[I5]]), "QUAL.OSS.LOOP.LOWER.BOUND"(ptr @compute_lb.19, ptr [[UB_ADDR]]), "QUAL.OSS.LOOP.UPPER.BOUND"(ptr @compute_ub.20, ptr [[LB_ADDR]]), "QUAL.OSS.LOOP.STEP"(ptr @compute_step.21, ptr [[STEP_ADDR]]), "QUAL.OSS.LOOP.TYPE"(i64 3, i64 0, i64 0, i64 0, i64 0) ], !dbg [[DBG91:![0-9]+]]
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, ptr [[I5]], align 4, !dbg [[DBG92:![0-9]+]]
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr @sum, align 4, !dbg [[DBG93:![0-9]+]]
// CHECK-NEXT:    [[ADD6:%.*]] = add i32 [[TMP15]], [[TMP14]], !dbg [[DBG93]]
// CHECK-NEXT:    store i32 [[ADD6]], ptr @sum, align 4, !dbg [[DBG93]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP13]]), !dbg [[DBG94:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG95:![0-9]+]]
//
//
// CHECK-LABEL: @compute_lb.10(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG97:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG97]]
//
//
// CHECK-LABEL: @compute_ub.11(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG100:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG100]]
//
//
// CHECK-LABEL: @compute_step.12(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG103:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG103]]
//
//
// CHECK-LABEL: @compute_lb.13(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG106:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG106]]
//
//
// CHECK-LABEL: @compute_ub.14(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG109:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG109]]
//
//
// CHECK-LABEL: @compute_step.15(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG112:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG112]]
//
//
// CHECK-LABEL: @compute_lb.16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG115:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG115]]
//
//
// CHECK-LABEL: @compute_ub.17(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG118:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG118]]
//
//
// CHECK-LABEL: @compute_step.18(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG121:![0-9]+]]
// CHECK-NEXT:    [[SUB:%.*]] = sub i32 0, [[TMP0]], !dbg [[DBG121]]
// CHECK-NEXT:    ret i32 [[SUB]], !dbg [[DBG121]]
//
//
// CHECK-LABEL: @compute_lb.19(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[UB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[UB:%.*]], ptr [[UB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[UB]], align 4, !dbg [[DBG124:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG124]]
//
//
// CHECK-LABEL: @compute_ub.20(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LB_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[LB:%.*]], ptr [[LB_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[LB]], align 4, !dbg [[DBG127:![0-9]+]]
// CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG127]]
//
//
// CHECK-LABEL: @compute_step.21(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STEP_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[STEP:%.*]], ptr [[STEP_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[STEP]], align 4, !dbg [[DBG130:![0-9]+]]
// CHECK-NEXT:    [[SUB:%.*]] = sub i32 0, [[TMP0]], !dbg [[DBG130]]
// CHECK-NEXT:    ret i32 [[SUB]], !dbg [[DBG130]]
//
