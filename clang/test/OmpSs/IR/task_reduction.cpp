// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py UTC_ARGS: --include-generated-funcs
// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int &rx) {
    #pragma oss task reduction(+: rx)
    {}
}





// CHECK-LABEL: @_Z3fooRi(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RX_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[RX:%.*]], ptr [[RX_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[RX_ADDR]], align 8, !dbg [[DBG9:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr [[TMP0]], i32 undef), "QUAL.OSS.DEP.REDUCTION"(i32 6000, ptr [[TMP0]], [3 x i8] c"rx\00", ptr @compute_dep, ptr [[TMP0]]), "QUAL.OSS.DEP.REDUCTION.INIT"(ptr [[TMP0]], ptr @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(ptr [[TMP0]], ptr @red_comb) ], !dbg [[DBG9]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP1]]), !dbg [[DBG10:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG11:![0-9]+]]
//
//
// CHECK-LABEL: @red_init(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTADDR2:%.*]] = alloca i64, align 8
// CHECK-NEXT:    store ptr [[TMP0:%.*]], ptr [[DOTADDR]], align 8
// CHECK-NEXT:    store ptr [[TMP1:%.*]], ptr [[DOTADDR1]], align 8
// CHECK-NEXT:    store i64 [[TMP2:%.*]], ptr [[DOTADDR2]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK-NEXT:    [[TMP5:%.*]] = load i64, ptr [[DOTADDR2]], align 8
// CHECK-NEXT:    [[TMP6:%.*]] = udiv exact i64 [[TMP5]], 4
// CHECK-NEXT:    [[ARRAYCTOR_DST_END:%.*]] = getelementptr inbounds i32, ptr [[TMP3]], i64 [[TMP6]]
// CHECK-NEXT:    br label [[ARRAYCTOR_LOOP:%.*]]
// CHECK:       arrayctor.loop:
// CHECK-NEXT:    [[ARRAYCTOR_DST_CUR:%.*]] = phi ptr [ [[TMP3]], [[ENTRY:%.*]] ], [ [[ARRAYCTOR_DST_NEXT:%.*]], [[ARRAYCTOR_LOOP]] ]
// CHECK-NEXT:    [[ARRAYCTOR_SRC_CUR:%.*]] = phi ptr [ [[TMP4]], [[ENTRY]] ], [ [[ARRAYCTOR_SRC_NEXT:%.*]], [[ARRAYCTOR_LOOP]] ]
// CHECK-NEXT:    store i32 0, ptr [[ARRAYCTOR_DST_CUR]], align 4
// CHECK-NEXT:    [[ARRAYCTOR_DST_NEXT]] = getelementptr inbounds i32, ptr [[ARRAYCTOR_DST_CUR]], i64 1
// CHECK-NEXT:    [[ARRAYCTOR_SRC_NEXT]] = getelementptr inbounds i32, ptr [[ARRAYCTOR_SRC_CUR]], i64 1
// CHECK-NEXT:    [[ARRAYCTOR_DONE:%.*]] = icmp eq ptr [[ARRAYCTOR_DST_NEXT]], [[ARRAYCTOR_DST_END]]
// CHECK-NEXT:    br i1 [[ARRAYCTOR_DONE]], label [[ARRAYCTOR_CONT:%.*]], label [[ARRAYCTOR_LOOP]]
// CHECK:       arrayctor.cont:
// CHECK-NEXT:    ret void, !dbg [[DBG14:![0-9]+]]
//
//
// CHECK-LABEL: @red_comb(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTADDR2:%.*]] = alloca i64, align 8
// CHECK-NEXT:    store ptr [[TMP0:%.*]], ptr [[DOTADDR]], align 8
// CHECK-NEXT:    store ptr [[TMP1:%.*]], ptr [[DOTADDR1]], align 8
// CHECK-NEXT:    store i64 [[TMP2:%.*]], ptr [[DOTADDR2]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK-NEXT:    [[TMP5:%.*]] = load i64, ptr [[DOTADDR2]], align 8
// CHECK-NEXT:    [[TMP6:%.*]] = udiv exact i64 [[TMP5]], 4
// CHECK-NEXT:    [[ARRAYCTOR_DST_END:%.*]] = getelementptr inbounds i32, ptr [[TMP3]], i64 [[TMP6]]
// CHECK-NEXT:    br label [[ARRAYCTOR_LOOP:%.*]]
// CHECK:       arrayctor.loop:
// CHECK-NEXT:    [[ARRAYCTOR_DST_CUR:%.*]] = phi ptr [ [[TMP3]], [[ENTRY:%.*]] ], [ [[ARRAYCTOR_DST_NEXT:%.*]], [[ARRAYCTOR_LOOP]] ]
// CHECK-NEXT:    [[ARRAYCTOR_SRC_CUR:%.*]] = phi ptr [ [[TMP4]], [[ENTRY]] ], [ [[ARRAYCTOR_SRC_NEXT:%.*]], [[ARRAYCTOR_LOOP]] ]
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[ARRAYCTOR_DST_CUR]], align 4, !dbg [[DBG17:![0-9]+]]
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[ARRAYCTOR_SRC_CUR]], align 4, !dbg [[DBG17]]
// CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP7]], [[TMP8]], !dbg [[DBG19:![0-9]+]]
// CHECK-NEXT:    store i32 [[ADD]], ptr [[ARRAYCTOR_DST_CUR]], align 4, !dbg [[DBG19]]
// CHECK-NEXT:    [[ARRAYCTOR_DST_NEXT]] = getelementptr inbounds i32, ptr [[ARRAYCTOR_DST_CUR]], i64 1
// CHECK-NEXT:    [[ARRAYCTOR_SRC_NEXT]] = getelementptr inbounds i32, ptr [[ARRAYCTOR_SRC_CUR]], i64 1
// CHECK-NEXT:    [[ARRAYCTOR_DONE:%.*]] = icmp eq ptr [[ARRAYCTOR_DST_NEXT]], [[ARRAYCTOR_DST_END]]
// CHECK-NEXT:    br i1 [[ARRAYCTOR_DONE]], label [[ARRAYCTOR_CONT:%.*]], label [[ARRAYCTOR_LOOP]]
// CHECK:       arrayctor.cont:
// CHECK-NEXT:    ret void, !dbg [[DBG17]]
//
//
// CHECK-LABEL: @compute_dep(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT__DEPEND_UNPACK_T:%.*]], align 8
// CHECK-NEXT:    [[RX_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[RX:%.*]], ptr [[RX_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw [[STRUCT__DEPEND_UNPACK_T]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:    store ptr [[RX]], ptr [[TMP0]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw [[STRUCT__DEPEND_UNPACK_T]], ptr [[RETVAL]], i32 0, i32 1
// CHECK-NEXT:    store i64 4, ptr [[TMP1]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds nuw [[STRUCT__DEPEND_UNPACK_T]], ptr [[RETVAL]], i32 0, i32 2
// CHECK-NEXT:    store i64 0, ptr [[TMP2]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds nuw [[STRUCT__DEPEND_UNPACK_T]], ptr [[RETVAL]], i32 0, i32 3
// CHECK-NEXT:    store i64 4, ptr [[TMP3]], align 8
// CHECK-NEXT:    [[TMP4:%.*]] = load [[STRUCT__DEPEND_UNPACK_T]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret [[STRUCT__DEPEND_UNPACK_T]] [[TMP4]]
//
