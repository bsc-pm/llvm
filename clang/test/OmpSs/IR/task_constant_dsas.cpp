// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -triple x86_64-gnu-linux -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s --check-prefixes=LIN64
// RUN: %clang_cc1 -triple ppc64 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s --check-prefixes=PPC64
// RUN: %clang_cc1 -triple aarch64 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s --check-prefixes=AARCH64
// expected-no-diagnostics


// This test checks that we emit DSA for constants. The use case of them is
// when the code gets the constant address.

constexpr int a = 0;
constexpr int *p = 0;
constexpr int const &ra = a;
constexpr int *const &rp = p;
constexpr int N = 4;
int asdf();
const int b = asdf();
enum { M = 5 };

// LIN64-LABEL: @_Z3foov(
// LIN64-NEXT:  entry:
// LIN64-NEXT:    [[TMP0:%.*]] = load ptr, ptr @ra, align 8, !dbg [[DBG13:![0-9]+]]
// LIN64-NEXT:    [[TMP1:%.*]] = load ptr, ptr @rp, align 8, !dbg [[DBG13]]
// LIN64-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1N, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1a, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP1]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1p, ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1b, i32 undef) ], !dbg [[DBG13]]
// LIN64-NEXT:    [[X:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[Y:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[Z:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[I:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[J:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[K:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[L:%.*]] = alloca i32, align 4
// LIN64-NEXT:    store ptr @_ZL1N, ptr [[X]], align 8, !dbg [[DBG14:![0-9]+]]
// LIN64-NEXT:    store ptr [[TMP0]], ptr [[Y]], align 8, !dbg [[DBG15:![0-9]+]]
// LIN64-NEXT:    store ptr @_ZL1a, ptr [[Z]], align 8, !dbg [[DBG16:![0-9]+]]
// LIN64-NEXT:    store ptr [[TMP1]], ptr [[I]], align 8, !dbg [[DBG17:![0-9]+]]
// LIN64-NEXT:    store ptr @_ZL1p, ptr [[J]], align 8, !dbg [[DBG18:![0-9]+]]
// LIN64-NEXT:    store ptr @_ZL1b, ptr [[K]], align 8, !dbg [[DBG19:![0-9]+]]
// LIN64-NEXT:    store i32 5, ptr [[L]], align 4, !dbg [[DBG20:![0-9]+]]
// LIN64-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG21:![0-9]+]]
// LIN64-NEXT:    ret void, !dbg [[DBG22:![0-9]+]]
//
// PPC64-LABEL: @_Z3foov(
// PPC64-NEXT:  entry:
// PPC64-NEXT:    [[TMP0:%.*]] = load ptr, ptr @ra, align 8, !dbg [[DBG13:![0-9]+]]
// PPC64-NEXT:    [[TMP1:%.*]] = load ptr, ptr @rp, align 8, !dbg [[DBG13]]
// PPC64-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1N, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1a, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP1]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1p, ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1b, i32 undef) ], !dbg [[DBG13]]
// PPC64-NEXT:    [[X:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[Y:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[Z:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[I:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[J:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[K:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[L:%.*]] = alloca i32, align 4
// PPC64-NEXT:    store ptr @_ZL1N, ptr [[X]], align 8, !dbg [[DBG14:![0-9]+]]
// PPC64-NEXT:    store ptr [[TMP0]], ptr [[Y]], align 8, !dbg [[DBG15:![0-9]+]]
// PPC64-NEXT:    store ptr @_ZL1a, ptr [[Z]], align 8, !dbg [[DBG16:![0-9]+]]
// PPC64-NEXT:    store ptr [[TMP1]], ptr [[I]], align 8, !dbg [[DBG17:![0-9]+]]
// PPC64-NEXT:    store ptr @_ZL1p, ptr [[J]], align 8, !dbg [[DBG18:![0-9]+]]
// PPC64-NEXT:    store ptr @_ZL1b, ptr [[K]], align 8, !dbg [[DBG19:![0-9]+]]
// PPC64-NEXT:    store i32 5, ptr [[L]], align 4, !dbg [[DBG20:![0-9]+]]
// PPC64-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG21:![0-9]+]]
// PPC64-NEXT:    ret void, !dbg [[DBG22:![0-9]+]]
//
// AARCH64-LABEL: @_Z3foov(
// AARCH64-NEXT:  entry:
// AARCH64-NEXT:    [[TMP0:%.*]] = load ptr, ptr @ra, align 8, !dbg [[DBG17:![0-9]+]]
// AARCH64-NEXT:    [[TMP1:%.*]] = load ptr, ptr @rp, align 8, !dbg [[DBG17]]
// AARCH64-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1N, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1a, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP1]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1p, ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr @_ZL1b, i32 undef) ], !dbg [[DBG17]]
// AARCH64-NEXT:    [[X:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[Y:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[Z:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[I:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[J:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[K:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[L:%.*]] = alloca i32, align 4
// AARCH64-NEXT:    store ptr @_ZL1N, ptr [[X]], align 8, !dbg [[DBG18:![0-9]+]]
// AARCH64-NEXT:    store ptr [[TMP0]], ptr [[Y]], align 8, !dbg [[DBG19:![0-9]+]]
// AARCH64-NEXT:    store ptr @_ZL1a, ptr [[Z]], align 8, !dbg [[DBG20:![0-9]+]]
// AARCH64-NEXT:    store ptr [[TMP1]], ptr [[I]], align 8, !dbg [[DBG21:![0-9]+]]
// AARCH64-NEXT:    store ptr @_ZL1p, ptr [[J]], align 8, !dbg [[DBG22:![0-9]+]]
// AARCH64-NEXT:    store ptr @_ZL1b, ptr [[K]], align 8, !dbg [[DBG23:![0-9]+]]
// AARCH64-NEXT:    store i32 5, ptr [[L]], align 4, !dbg [[DBG24:![0-9]+]]
// AARCH64-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG25:![0-9]+]]
// AARCH64-NEXT:    ret void, !dbg [[DBG26:![0-9]+]]
//
void foo() {
    #pragma oss task firstprivate(N, ra, a, rp, p, b)
    {
        const int *x = &N;
        const int *y = &ra;
        const int *z = &a;
        const int *const *i = &rp;
        const int *const *j = &p;
        const int *k = &b;
        const int l = M;
    }
}

// LIN64-LABEL: @_Z3barv(
// LIN64-NEXT:  entry:
// LIN64-NEXT:    [[A:%.*]] = alloca i32, align 4
// LIN64-NEXT:    [[P:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[N:%.*]] = alloca i32, align 4
// LIN64-NEXT:    [[B:%.*]] = alloca i32, align 4
// LIN64-NEXT:    store i32 0, ptr [[A]], align 4, !dbg [[DBG24:![0-9]+]]
// LIN64-NEXT:    store ptr null, ptr [[P]], align 8, !dbg [[DBG25:![0-9]+]]
// LIN64-NEXT:    store i32 4, ptr [[N]], align 4, !dbg [[DBG26:![0-9]+]]
// LIN64-NEXT:    [[CALL:%.*]] = call noundef i32 @_Z4asdfv(), !dbg [[DBG27:![0-9]+]]
// LIN64-NEXT:    store i32 [[CALL]], ptr [[B]], align 4, !dbg [[DBG28:![0-9]+]]
// LIN64-NEXT:    [[TMP0:%.*]] = load ptr, ptr @ra, align 8, !dbg [[DBG29:![0-9]+]]
// LIN64-NEXT:    [[TMP1:%.*]] = load ptr, ptr @rp, align 8, !dbg [[DBG29]]
// LIN64-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr [[N]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[A]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP1]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[P]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[B]], i32 undef) ], !dbg [[DBG29]]
// LIN64-NEXT:    [[X:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[Z:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[J:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[K:%.*]] = alloca ptr, align 8
// LIN64-NEXT:    [[L:%.*]] = alloca i32, align 4
// LIN64-NEXT:    store ptr [[N]], ptr [[X]], align 8, !dbg [[DBG30:![0-9]+]]
// LIN64-NEXT:    store ptr [[A]], ptr [[Z]], align 8, !dbg [[DBG31:![0-9]+]]
// LIN64-NEXT:    store ptr [[P]], ptr [[J]], align 8, !dbg [[DBG32:![0-9]+]]
// LIN64-NEXT:    store ptr [[B]], ptr [[K]], align 8, !dbg [[DBG33:![0-9]+]]
// LIN64-NEXT:    store i32 5, ptr [[L]], align 4, !dbg [[DBG34:![0-9]+]]
// LIN64-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG35:![0-9]+]]
// LIN64-NEXT:    ret void, !dbg [[DBG36:![0-9]+]]
//
// PPC64-LABEL: @_Z3barv(
// PPC64-NEXT:  entry:
// PPC64-NEXT:    [[A:%.*]] = alloca i32, align 4
// PPC64-NEXT:    [[P:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[N:%.*]] = alloca i32, align 4
// PPC64-NEXT:    [[B:%.*]] = alloca i32, align 4
// PPC64-NEXT:    store i32 0, ptr [[A]], align 4, !dbg [[DBG24:![0-9]+]]
// PPC64-NEXT:    store ptr null, ptr [[P]], align 8, !dbg [[DBG25:![0-9]+]]
// PPC64-NEXT:    store i32 4, ptr [[N]], align 4, !dbg [[DBG26:![0-9]+]]
// PPC64-NEXT:    [[CALL:%.*]] = call noundef signext i32 @_Z4asdfv(), !dbg [[DBG27:![0-9]+]]
// PPC64-NEXT:    store i32 [[CALL]], ptr [[B]], align 4, !dbg [[DBG28:![0-9]+]]
// PPC64-NEXT:    [[TMP0:%.*]] = load ptr, ptr @ra, align 8, !dbg [[DBG29:![0-9]+]]
// PPC64-NEXT:    [[TMP1:%.*]] = load ptr, ptr @rp, align 8, !dbg [[DBG29]]
// PPC64-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr [[N]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[A]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP1]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[P]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[B]], i32 undef) ], !dbg [[DBG29]]
// PPC64-NEXT:    [[X:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[Z:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[J:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[K:%.*]] = alloca ptr, align 8
// PPC64-NEXT:    [[L:%.*]] = alloca i32, align 4
// PPC64-NEXT:    store ptr [[N]], ptr [[X]], align 8, !dbg [[DBG30:![0-9]+]]
// PPC64-NEXT:    store ptr [[A]], ptr [[Z]], align 8, !dbg [[DBG31:![0-9]+]]
// PPC64-NEXT:    store ptr [[P]], ptr [[J]], align 8, !dbg [[DBG32:![0-9]+]]
// PPC64-NEXT:    store ptr [[B]], ptr [[K]], align 8, !dbg [[DBG33:![0-9]+]]
// PPC64-NEXT:    store i32 5, ptr [[L]], align 4, !dbg [[DBG34:![0-9]+]]
// PPC64-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG35:![0-9]+]]
// PPC64-NEXT:    ret void, !dbg [[DBG36:![0-9]+]]
//
// AARCH64-LABEL: @_Z3barv(
// AARCH64-NEXT:  entry:
// AARCH64-NEXT:    [[A:%.*]] = alloca i32, align 4
// AARCH64-NEXT:    [[P:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[N:%.*]] = alloca i32, align 4
// AARCH64-NEXT:    [[B:%.*]] = alloca i32, align 4
// AARCH64-NEXT:    store i32 0, ptr [[A]], align 4, !dbg [[DBG28:![0-9]+]]
// AARCH64-NEXT:    store ptr null, ptr [[P]], align 8, !dbg [[DBG29:![0-9]+]]
// AARCH64-NEXT:    store i32 4, ptr [[N]], align 4, !dbg [[DBG30:![0-9]+]]
// AARCH64-NEXT:    [[CALL:%.*]] = call noundef i32 @_Z4asdfv(), !dbg [[DBG31:![0-9]+]]
// AARCH64-NEXT:    store i32 [[CALL]], ptr [[B]], align 4, !dbg [[DBG32:![0-9]+]]
// AARCH64-NEXT:    [[TMP0:%.*]] = load ptr, ptr @ra, align 8, !dbg [[DBG33:![0-9]+]]
// AARCH64-NEXT:    [[TMP1:%.*]] = load ptr, ptr @rp, align 8, !dbg [[DBG33]]
// AARCH64-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr [[N]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[A]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP1]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[P]], ptr undef), "QUAL.OSS.FIRSTPRIVATE"(ptr [[B]], i32 undef) ], !dbg [[DBG33]]
// AARCH64-NEXT:    [[X:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[Z:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[J:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[K:%.*]] = alloca ptr, align 8
// AARCH64-NEXT:    [[L:%.*]] = alloca i32, align 4
// AARCH64-NEXT:    store ptr [[N]], ptr [[X]], align 8, !dbg [[DBG34:![0-9]+]]
// AARCH64-NEXT:    store ptr [[A]], ptr [[Z]], align 8, !dbg [[DBG35:![0-9]+]]
// AARCH64-NEXT:    store ptr [[P]], ptr [[J]], align 8, !dbg [[DBG36:![0-9]+]]
// AARCH64-NEXT:    store ptr [[B]], ptr [[K]], align 8, !dbg [[DBG37:![0-9]+]]
// AARCH64-NEXT:    store i32 5, ptr [[L]], align 4, !dbg [[DBG38:![0-9]+]]
// AARCH64-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG39:![0-9]+]]
// AARCH64-NEXT:    ret void, !dbg [[DBG40:![0-9]+]]
//
void bar() {
    constexpr int a = 0;
    constexpr int *p = 0;
    constexpr int N = 4;
    const int b = asdf();
    #pragma oss task firstprivate(N, ra, a, rp, p, b)
    {
        const int *x = &N;
        const int *z = &a;
        const int *const*j = &p;
        const int *k = &b;
        const int l = M;
    }
}


