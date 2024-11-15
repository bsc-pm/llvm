// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-LABEL: @foo(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr [[A]], i32 undef) ], !dbg [[DBG9:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP0]]), !dbg [[DBG10:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(ptr [[A]], i32 undef) ], !dbg [[DBG11:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP1]]), !dbg [[DBG12:![0-9]+]]
// CHECK-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr [[A]], i32 undef) ], !dbg [[DBG13:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG14:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG15:![0-9]+]]
//
void foo() {
    int a;
    #pragma oss task shared(a) shared(a)
    {}
    #pragma oss task private(a) private(a)
    {}
    #pragma oss task firstprivate(a) firstprivate(a)
    {}
}

