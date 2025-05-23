// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-LABEL: @main(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[X:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep, ptr [[X]]) ], !dbg [[DBG9:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr [[X]], i32 undef) ], !dbg [[DBG10:![0-9]+]]
// CHECK-NEXT:    [[TMP2:%.*]] = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep.1, ptr [[X]]) ], !dbg [[DBG11:![0-9]+]]
// CHECK-NEXT:    [[TMP3:%.*]] = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.COMMUTATIVE"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep.2, ptr [[X]]) ], !dbg [[DBG12:![0-9]+]]
// CHECK-NEXT:    [[TMP4:%.*]] = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.WEAKCOMMUTATIVE"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep.3, ptr [[X]]) ], !dbg [[DBG13:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP1]]), !dbg [[DBG14:![0-9]+]]
// CHECK-NEXT:    [[TMP5:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr [[X]], i32 undef), "QUAL.OSS.DEP.IN"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep.4, ptr [[X]]) ], !dbg [[DBG15:![0-9]+]]
// CHECK-NEXT:    [[TMP6:%.*]] = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep.5, ptr [[X]]) ], !dbg [[DBG16:![0-9]+]]
// CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.CONCURRENT"(ptr [[X]], [2 x i8] c"x\00", ptr @compute_dep.6, ptr [[X]]) ], !dbg [[DBG17:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP5]]), !dbg [[DBG18:![0-9]+]]
// CHECK-NEXT:    ret i32 0, !dbg [[DBG19:![0-9]+]]
//
int main() {
    int x;
    #pragma oss release in(x)
    #pragma oss task
    {
        #pragma oss release in(x)
        #pragma oss release commutative(x)
        #pragma oss release weakcommutative(x)
    }
    #pragma oss task in(x)
    {
        #pragma oss release in(x)
        #pragma oss release concurrent(x)
    }
}

