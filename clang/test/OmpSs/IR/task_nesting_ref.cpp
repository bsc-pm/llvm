// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-LABEL: @main(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[X:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[RX1:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[X]], ptr [[RX1]], align 8, !dbg [[DBG9:![0-9]+]]
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[RX1]], align 8, !dbg [[DBG10:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr [[TMP0]], i32 undef) ], !dbg [[DBG10]]
// CHECK-NEXT:    store i32 34, ptr [[TMP0]], align 4, !dbg [[DBG11:![0-9]+]]
// CHECK-NEXT:    [[TMP2:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ], !dbg [[DBG12:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP2]]), !dbg [[DBG13:![0-9]+]]
// CHECK-NEXT:    store i32 34, ptr [[TMP0]], align 4, !dbg [[DBG14:![0-9]+]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP1]]), !dbg [[DBG15:![0-9]+]]
// CHECK-NEXT:    ret i32 0, !dbg [[DBG16:![0-9]+]]
//
int main() {
    int x;
    int &rx1 = x;
    #pragma oss task firstprivate(rx1)
    {
        rx1 = 34;
        #pragma oss task
        {}
        // The bug was the map of Addresses for refs was cleaned after a task
        // emission, so this stmt could not get its Address
        rx1 = 34;
    }
}


