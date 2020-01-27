// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

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


// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %0) ]
// CHECK-NEXT: store i32 34, i32* %0, align 4
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)
// CHECK-NEXT: store i32 34, i32* %0, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)
