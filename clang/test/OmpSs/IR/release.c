// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main() {
    int x;
    #pragma oss release in(x)
    #pragma oss task
    {
        #pragma oss release in(x)
    }
    #pragma oss task in(x)
    {
        #pragma oss release in(x)
    }
}

// CHECK: %0 = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %x)
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %x) ]
// CHECK-NEXT: %2 = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.0 (i32*)* @compute_dep.1, i32* %x) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.1 (i32*)* @compute_dep.2, i32* %x) ]
// CHECK-NEXT: %4 = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t.2 (i32*)* @compute_dep.3, i32* %x) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %3)
