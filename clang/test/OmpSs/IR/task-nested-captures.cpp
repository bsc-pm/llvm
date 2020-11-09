// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int &x) {
    #pragma oss task in(x)
    {
        #pragma oss task in(x)
        {
            x++;
        }
    }
}

// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %0), "QUAL.OSS.DEP.IN"(i32* %0, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %0) ]
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %0), "QUAL.OSS.DEP.IN"(i32* %0, [2 x i8] c"x\00", %struct._depend_unpack_t.0 (i32*)* @compute_dep.1, i32* %0) ]

