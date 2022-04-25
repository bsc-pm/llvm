// RUN: %clang -x c++ -fompss-2 -Xclang -disable-llvm-passes %s -S -g -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
struct S {
    int x = 4;
    void foo() {
        #pragma oss task
        {
            x++;
            x++;
        }
    }
};

int main() {
    int x = 10;
    int vla[x];
    int array[10];
    S s;
    s.foo();
    x = vla[0] = array[0] = 43;
    #pragma oss task
    {
        x++;
        vla[0]++;
        array[0]++;
    }
}

// This test checks we reemit debug intrinsics again

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %x), "QUAL.OSS.FIRSTPRIVATE"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.FIRSTPRIVATE"([10 x i32]* %array), "QUAL.OSS.CAPTURED"(i64 %1) ]
// CHECK-NEXT: call void @llvm.dbg.declare(metadata i32* %x
// CHECK-NEXT: call void @llvm.dbg.declare(metadata i32* %vla
// CHECK-NEXT: call void @llvm.dbg.declare(metadata [10 x i32]* %array

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1) ]
// CHECK-NEXT: call void @llvm.dbg.declare(metadata %struct.S* %this1

