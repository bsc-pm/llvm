// RUN: %clang_cc1 -verify -fompss-2 -fexceptions -fcxx-exceptions -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics


// TODO: Fix this
void foo1() {
    #pragma oss task
    {
        throw 1;
    }
}

void foo1() {
    #pragma oss task
    {
        try {
            throw 1;
        } catch (int e) {
        }
    }
    #pragma oss task
    {
        try {
            throw 1;
        } catch (int e) {
        }
    }
}

// Each task has its own exn.slot, ehselector.slot, and all it's exception
// handling BB

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT: %exn.slot = alloca i8*
// CHECK-NEXT: %ehselector.slot = alloca i32
// CHECK: try.cont:                                         ; preds = %catch
// CHECK: call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT: %exn.slot4 = alloca i8*
// CHECK-NEXT: %ehselector.slot5 = alloca i32
// CHECK: try.cont12:                                       ; preds = %catch9
// CHECK: call void @llvm.directive.region.exit(token %9)
// CHECK-NEXT: ret void

