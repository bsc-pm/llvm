// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
    int x;
    #pragma oss task
    void foo1(int x) {}
};

void bar() {
    S s;
    s.foo1(1 + 2);
    S *p;
    p->foo1(1 + 2);
}

// CHECK: store i32 3, i32* %call_arg, align 4
// CHECK-NEXT: store %struct.S* %s, %struct.S** %call_arg1, align 8
// CHECK-NEXT: %0 = load %struct.S*, %struct.S** %call_arg1, align 8
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %0), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg)
// CHECK-NEXT: %2 = load i32, i32* %call_arg, align 4
// CHECK-NEXT: call void @_ZN1S4foo1Ei(%struct.S* noundef nonnull align 4 dereferenceable(4) %0, i32 noundef{{( signext)?}} %2)

// CHECK: store i32 3, i32* %call_arg2, align 4
// CHECK-NEXT: %3 = load %struct.S*, %struct.S** %p, align 8
// CHECK-NEXT: store %struct.S* %3, %struct.S** %call_arg3, align 8
// CHECK-NEXT: %4 = load %struct.S*, %struct.S** %call_arg3, align 8
// CHECK-NEXT: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %4), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg2)
// CHECK-NEXT: %6 = load i32, i32* %call_arg2, align 4
// CHECK-NEXT: call void @_ZN1S4foo1Ei(%struct.S* noundef nonnull align 4 dereferenceable(4) %4, i32 noundef{{( signext)?}} %6)

