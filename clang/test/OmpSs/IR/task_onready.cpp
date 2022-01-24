// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
    int x;
    void foo() {
        #pragma oss task onready(x)
        {}
    }
};

template<typename T> T foo() { return T(); }

#pragma oss task onready(vla[3])
void foo1(int n, int *vla[n]) {}

#pragma oss task onready(foo<int *>())
void foo2() {}

void bar(int n) {
    S s;
    s.foo();
    n = -1;
    const int m = -1;
    int *vla[n];
    #pragma oss task onready(vla[3])
    {}
    #pragma oss task onready(foo<int>())
    {}
    #pragma oss task onready(n)
    {}
    #pragma oss task onready(m)
    {}
    #pragma oss task onready(-1)
    {}
    #pragma oss task onready([&n]() {})
    {}
    foo1(10, vla);
    foo2();
}

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %vla), "QUAL.OSS.VLA.DIMS"(i32** %vla, i64 %1), "QUAL.OSS.ONREADY"(void (i32**, i64)* @compute_onready, i32** %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ]
// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.ONREADY"(void ()* @compute_onready.1) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.ONREADY"(void (i32*)* @compute_onready.2, i32* %n.addr) ]
// CHECK: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %m), "QUAL.OSS.ONREADY"(void ()* @compute_onready.3) ]
// CHECK: %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.ONREADY"(void ()* @compute_onready.4) ]
// CHECK: %8 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.ONREADY"(void (i32*)* @compute_onready.5, i32* %n.addr) ]
// CHECK: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.FIRSTPRIVATE"(i32*** %call_arg1), "QUAL.OSS.ONREADY"(void (i32***)* @compute_onready.6, i32*** %call_arg1), "QUAL.OSS.DECL.SOURCE"([10 x i8] c"foo1:14:9\00") ]
// CHECK: %12 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.ONREADY"(void ()* @compute_onready.7), "QUAL.OSS.DECL.SOURCE"([10 x i8] c"foo2:17:9\00") ]

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1), "QUAL.OSS.ONREADY"(void (%struct.S*)* @compute_onready.8, %struct.S* %this1) ]

// CHECK: define internal void @compute_onready(i32** %vla, i64 %0)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %vla.addr = alloca i32**, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32** %vla, i32*** %vla.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   %arrayidx = getelementptr inbounds i32*, i32** %vla, i64 3
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.1()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %call = call noundef{{( signext)?}} i32 @_Z3fooIiET_v()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.2(i32* %n)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %n.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %n, i32** %n.addr, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.3()
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.4()
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.5(i32* %n)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %n.addr = alloca i32*, align 8
// CHECK-NEXT:   %agg.tmp.ensured = alloca %class.anon, align 8
// CHECK-NEXT:   store i32* %n, i32** %n.addr, align 8
// CHECK-NEXT:   %0 = getelementptr inbounds %class.anon, %class.anon* %agg.tmp.ensured, i32 0, i32 0
// CHECK-NEXT:   store i32* %n, i32** %0, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.6(i32*** %vla)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %vla.addr = alloca i32***, align 8
// CHECK-NEXT:   store i32*** %vla, i32**** %vla.addr, align 8
// CHECK-NEXT:   %0 = load i32**, i32*** %vla, align 8
// CHECK-NEXT:   %arrayidx = getelementptr inbounds i32*, i32** %0, i64 3
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.7()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %call = call noundef i32* @_Z3fooIPiET_v()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @compute_onready.8(%struct.S* %this)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %this.addr = alloca %struct.S*, align 8
// CHECK-NEXT:   store %struct.S* %this, %struct.S** %this.addr, align 8
// CHECK-NEXT:   %this1 = load %struct.S*, %struct.S** %this.addr, align 8
// CHECK-NEXT:   %x = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

