// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

#pragma oss task priority(foo<int>())
void foo1() {}
#pragma oss task priority(n)
void foo2(int n) {}

void bar(int n) {
    int vla[n];
    #pragma oss task priority(foo<int>())
    {}
    #pragma oss task priority(n)
    {}
    #pragma oss task priority(vla[1])
    {}
    foo1();
    foo2(n);
}

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIORITY"(i32 ()* @compute_priority) ]
// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.PRIORITY"(i32 (i32*)* @compute_priority.1, i32* %n.addr) ]
// CHECK: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1), "QUAL.OSS.PRIORITY"(i32 (i32*, i64)* @compute_priority.2, i32* %vla, i64 %1), "QUAL.OSS.CAPTURED"(i64 %1) ]
// CHECK: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIORITY"(i32 ()* @compute_priority.3), "QUAL.OSS.DECL.SOURCE"([9 x i8] c"foo1:5:9\00") ]
// CHECK: %8 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %call_arg), "QUAL.OSS.PRIORITY"(i32 (i32*)* @compute_priority.4, i32* %call_arg), "QUAL.OSS.DECL.SOURCE"([9 x i8] c"foo2:7:9\00") ]

// CHECK: define internal i32 @compute_priority.1(i32* %n)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %n.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %n, i32** %n.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %n, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_priority.2(i32* %vla, i64 %0)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %vla.addr = alloca i32*, align 8
// CHECK-NEXT:   %.addr = alloca i64, align 8
// CHECK-NEXT:   store i32* %vla, i32** %vla.addr, align 8
// CHECK-NEXT:   store i64 %0, i64* %.addr, align 8
// CHECK-NEXT:   %arrayidx = getelementptr inbounds i32, i32* %vla, i64 1
// CHECK-NEXT:   %1 = load i32, i32* %arrayidx, align 4
// CHECK-NEXT:   ret i32 %1
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_priority.3()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %call = call noundef{{( signext)?}} i32 @_Z3fooIiET_v()
// CHECK-NEXT:   ret i32 %call
// CHECK-NEXT: }

// CHECK: define internal i32 @compute_priority.4(i32* %n)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %n.addr = alloca i32*, align 8
// CHECK-NEXT:   store i32* %n, i32** %n.addr, align 8
// CHECK-NEXT:   %0 = load i32, i32* %n, align 4
// CHECK-NEXT:   ret i32 %0
// CHECK-NEXT: }



