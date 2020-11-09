// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task depend(mutexinoutset: p[1 : 5]) concurrent(p[1 ; 5])
void foo1(int *p) {}

void bar() {
    foo1(0);
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %call_arg), "QUAL.OSS.DEP.CONCURRENT"(i32** %call_arg, [9 x i8] c"p[1 ; 5]\00", %struct._depend_unpack_t (i32**)* @compute_dep, i32** %call_arg), "QUAL.OSS.DEP.COMMUTATIVE"(i32** %call_arg, [9 x i8] c"p[1 : 5]\00", %struct._depend_unpack_t.0 (i32**)* @compute_dep.1, i32** %call_arg) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %0 = load i32*, i32** %p, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 20, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 24, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %0 = load i32*, i32** %p, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 20, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 24, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %5
// CHECK-NEXT: }

void bar1(int *p) {
    #pragma oss task weakcommutative(*p)
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %p.addr), "QUAL.OSS.DEP.WEAKCOMMUTATIVE"(i32** %p.addr, [3 x i8] c"*p\00", %struct._depend_unpack_t.1 (i32**)* @compute_dep.2, i32** %p.addr) ]

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32** %p)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %0 = load i32*, i32** %p, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %5
// CHECK-NEXT: }
