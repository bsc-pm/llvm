// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int x) {
  #pragma oss task reduction(+: x)
  {}
  #pragma oss task weakreduction(+: x)
  {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x.addr), "QUAL.OSS.DEP.REDUCTION"(i32 6000, i32* %x.addr, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %x.addr), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x.addr, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x.addr, void (i32*, i32*, i64)* @red_comb) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x.addr), "QUAL.OSS.DEP.WEAKREDUCTION"(i32 6000, i32* %x.addr, [2 x i8] c"x\00", %struct._depend_unpack_t.0 (i32*)* @compute_dep.1, i32* %x.addr), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x.addr, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x.addr, void (i32*, i32*, i64)* @red_comb) ]

// CHECK: define internal void @red_init(i32* noundef %0, i32* noundef %1, i64 noundef %2)
// CHECK: store i32 0, i32* %arrayctor.dst.cur, align 4

// CHECK: define internal void @red_comb(i32* noundef %0, i32* noundef %1, i64 noundef %2)
// CHECK: %add = add nsw i32 %7, %8
// CHECK: store i32 %add, i32* %arrayctor.dst.cur, align 4

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %x)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %x)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %4
// CHECK-NEXT: }
