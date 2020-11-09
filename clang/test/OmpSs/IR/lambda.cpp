// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void bar(int x) {
    auto foo = [&x]() {
      #pragma oss task in(x)
      x++;
    };
    foo();
    #pragma oss taskwait
}

// CHECK: %1 = load i32*, i32** %0, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %1), "QUAL.OSS.DEP.IN"(i32* %1, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %1) ]
// CHECK-NEXT: %3 = load i32, i32* %1, align 4
// CHECK-NEXT: %inc = add nsw i32 %3, 1
// CHECK-NEXT: store i32 %inc, i32* %1, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)

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


