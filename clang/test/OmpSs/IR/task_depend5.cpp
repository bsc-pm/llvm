// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int bar();

void foo() {
    int array[10];
    #pragma oss task in(array[bar()])
    {}
}


// CHECK: define internal %struct._depend_unpack_t @compute_dep([10 x i32]* %array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %call = call noundef{{( signext)?}} i32 @_Z3barv()
// CHECK-NEXT:   %0 = sext i32 %call to i64
// CHECK-NEXT:   %1 = add i64 %0, 1
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %array, i64 0, i64 0
// CHECK-NEXT:   %2 = mul i64 %0, 4
// CHECK-NEXT:   %3 = mul i64 %1, 4
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 %2, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %3, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %8
// CHECK-NEXT: }
