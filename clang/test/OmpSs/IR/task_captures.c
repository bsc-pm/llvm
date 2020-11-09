// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
void vla_senction_dep(int n, int k, int j) {
    int array[n + 1][k + 2][j + 3];
    int array2[n + 1][k + 2][j + 3];
    // DSA duplicated clauses are removed
    #pragma oss task out(array[0 : 5]) in(array, array2) shared(array, array) shared(array)
    {}
}

// CHECK: %17 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.SHARED"(i32* %vla6), "QUAL.OSS.VLA.DIMS"(i32* %vla6, i64 %10, i64 %12, i64 %14), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5, i64 %10, i64 %12, i64 %14), "QUAL.OSS.DEP.IN"(i32* %vla, [6 x i8] c"array\00", %struct._depend_unpack_t (i32*, i64, i64, i64)* @compute_dep, i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.DEP.IN"(i32* %vla6, [7 x i8] c"array2\00", %struct._depend_unpack_t.0 (i32*, i64, i64, i64)* @compute_dep.1, i32* %vla6, i64 %10, i64 %12, i64 %14), "QUAL.OSS.DEP.OUT"(i32* %vla, [13 x i8] c"array[0 : 5]\00", %struct._depend_unpack_t.1 (i32*, i64, i64, i64)* @compute_dep.2, i32* %vla, i64 %1, i64 %3, i64 %5) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %array, i64 %0, i64 %1, i64 %2)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %3 = mul i64 %2, 4
// CHECK-NEXT:   %4 = mul i64 %2, 4
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %array, i32** %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %4, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %1, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %1, i64* %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 7
// CHECK-NEXT:   store i64 %0, i64* %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 8
// CHECK-NEXT:   store i64 0, i64* %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 9
// CHECK-NEXT:   store i64 %0, i64* %14, align 8
// CHECK-NEXT:   %15 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %15
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %array2, i64 %0, i64 %1, i64 %2)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %3 = mul i64 %2, 4
// CHECK-NEXT:   %4 = mul i64 %2, 4
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %array2, i32** %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %4, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %1, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %1, i64* %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 7
// CHECK-NEXT:   store i64 %0, i64* %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 8
// CHECK-NEXT:   store i64 0, i64* %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 9
// CHECK-NEXT:   store i64 %0, i64* %14, align 8
// CHECK-NEXT:   %15 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %15
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32* %array, i64 %0, i64 %1, i64 %2)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %3 = mul i64 %2, 4
// CHECK-NEXT:   %4 = mul i64 %2, 4
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %array, i32** %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 %3, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %7, align 8
// CHECK-NEXT:   %8 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %4, i64* %8, align 8
// CHECK-NEXT:   %9 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 4
// CHECK-NEXT:   store i64 %1, i64* %9, align 8
// CHECK-NEXT:   %10 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 5
// CHECK-NEXT:   store i64 0, i64* %10, align 8
// CHECK-NEXT:   %11 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 6
// CHECK-NEXT:   store i64 %1, i64* %11, align 8
// CHECK-NEXT:   %12 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 7
// CHECK-NEXT:   store i64 %0, i64* %12, align 8
// CHECK-NEXT:   %13 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 8
// CHECK-NEXT:   store i64 0, i64* %13, align 8
// CHECK-NEXT:   %14 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 9
// CHECK-NEXT:   store i64 6, i64* %14, align 8
// CHECK-NEXT:   %15 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %15
// CHECK-NEXT: }

