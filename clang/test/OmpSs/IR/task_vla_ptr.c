// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
void matmul(const long TS, void *C) {
    #pragma oss task weakinout(C)
    {
        __attribute__((unused))
        double (* restrict C_block)[TS] = C;
    }
}

// Check we firstprivatize TS

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i8** %C.addr), "QUAL.OSS.FIRSTPRIVATE"(i64* %TS.addr), "QUAL.OSS.DEP.WEAKINOUT"(i8** %C.addr, [2 x i8] c"C\00", %struct._depend_unpack_t (i8**)* @compute_dep, i8** %C.addr) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i8** %C)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK-NEXT:   %C.addr = alloca i8**, align 8
// CHECK-NEXT:   store i8** %C, i8*** %C.addr, align 8
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i8** %C, i8*** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 8, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 8, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %4
// CHECK-NEXT: }
