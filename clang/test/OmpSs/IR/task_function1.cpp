// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// References are passed as shared so task outline can modify
// the original variable

#pragma oss task in([1]p)
void foo(int &x, int *&p) {}

int main() {
    int array[10];
    int *p = array;
    foo(*(array + 4), p);
}

// CHECK: store i32* %add.ptr, i32** %call_arg, align 8
// CHECK-NEXT: %0 = load i32*, i32** %call_arg, align 8
// CHECK-NEXT: store i32** %p, i32*** %call_arg2, align 8
// CHECK-NEXT: %1 = load i32**, i32*** %call_arg2, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %0), "QUAL.OSS.SHARED"(i32** %1), "QUAL.OSS.DEP.IN"(i32** %1, [5 x i8] c"[1]p\00", %struct._depend_unpack_t (i32**)* @compute_dep, i32** %1)
// CHECK-NEXT: call void @_Z3fooRiRPi(i32* noundef nonnull align 4 dereferenceable(4) %0, i32** noundef nonnull align 8 dereferenceable(8) %1)
