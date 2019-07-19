// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int array[10][10];
int main(void) {
    int a;
    #pragma oss task depend(out : a)
    {}
    #pragma oss task depend(out : array[0 : 5])
    {}
    #pragma oss task depend(out : array[ : a])
    {}
    #pragma oss task depend(out : array[ : ])
    {}
    #pragma oss task depend(out : array[ : ][ : ])
    {}
}

// CHECK:       #pragma oss task depend(out : a)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[0:5])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[:a])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[:])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[:][:])
// CHECK-NEXT: {
// CHECK-NEXT: }

