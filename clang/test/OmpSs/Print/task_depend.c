// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int array[10][10];
int main(void) {
    int a;
    int b;
    #pragma oss task depend(out : a) out(b)
    {}
    #pragma oss task depend(out : array[0 : 5])
    {}
    #pragma oss task depend(out : array[ : a]) out(array[ ; a])
    {}
    #pragma oss task depend(out : array[ : ]) out(array[ ; ])
    {}
    #pragma oss task depend(out : array[ : ][ : ]) out(array[ ; ][ ; ])
    {}
    #pragma oss task depend(in : [1][2][3]array)
    {}
}

// CHECK:       #pragma oss task depend(out : a) out(b)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[0:5])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[:a]) out(array[;a])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[:]) out(array[;])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(out : array[:][:]) out(array[;][;])
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss task depend(in : [1][2][3]array)
// CHECK-NEXT: {
// CHECK-NEXT: }

