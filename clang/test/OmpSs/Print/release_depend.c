// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int array[10][10];
int main(void) {
    int a;
    int b;
    #pragma oss release depend(out : a) out(b)
    #pragma oss release depend(out : array[0 : 5])
    #pragma oss release depend(out : array[ : a]) out(array[ ; a])
    #pragma oss release depend(out : array[ : ]) out(array[ ; ])
    #pragma oss release depend(out : array[ : ][ : ]) out(array[ ; ][ ; ])
    #pragma oss release depend(in : [1][2][3]array)
}

// CHECK: #pragma oss release depend(out : a) out(b)
// CHECK-NEXT: #pragma oss release depend(out : array[0:5])
// CHECK-NEXT: #pragma oss release depend(out : array[:a]) out(array[;a])
// CHECK-NEXT: #pragma oss release depend(out : array[:]) out(array[;])
// CHECK-NEXT: #pragma oss release depend(out : array[:][:]) out(array[;][;])
// CHECK-NEXT: #pragma oss release depend(in : [1][2][3]array)

