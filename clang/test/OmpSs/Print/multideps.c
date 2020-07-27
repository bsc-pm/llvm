// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int main() {
    int v[10];
    int M[10][10];
    #pragma oss task in({v[i], i=0:10:1})
    {}
    #pragma oss task in({v[i], i=0;10:1})
    {}
    #pragma oss task in({v[i], i={0, 1, 2}})
    {}
    #pragma oss task in({M[i][j], i={0, 1, 2}, j={0, 1, 2}})
    {}
}

// CHECK:    #pragma oss task in({v[i], i = 0:10:1})
// CHECK-NEXT:        {
// CHECK-NEXT:        }
// CHECK-NEXT:    #pragma oss task in({v[i], i = 0;10:1})
// CHECK-NEXT:        {
// CHECK-NEXT:        }
// CHECK-NEXT:    #pragma oss task in({v[i], i = {0, 1, 2}})
// CHECK-NEXT:        {
// CHECK-NEXT:        }
// CHECK-NEXT:    #pragma oss task in({M[i][j], i = {0, 1, 2}, j = {0, 1, 2}})
// CHECK-NEXT:        {
// CHECK-NEXT:        }

