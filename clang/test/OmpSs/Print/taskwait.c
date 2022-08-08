// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int main(void) {
    int *p;
    #pragma oss taskwait on([1]p)
}

// CHECK: int main(void) {
// CHECK-NEXT:     int *p;
// CHECK-NEXT:     #pragma oss taskwait inout([1]p)
// CHECK-NEXT: }
