// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int main(void) {
    int a;
    #pragma oss task depend(out : a)
    {
    }
}

// CHECK: int main() {
// CHECK-NEXT:     int a;
// CHECK-NEXT:     #pragma oss task depend(out : a)
// CHECK-NEXT:     {
// CHECK-NEXT:     }
// CHECK-NEXT: }
