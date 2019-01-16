// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

int main(void) {
    #pragma oss taskwait
}

// CHECK: int main() {
// CHECK-NEXT:     #pragma oss taskwait
// CHECK-NEXT: }
