// RUN: %clang_cc1 -verify -fompss-2 -ast-print -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int main() {
    int x;
    #pragma oss task default(shared)
    { x = 3; }
}

// CHECK: #pragma oss task default(shared)
// CHECK-NEXT: {
// CHECK-NEXT:   x = 3;
// CHECK-NEXT: }

