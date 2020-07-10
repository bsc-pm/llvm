// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

// UNSUPPORTED: true

int main() {
    int x;
    #pragma oss task reduction(operator+: x)
    {}
    #pragma oss task reduction(+: x)
    {}
}

// CHECK: #pragma oss task reduction(+: x)
// CHECK: #pragma oss task reduction(+: x)

