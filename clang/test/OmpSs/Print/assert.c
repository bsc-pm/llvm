// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss assert("a == b")
#pragma oss assert("b == c")

int main() {

}

// CHECK: #pragma oss assert("a == b")
// CHECK: #pragma oss assert("b == c")
