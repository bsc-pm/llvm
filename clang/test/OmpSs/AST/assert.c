// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

#pragma oss assert("a == b")

int main() {}

// CHECK: OSSAssertDecl 0x{{.*}} <{{.*}}:{{.*}}:{{.*}}> col:{{.*}}
