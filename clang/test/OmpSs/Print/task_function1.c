// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task depend(mutexinoutset: *a, *a) concurrent(*a, *a)
void foo(int *a) {}

int main() {
    foo(0);
}

// CHECK: #pragma oss task concurrent(*a, *a) depend(mutexinoutset:*a, *a)

