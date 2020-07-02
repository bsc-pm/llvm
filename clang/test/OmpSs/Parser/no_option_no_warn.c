// RUN: %clang_cc1 -verify -Wno-source-uses-ompss-2 -o - %s
// expected-no-diagnostics

void foo() {
  #pragma oss task
  {}
}
