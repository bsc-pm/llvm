// RUN: %clang_cc1 -verify -Wsource-uses-ompss-2 -o - %s

void foo() {
  #pragma oss task // expected-warning {{unexpected '#pragma oss ...' in program}}
  {}
}
