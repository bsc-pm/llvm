// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

struct S {
  int x;
} s;  // expected-note {{'s' defined here}}

void bar() {}

void foo(int *x) {
  #pragma oss task reduction(+ : s.x, *x, bar) // expected-error 3 {{expected variable name}}
  {}
  #pragma oss task reduction(+ : x[0], [10]x, x[:1]) // expected-error 3 {{expected variable name}}
  {}
  #pragma oss task reduction(+ : s) // expected-error {{list item of type 'struct S' is not valid for specified reduction operation: unable to provide default initialization value}}
  {}
}
