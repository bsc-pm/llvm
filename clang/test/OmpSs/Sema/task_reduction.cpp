// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

// UNSUPPORTED: true

struct S {
  int x;
} s;

struct U {
  ~U();
  operator int();
  U &operator=(int i);
} u;

void bar() {}

void foo(int *x) {
  #pragma oss task reduction(+ : s.x, *x, bar) // expected-error 3 {{expected variable name or array shaping}}
  {}
  #pragma oss task reduction(+ : x[0], [10]x, x[:1]) // expected-error 2 {{expected variable name or array shaping}}
  {}
  #pragma oss task reduction(+ : s) // expected-error {{invalid operands to binary expression ('struct S' and 'struct S')}}
  {}
  int i;
  #pragma oss task reduction(::+ : i) // expected-error {{expected unqualified-id}}
  {}
  #pragma oss task reduction( // expected-error {{expected unqualified-id}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma oss task reduction(+ // expected-warning {{missing ':' after reduction identifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma oss task reduction(+ : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma oss task reduction( : // expected-error {{expected unqualified-id}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma oss task reduction(+ : x // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{invalid operands to binary expression ('int *' and 'int *')}}
  {}
  #pragma oss task reduction(+ : ) // expected-error {{expected expression}}
  {}
}

template<typename T, class  ...Types>
void foo1(T *t, Types ...args) {
  #pragma oss task reduction(+ : t[0], [10]t, t[:1], args) // expected-error 2 {{expected variable name or array shaping}} // expected-error {{variadic templates are not allowed in OmpSs-2 clauses}}
  {}
}

void foo2() {
  foo1<int, int>(nullptr, 0); // expected-note {{in instantiation of function template specialization 'foo1<int, int>' requested here}}
  U &ru = u;
  #pragma oss task reduction(-: u, ru) // expected-error 2 {{non-PODs value are not allowed in reductions}}
  {}
}

