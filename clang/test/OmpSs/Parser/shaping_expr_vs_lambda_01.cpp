// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ferror-limit 100 %s

void foo(int *p) {
  // Shape
  #pragma oss task in([(1, 4)]p) // expected-warning {{left operand of comma operator has no effect}}
  {}
  // Lambda
  #pragma oss task in([10(int *)p) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Lambda
  #pragma oss task in([10](int *, )) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Shape
  #pragma oss task in([10](p, )) // expected-error {{expected expression}}
  {}
}

void foo1(int *p) {
  // Lambda
  #pragma oss task in([ // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Lambda
  #pragma oss task in([) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Lambda
  #pragma oss task in([10) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Lambda
  #pragma oss task in([]) // expected-error {{expected body of lambda expression}}
  {}
  // Shape
  #pragma oss task in([19]) // expected-error {{expected expression}}
  {}
  // Lambda
  #pragma oss task in([19]() // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Lambda
  #pragma oss task in([19]()) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Lambda
  #pragma oss task in([19](int * // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Shape
  #pragma oss task in([19](int *) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Shape
  #pragma oss task in([19](p // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Shape
  #pragma oss task in([19](p) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
}

template <typename T>
void bar() {
  // Lambda
  #pragma oss task in([10](T *, )) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  // Shape
  #pragma oss task in([10](S *, )) // expected-error {{use of undeclared identifier 'S'}} expected-error {{expected expression}} expected-error {{expected expression}}
  {}
  // Shape
  #pragma oss task in([10](S *) {}) // expected-error {{use of undeclared identifier 'S'}} expected-error {{expected expression}}
  {}
}

