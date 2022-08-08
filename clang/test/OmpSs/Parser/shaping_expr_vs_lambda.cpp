// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ferror-limit 100 %s

int array[10][20][30][40];

void foo1() {
  int kaka;
  #pragma oss task depend(in: [1][[2]array][3]array[3][4]) // expected-error {{OmpSs-2 array shaping is not allowed here}}
  {}
  #pragma oss task depend(in: ([1][2][3]array)[4:5][5])
  {}
  #pragma oss task depend(in: [1][2]array[4:5]) // expected-error {{OmpSs-2 array section is not allowed here}}
  {}
  #pragma oss task depend(in: [1][2 : 3]array) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  {}
  ([kaka]array)[x]; // expected-error {{OmpSs-2 array shaping is not allowed here}} expected-error {{use of undeclared identifier 'x'}}
  [77]array[x]; // expected-error {{OmpSs-2 array shaping is not allowed here}} expected-error {{use of undeclared identifier 'x'}}
}

void foo2(int x) {
  #pragma oss task depend(in : [](), [x](), [x](array) ) // expected-error {{expected body of lambda expression}} expected-error {{expected body of lambda expression}}
  {}
  #pragma oss task depend(in : [x + x]() {} ) // expected-error {{expected ',' or ']' in lambda capture list}}
  {}
  #pragma oss task depend(in : [x,](array),  [x,](array){}, [3](x, array)) // expected-error {{expected variable name or 'this' in lambda capture list}} expected-warning {{left operand of comma operator has no effect}}
  {}
  #pragma oss task depend(in : [x](int a) {} ) // expected-error {{expected addressable lvalue expression, array element, array shape or array section}}
  {}
  #pragma oss task depend(in : [3]array, [](){}) // expected-error {{expected addressable lvalue expression, array element, array shape or array section}}
  {}
  #pragma oss task depend(in : [3]]array, [[3]array) // expected-error {{expected expression}} expected-error {{expected variable name or 'this' in lambda capture list}} expected-error{{expected ')'}} expected-note{{to match this '('}}
  {}
}

void foo3(char *p) {
  #pragma oss task depend(in : [10](int *)p)
  {}
}

using T = int;
void foo3() {
  auto l = [](T()) {};
  auto l1 = [](T(1)) {}; // expected-error {{expected ')'}} expected-note {{to match this '('}}
}

