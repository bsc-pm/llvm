// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - %s

struct P {
  int x;
};

struct S {
  P p;
};

#pragma oss task inout(a.p.x) // expected-error {{expected lvalue reference, global variable, dereference, array element, array section or array shape}}
void foo(struct S a) {
}

