// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

// UNSUPPORTED: true

struct S {
  int x;
} s;  // expected-note {{'s' defined here}}

int **get() {return 0;}

void bar() {}

void foo(int *x) {
  #pragma oss task reduction(+ : s.x, *x, bar) // expected-error 3 {{expected variable name or array shaping}}
  {}
  #pragma oss task reduction(+ : x[0], [10]x, x[:1]) // expected-error 2 {{expected variable name or array shaping}}
  {}
  #pragma oss task reduction(+ : s) // expected-error {{list item of type 'struct S' is not valid for specified reduction operation: unable to provide default initialization value}}
  {}
}

int main() {
    int **p;

    #pragma oss task reduction(+: [4](p[4])) // expected-error {{expected variable name as a base of the array shaping}}
    {}
    #pragma oss task reduction(+: [4](get()[3])) // expected-error {{expected variable name as a base of the array shaping}}
    {}
    #pragma oss task reduction(+: ([2]p)[4]) // expected-error {{expected variable name or array shaping}}
    {}
}
