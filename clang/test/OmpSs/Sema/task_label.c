// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

const char *s = "asdf";
#pragma oss task label(s) // expected-error {{this variable cannot be used here}}
void foo(void);

#pragma oss task label(p) // expected-error {{this variable cannot be used here}}
void foo1(const char *p);

void bar() {
    const char *o = "blabla";
    #pragma oss task label(o) // expected-error {{this variable cannot be used here}}
    #pragma oss task label("L1")
    {}
    const int *pint = 0;
    #pragma oss task label(pint) // expected-warning {{incompatible pointer types initializing 'const char *' with an expression of type 'const int *'}} expected-error {{this variable cannot be used here}}
    {}
    const char *a;
    #pragma oss task label("x", a)
    {}
    #pragma oss task label("x", a, // expected-error {{expected ')'}} expected-note {{to match this '('}}
    {}
    #pragma oss task label("x", a, a) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    {}
}
