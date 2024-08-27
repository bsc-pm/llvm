// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

const char *s = "asdf";
#pragma oss task label(s) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
void foo(void);

#pragma oss task label(p) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
void foo1(const char *p);

void bar() {
    const char *o = "blabla";
    #pragma oss task label(o) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    #pragma oss task label("L1")
    {}
    const int *pint = 0;
    #pragma oss task label(pint) // expected-warning {{incompatible pointer types initializing 'const char *' with an expression of type 'const int *'}} expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    {}
    #pragma oss task label(0 ? "a" : "b") // constant evaluated ConditionalOperator condition are valid
    {}
    const char *a;
    #pragma oss task label("x", a)
    {}
    #pragma oss task label("x", a, // expected-error {{expected ')'}} expected-note {{to match this '('}}
    {}
    #pragma oss task label("x", a, a) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    {}
    #pragma oss task label(0 ? a : a) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    {}
    #pragma oss task label(1 ? a : a) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    {}
}
