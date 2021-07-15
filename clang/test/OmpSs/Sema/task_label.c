// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

const char *s = "asdf";
#pragma oss task label(s)
void foo(void);

#pragma oss task label(p)
void foo1(const char *p);

void bar() {
    const char *o = "blabla";
    #pragma oss task label(o)
    #pragma oss task label("L1")
    {}
    const int *pint = 0;
    #pragma oss task label(pint) // expected-warning {{incompatible pointer types initializing 'const char *' with an expression of type 'const int *'}}
    {}
}
