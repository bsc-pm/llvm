// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

const char *s = "asdf";
#pragma oss task label(s) // expected-error {{expression is not a string literal}}
void foo1();

#pragma oss task label(p) // expected-error {{expression is not a string literal}}
void foo1(const char *p);

void bar() {
    const char *o = "blabla";
    #pragma oss task label(o) // expected-error {{expression is not a string literal}}
    #pragma oss task label("L1")
    {}
}
