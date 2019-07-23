// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int foo() {
    int x;
    #pragma oss task depend(in : x x) // expected-error {{expected ',' or ')' in 'depend' clause}}
    {}
}
