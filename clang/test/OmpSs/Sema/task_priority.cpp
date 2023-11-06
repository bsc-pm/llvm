// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -Wno-vla -std=c++11 %s
template<typename T> T foo() { return T(); }

#pragma oss task priority(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
void foo1(int n, int *vla[n]) {}

void bar(int n) {
    n = -1;
    const int m = -1;
    int *vla[n];
    #pragma oss task priority(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    {}
    #pragma oss task cost(foo<int *>()) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    {}
    #pragma oss task priority(foo<int>())
    {}
    #pragma oss task priority(n)
    {}
}
