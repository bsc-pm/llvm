// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
template<typename T> T foo() { return T(); }

void bar(int n) {
    n = -1;
    const int m = -1;
    int *vla[n];
    #pragma oss task cost(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    {}
    #pragma oss task cost(foo<int *>()) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    {}
    #pragma oss task cost(foo<int>())
    {}
    #pragma oss task cost(n)
    {}
    #pragma oss task cost(m) // expected-error {{argument to 'cost' clause must be a non-negative integer value}}
    {}
    #pragma oss task cost(-1) // expected-error {{argument to 'cost' clause must be a non-negative integer value}}
    {}
}
