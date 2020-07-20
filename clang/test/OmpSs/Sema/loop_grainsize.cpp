// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
template<typename T> T foo() { return T(); }

void bar(int n) {
    n = -1;
    const int m = -1;
    int *vla[n];
    #pragma oss taskloop grainsize(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(foo<int *>()) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(foo<int *>()) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(m) // expected-error {{argument to 'grainsize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(m) // expected-error {{argument to 'grainsize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(-1) // expected-error {{argument to 'grainsize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(-1) // expected-error {{argument to 'grainsize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
}
