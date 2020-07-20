// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
template<typename T> T foo() { return T(); }

void bar(int n) {
    n = -1;
    const int m = -1;
    int *vla[n];
    #pragma oss task for chunksize(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(vla[3]) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(foo<int *>()) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(foo<int *>()) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(m) // expected-error {{argument to 'chunksize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(m) // expected-error {{argument to 'chunksize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss task for chunksize(-1) // expected-error {{argument to 'chunksize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for chunksize(-1) // expected-error {{argument to 'chunksize' clause must be a non-negative integer value}}
    for (int i = 0; i < 10; ++i) {}
}
