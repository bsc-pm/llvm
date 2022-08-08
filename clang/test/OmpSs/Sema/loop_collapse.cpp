// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

template<int N>
void foo() {
    #pragma oss taskloop collapse(N)
    for (int i = 0; i < 10; ++i) { // expected-error {{expected 2 for loops after '#pragma oss taskloop', but found only 1}}
    }
    #pragma oss taskloop collapse(N)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
        }
    }
}

// expected-note@+1 {{declared here}}
void bar(int n) {
    n = -1;
    const int m = 1;
    // #pragma oss task for collapse(-1)
    // for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop collapse(-1) // expected-error {{argument to 'collapse' clause must be a strictly positive integer value}}
    for (int i = 0; i < 10; ++i) {}
    // #pragma oss taskloop for collapse(-1)
    // for (int i = 0; i < 10; ++i) {}

    // #pragma oss task for collapse(m)
    // for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop collapse(m)
    for (int i = 0; i < 10; ++i) {}
    // #pragma oss taskloop for collapse(m)
    // for (int i = 0; i < 10; ++i) {}

    // #pragma oss task for collapse(m)
    // for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop collapse(n) // expected-error {{expression is not an integral constant expression}} expected-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
    for (int i = 0; i < 10; ++i) {}
    // #pragma oss taskloop for collapse(m)
    // for (int i = 0; i < 10; ++i) {}
    foo<2>(); // expected-note {{in instantiation of function template specialization 'foo<2>' requested here}}
}
