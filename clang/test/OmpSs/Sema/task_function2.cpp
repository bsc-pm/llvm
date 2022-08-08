// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - %s

template<typename T>
struct S {
    int x;

    #pragma oss task final(this) in(*y) // expected-warning {{'this' pointer cannot be null in well-defined C++ code; pointer may be assumed to always convert to true}}
    void foo(int *y) {
    }
};

struct P {
    int x;

    #pragma oss task device(cuda) // expected-error {{device(cuda) is not allowed in member functions}}
    void foo() {}
};

void bar() {
    int x;
    S<int> s; // expected-note {{in instantiation of template class 'S<int>' requested here}}
    s.foo(&x);
    P p;
    p.foo();
}
