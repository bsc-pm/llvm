// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
struct S {
    operator int();
    S &operator+=(int);
};

struct P {
    int i;
};

template<typename T>
void bar() {
    P p;
    // expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
    #pragma oss taskloop
    for (p.i = 0; p.i < 10; ++p.i) {}
}

template<typename T>
struct Q {
    int i;
    void foo() {
        // expected-error@+3 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
        // expected-error@+2 {{induction variable cannot be a member in OmpSs-2 for loop}}
        #pragma oss taskloop
        for (i = 0; i < 10; ++i) {}
    }
};

void foo() {
    S s1;
    // expected-error@+2 {{variable must be of integer type}}
    #pragma oss taskloop
    for (S s = S(); s < s1; s += 1) {}
    #pragma oss taskloop in(i)
    for (int i = 0; i < 10; ++i) {}
    // expected-error@+2 {{statement after '#pragma oss taskloop' must be a for loop}}
    #pragma oss taskloop
    if (true) {}

    bar<int>(); // expected-note {{in instantiation of function template specialization 'bar<int>' requested here}}
    Q<int> q;
    q.foo(); // expected-note {{in instantiation of member function 'Q<int>::foo' requested here}}

    // Check we cleanup late parsed tokens even if init-stmt is not well-formed
    #pragma oss taskloop
    for ( = ; i < 10; ++i) { // expected-error 2 {{expected expression}} expected-error 2 {{use of undeclared identifier 'i'}}
        #pragma oss taskloop
        for (int j = 0; j < 10; ++j)
        ;
    }
}
