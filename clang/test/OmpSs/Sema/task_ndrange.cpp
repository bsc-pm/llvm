// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

#pragma oss task device(cuda) ndrange() // expected-error {{expected expression}}
void foo1();
#pragma oss task device(cuda) ndrange(1, 1, 1, 2, 2) // expected-error {{'ndrange(1, argument-list)' clause requires 1 or 2 arguments in argument-list, but got 4}}
void foo2();
#pragma oss task device(cuda) ndrange(x) // expected-error {{expression is not an integral constant expression}} expected-note {{function parameter 'x' with unknown value cannot be used in a constant expression}}
void foo3(int x); // expected-note {{declared here}}
#pragma oss task device(cuda) ndrange(1, x, 1) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
void foo4(int *x);
#pragma oss task device(cuda) ndrange(1, *x, 1) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
template<typename T>
void foo5(T *x);
#pragma oss task device(cuda) ndrange(1, -1, -1) // expected-error 2 {{argument to 'ndrange' clause must be a strictly positive integer value}}
void foo6();
#pragma oss task device(cuda) ndrange(1, N, N) // expected-error 4 {{argument to 'ndrange' clause must be a strictly positive integer value}}
template<int N>
void foo7();


void bar() {
    int x;
    int *p;
    foo5(&x);
    foo5(&p); // expected-note {{in instantiation of function template specialization 'foo5<int *>' requested here}}
    foo7<-1>(); // expected-note {{in instantiation of function template specialization 'foo7<-1>' requested here}}
    foo7<0>(); // expected-note {{in instantiation of function template specialization 'foo7<0>' requested here}}
    foo7<1>();
    #pragma oss task device(cuda) ndrange(1, 1, 1) // expected-error {{unexpected OmpSs-2 clause 'ndrange' in directive '#pragma oss task'}}
    {}
}


