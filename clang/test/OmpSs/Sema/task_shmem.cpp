// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

#pragma oss task device(cuda) shmem(1) // expected-error {{shmem clause requires ndrange or grid clause to be specified}}
void foo0();
#pragma oss task device(cuda) ndrange(1, 1, 1) shmem() // expected-error {{expected expression}}
void foo1();
#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(-1) // expected-error {{argument to 'shmem' clause must be a non-negative integer value}}
void foo2();
#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(x)
void foo3(int x);
#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(x) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
void foo4(int *x);
#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(x) // expected-error {{expression must have integral or unscoped enumeration type, not 'int *'}}
template<typename T>
void foo5(T x);
#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(N) // expected-error {{argument to 'shmem' clause must be a non-negative integer value}}
template<int N>
void foo6();

void bar() {
    int x;
    int *p;
    foo5(x);
    foo5(p); // expected-note {{in instantiation of function template specialization 'foo5<int *>' requested here}}
    foo6<-1>(); // expected-note {{in instantiation of function template specialization 'foo6<-1>' requested here}}
    foo6<1>();
    #pragma oss task device(cuda) ndrange(1, 1, 1) shmem(x) // expected-error {{unexpected OmpSs-2 clause 'ndrange' in directive '#pragma oss task'}} expected-error {{unexpected OmpSs-2 clause 'shmem' in directive '#pragma oss task'}}
    {}
    #pragma oss task device(cuda) ndrange(1, 1, 1) shmem(-1) // expected-error {{unexpected OmpSs-2 clause 'ndrange' in directive '#pragma oss task'}} expected-error {{unexpected OmpSs-2 clause 'shmem' in directive '#pragma oss task'}}
    {}
}


