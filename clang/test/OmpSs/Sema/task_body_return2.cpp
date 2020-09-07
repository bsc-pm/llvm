// RUN: %clang_cc1 -verify -x c++ -std=c++11 -fompss-2 -ferror-limit 100 -o - %s

// This test checks that copy assignment method does not trigger a ompss-2
// return not allowed diagnostic.

struct A {
  A();
  ~A();
};
#pragma oss declare reduction(red1 : A : omp_out) initializer(omp_priv = A())
#pragma oss declare reduction(red2 : A : omp_out) initializer(omp_priv = omp_priv = A()) // expected-warning {{variable 'omp_priv' is uninitialized when used within its own initialization}}
