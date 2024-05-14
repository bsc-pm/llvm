// RUN: %clang_cc1 -verify -fompss-2 -fompss-2=libnodes -ferror-limit 100 -o - -std=c++11 %s
void foo(int n) {
  #pragma oss taskiter unroll(0) // expected-error {{argument to 'unroll' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i) { }
  #pragma oss taskiter unroll(-1) // expected-error {{argument to 'unroll' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i) { }
}

