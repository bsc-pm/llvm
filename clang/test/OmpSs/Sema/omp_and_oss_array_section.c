// RUN: %clang_cc1 -verify=oss -fompss-2 -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify=omp -fopenmp -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify=omp  -fompss-2 -fopenmp -ferror-limit 100 -o - %s

#pragma oss task in(p[0;1])
void bar(int *p);

// This test checks that we parse the right section
// depending on the parsing context
void foo() {
  int array[10];
  array[0 : 1]; // oss-error {{OmpSs-2 array section is not allowed here}} omp-error {{OpenMP array section is not allowed here}}
  array[0 ; 1]; // oss-error {{OmpSs-2 array section is not allowed here}} omp-error {{expected ']'}} omp-error {{extraneous ']' before ';'}} omp-warning {{expression result unused}} omp-note {{to match this '['}}
  #pragma omp task depend(in: array[0:1], array[0;1]) // omp-error {{expected ']'}} omp-error {{expected ',' or ')' in 'depend' clause}} omp-error {{expected expression}} omp-note {{to match this '['}}
  {}
  #pragma oss task in(array[0:1], array[0;1])
  {}
  #pragma oss task in({ array[array[0]; array[1]], i=0;10 })
  {}
  #pragma oss taskloop in(array[0:1], array[0;1]) in({array[0:1],k=0;10}, {array[0;1],k=0;10})
  for (int i = 0; i < 10; ++i)
  {}
  #pragma oss taskloop collapse(1) out(array[1 ;3]) out({array[0:1],t=0;10}, {array[0;1],t=0;10})
  for (int i = 0; i < 10; ++i) {
  }
  #pragma oss taskloop collapse(2) out(array[1 ;3]) out({array[0:1],t=0;10}, {array[0;1],t=0;10})
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
    }
  }
  #pragma oss taskloop collapse(3) out(array[1 ;3]) out({array[0:1],t=0;10}, {array[0;1],t=0;10})
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) {
      }
    }
  }
}
