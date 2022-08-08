// RUN: %clang_cc1 -verify=oss -fompss-2 -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify=omp -fopenmp -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify=omp  -fompss-2 -fopenmp -ferror-limit 100 -o - %s

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
}
