// RUN: %clangXX %openmp_flags %flags -c %s -DCTOR -o %t-ctor.o
// RUN: %clangXX %openmp_flags %flags -c %s -o %t-omp.o
// RUN: %clangXX %openmp_flags %flags %t-ctor.o %t-omp.o -o %t && %libomp-run

// OpenMP used from a global constructor of another translation unit that
// runs first

#include <stdio.h>

int foo();

#ifdef CTOR
static int result = foo();

int main() {
  if (result != 4950) {
    printf("failed: result = %d\n", result);
    return 1;
  }
  printf("passed\n");
  return 0;
}
#else
int foo() {
  int sum = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : sum) num_threads(2)
  for (int i = 0; i < 100; i++)
    sum += i;
  return sum;
}
#endif
