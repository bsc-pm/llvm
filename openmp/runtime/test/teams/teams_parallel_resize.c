// RUN: %libomp-compile-and-run

// gcc/icc target offloading is incompatible with libomp
// UNSUPPORTED: icc, gcc

// A parallel inside teams with fewer threads than the limit, then a full one
// with a barrier

#include <omp.h>
#include <stdio.h>

int main() {
  int a = 0, b = 0;
  for (int i = 0; i < 10; i++) {
#pragma omp teams num_teams(1) thread_limit(3)
    {
#pragma omp parallel num_threads(2)
      {
#pragma omp atomic
        a++;
      }
#pragma omp parallel
      {
#pragma omp atomic
        b++;
#pragma omp barrier
#pragma omp atomic
        b++;
      }
    }
  }
  if (a != 20 || b != 60) {
    printf("failed: a = %d, b = %d\n", a, b);
    return 1;
  }
  printf("passed\n");
  return 0;
}
