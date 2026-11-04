// RUN: %libomp-compile-and-run

// Taskloops without iterations

#include <omp.h>
#include <stdio.h>

int main() {
  volatile int n = 0;
  int count = 0;
#pragma omp parallel num_threads(2)
#pragma omp single
  for (int r = 0; r < 10; r++) {
#pragma omp taskloop
    for (int i = 0; i < n; i++) {
#pragma omp atomic
      count++;
    }
  }
  if (count != 0) {
    printf("failed: count = %d\n", count);
    return 1;
  }
  printf("passed\n");
  return 0;
}
