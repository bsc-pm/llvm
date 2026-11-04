// RUN: %libomp-compile-and-run

// Tasks with large private data

#include <omp.h>
#include <stdio.h>

#define N 2048

int main() {
  int a[N];
  long s1 = 0, s2 = 0, expected = (long)N * (N - 1) / 2;
  for (int i = 0; i < N; i++)
    a[i] = i;
#pragma omp parallel num_threads(2)
#pragma omp single
  {
#pragma omp task firstprivate(a) shared(s1)
    for (int i = 0; i < N; i++)
      s1 += a[i];
#pragma omp taskloop firstprivate(a) reduction(+ : s2) num_tasks(4)
    for (int i = 0; i < N; i++)
      s2 += a[i];
#pragma omp taskwait
  }
  if (s1 != expected || s2 != expected) {
    printf("failed: s1 = %ld, s2 = %ld\n", s1, s2);
    return 1;
  }
  printf("passed\n");
  return 0;
}
