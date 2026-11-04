// RUN: %libomp-compile-and-run
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8
// UNSUPPORTED: clang-5, clang-6, clang-7, clang-8, clang-9, clang-10, clang-11,
// UNSUPPORTED: clang-12, clang-13
// XFAIL: icc

// Undeferred tasks waiting for their child tasks

#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main() {
  int bodies = 0, childs = 0, x = 0;
#pragma omp parallel num_threads(2)
#pragma omp single
  for (int i = 0; i < 20; i++) {
#pragma omp task if (0) shared(bodies, childs)
    {
#pragma omp atomic
      bodies++;
#pragma omp task shared(childs)
      {
        usleep(1000);
#pragma omp atomic
        childs++;
      }
#pragma omp taskwait
    }
#pragma omp task if (0) shared(bodies, childs, x)
    {
#pragma omp atomic
      bodies++;
#pragma omp task depend(out : x) shared(childs)
      {
        usleep(1000);
#pragma omp atomic
        childs++;
      }
#pragma omp taskwait depend(in : x)
    }
  }
  if (bodies != 40 || childs != 40) {
    printf("failed: bodies = %d, childs = %d\n", bodies, childs);
    return 1;
  }
  printf("passed\n");
  return 0;
}
