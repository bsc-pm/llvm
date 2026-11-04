// RUN: %libomp-compile && env OMP_NUM_THREADS='8' %libomp-run
// RUN: %libomp-compile && env OMP_NUM_THREADS='3' %libomp-run
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8
// UNSUPPORTED: clang-5, clang-6, clang-7, clang-8, clang-9, clang-10, clang-11,
// UNSUPPORTED: clang-12, clang-13
// XFAIL: icc

// Producers finishing while the taskwait processes its dependences must not
// wake it up early. The first one is slower so the taskwait has to wait.

#include <omp.h>
#include <stdio.h>

#define REPS 2000
#define NDEPS 64

static int done[NDEPS];

static void work(int n) {
  volatile int x = 0;
  for (int i = 0; i < n; i++)
    x++;
}

int main() {
  int errors = 0;
#pragma omp parallel
#pragma omp single
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < NDEPS; i++) {
      done[i] = 0;
#pragma omp task depend(out : done[i]) firstprivate(i)
      {
        work(i == 0 ? 20000 : 50);
#pragma omp atomic write
        done[i] = 1;
      }
    }
#pragma omp taskwait depend(iterator(it = 0 : NDEPS), in : done[it])
    for (int i = 0; i < NDEPS; i++) {
      int d;
#pragma omp atomic read
      d = done[i];
      if (!d) {
        if (errors++ < 10)
          fprintf(stderr, "rep %d: task %d not completed\n", r, i);
      }
    }
  }
  if (errors) {
    printf("failed: %d\n", errors);
    return 1;
  }
  printf("passed\n");
  return 0;
}
