// RUN: %libomp-compile && env OMP_NUM_THREADS='4' %libomp-run
// RUN: %libomp-compile && env OMP_NUM_THREADS='1' %libomp-run

// A task opens a nested parallel while its child tasks run elsewhere. Each
// single construct switches the task team slot.

#include <omp.h>
#include <stdio.h>

#define REPS 20
#define NTASKS 16
#define NCHILDS 8
#define NINNER 4

static int count;

static void nested(void) {
#pragma omp parallel
#pragma omp single
  for (int i = 0; i < NINNER; i++) {
#pragma omp task
    {
#pragma omp atomic
      count++;
    }
  }
}

int main() {
  int expected = REPS * NTASKS * (NCHILDS + 1) * NINNER;
#pragma omp parallel
  for (int r = 0; r < REPS; r++) {
#pragma omp single
    for (int t = 0; t < NTASKS; t++) {
#pragma omp task
      {
        for (int c = 0; c < NCHILDS; c++) {
#pragma omp task
          nested();
        }
        nested();
      }
    }
  }
  if (count != expected) {
    printf("failed: count = %d, expected %d\n", count, expected);
    return 1;
  }
  printf("passed\n");
  return 0;
}
