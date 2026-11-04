// RUN: %libomp-compile-and-run

// Included child tasks run inside their final parent, which then creates more

#include <omp.h>
#include <stdio.h>

int main() {
  int count = 0;
#pragma omp parallel num_threads(2)
#pragma omp single
  {
    for (int i = 0; i < 64; i++) {
#pragma omp task final(1) shared(count)
      {
#pragma omp task shared(count)
        {
#pragma omp atomic
          count++;
        }
#pragma omp task shared(count)
        {
#pragma omp atomic
          count++;
        }
      }
    }
#pragma omp taskwait
  }
  if (count != 128) {
    printf("failed: count = %d\n", count);
    return 1;
  }
  printf("passed\n");
  return 0;
}
