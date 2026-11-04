// RUN: %libomp-compile-and-run

// Serialized untied tasks (outside parallel, if(0), final) run all their parts

#include <omp.h>
#include <stdio.h>

#define BODY                                                                   \
  for (int k = 0; k < 3; k++) {                                                \
    _Pragma("omp atomic") count++;                                             \
    _Pragma("omp taskyield")                                                   \
  }

static int count;

int main() {
#pragma omp task untied
  BODY
#pragma omp taskwait

#pragma omp parallel num_threads(2)
#pragma omp single
  {
#pragma omp task untied if (0)
    BODY
#pragma omp task final(1)
    {
#pragma omp task untied
      BODY
    }
#pragma omp taskwait
  }
  if (count != 9) {
    printf("failed: count = %d\n", count);
    return 1;
  }
  printf("passed\n");
  return 0;
}
