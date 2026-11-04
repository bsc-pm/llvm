// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run
// RUN: %libomp-compile && env OMP_CANCELLATION=true OMP_ENABLE_FREE_AGENTS=1 \
// RUN:   OMP_FREE_AGENT_TID_EMU=1 %libomp-run

// Barriers with tasks in flight in parallel regions that are cancelled and
// that could be but are not

#include <omp.h>
#include <stdio.h>

int main() {
  int x = 0, y = 0;
  for (int r = 0; r < 20; r++) {
#pragma omp parallel num_threads(2)
    {
#pragma omp task
      {
#pragma omp atomic
        x++;
      }
      if (omp_get_thread_num() == 1) {
#pragma omp cancel parallel
      }
#pragma omp barrier
    }
#pragma omp parallel num_threads(2)
    {
#pragma omp task
      {
#pragma omp atomic
        y++;
      }
      if (omp_get_thread_num() == 2) {
#pragma omp cancel parallel
      }
#pragma omp barrier
    }
  }
  // Cancelled regions may discard their tasks
  if (x > 40 || y != 40) {
    printf("failed: x = %d, y = %d\n", x, y);
    return 1;
  }
  printf("passed\n");
  return 0;
}
