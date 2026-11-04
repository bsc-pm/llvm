// RUN: %libomp-compile && env OMP_NUM_THREADS='3' %libomp-run
// RUN: %libomp-compile && env OMP_NUM_THREADS='1' %libomp-run

// Checked gcc 10.1 still does not support detach clause on task construct.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10
// gcc 11 introduced detach clause, but gomp interface in libomp has no support
// XFAIL: gcc-11, gcc-12
// clang supports detach clause since version 11.
// UNSUPPORTED: clang-10, clang-9, clang-8, clang-7
// icc compiler does not support detach clause.
// UNSUPPORTED: icc
// There is no support for detached tasks with gcc
// UNSUPPORTED: ompv-gomp

// The end of a taskgroup waits for its detached tasks

#include <omp.h>
#include <stdio.h>

#define REPS 1000

int main() {
  int errors = 0;
  #pragma omp parallel
  #pragma omp single
  for (int r = 0; r < REPS; r++) {
    omp_event_handle_t ev;
    int completed = 0;
    #pragma omp taskgroup
    {
      if (r & 1) {
        // Fulfilled after the body (proxy task path)
        #pragma omp task detach(ev) if(0) shared(completed)
        { completed = 1; }
        omp_fulfill_event(ev);
      } else {
        // Fulfilled before or after the body
        #pragma omp task detach(ev) shared(completed)
        { completed = 1; }
        #pragma omp task firstprivate(ev)
        omp_fulfill_event(ev);
      }
    }
    if (!completed)
      errors++;
  }
  if (errors) {
    printf("failed: %d detached tasks not completed after taskgroup\n", errors);
    return 1;
  }
  printf("passed\n");
  return 0;
}
