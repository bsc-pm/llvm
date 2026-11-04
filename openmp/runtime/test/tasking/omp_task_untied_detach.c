// RUN: %libomp-compile-and-run

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

// An untied detached task is only detached after its last part

#include <omp.h>
#include <stdio.h>

int main() {
  int done = 0;
#pragma omp parallel num_threads(2)
#pragma omp single
  for (int i = 0; i < 100; i++) {
    omp_event_handle_t ev;
#pragma omp task untied detach(ev) shared(done)
    {
#pragma omp taskyield
#pragma omp atomic
      done++;
      omp_fulfill_event(ev);
    }
#pragma omp taskwait
  }
  if (done != 100) {
    printf("failed: done = %d\n", done);
    return 1;
  }
  printf("passed\n");
  return 0;
}
