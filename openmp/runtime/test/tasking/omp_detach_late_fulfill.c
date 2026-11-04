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

// Detached tasks fulfilled after their body, from a task that goes on
// creating tasks and from a nested parallel region

#include <omp.h>
#include <stdio.h>

int main() {
  int done = 0, childs = 0, levels = 0;
#pragma omp parallel num_threads(2)
#pragma omp single
  {
    for (int i = 0; i < 50; i++) {
      volatile int body = 0;
      omp_event_handle_t ev;
#pragma omp task detach(ev) shared(body)
      body = 1;
#pragma omp task shared(body, childs, levels) firstprivate(ev)
      {
        while (!body) {
#pragma omp taskyield
        }
        omp_fulfill_event(ev);
        if (omp_get_level() != 1) {
#pragma omp atomic
          levels++;
        }
#pragma omp task shared(childs)
        {
#pragma omp atomic
          childs++;
        }
      }
#pragma omp taskwait
      done++;
    }
    for (int i = 0; i < 50; i++) {
      volatile int body = 0;
      omp_event_handle_t ev;
#pragma omp task detach(ev) shared(body)
      body = 1;
      while (!body) {
#pragma omp taskyield
      }
#pragma omp parallel num_threads(2)
      if (omp_get_thread_num() == 0)
        omp_fulfill_event(ev);
#pragma omp taskwait
      done++;
    }
  }
  if (done != 100 || childs != 50 || levels) {
    printf("failed: done = %d, childs = %d, levels = %d\n", done, childs,
           levels);
    return 1;
  }
  printf("passed\n");
  return 0;
}
