// RUN: %libomp-cxx-compile-and-run

// Firstprivate objects of untied tasks are destroyed once, after the last part

#include <omp.h>
#include <stdio.h>

static int ctors, dtors, bad;

struct S {
  int v;
  S() : v(1) { __atomic_add_fetch(&ctors, 1, __ATOMIC_RELAXED); }
  S(const S &o) : v(o.v) { __atomic_add_fetch(&ctors, 1, __ATOMIC_RELAXED); }
  ~S() {
    v = 0;
    __atomic_add_fetch(&dtors, 1, __ATOMIC_RELAXED);
  }
};

int main() {
#pragma omp parallel num_threads(2)
#pragma omp single
  {
    S s;
    for (int i = 0; i < 100; i++) {
#pragma omp task untied firstprivate(s)
      for (int k = 0; k < 3; k++) {
        if (s.v != 1)
          __atomic_add_fetch(&bad, 1, __ATOMIC_RELAXED);
#pragma omp taskyield
      }
    }
#pragma omp taskwait
  }
  if (bad || ctors != dtors) {
    printf("failed: bad=%d ctors=%d dtors=%d\n", bad, ctors, dtors);
    return 1;
  }
  printf("passed\n");
  return 0;
}
