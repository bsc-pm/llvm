// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>

int main() {
  int i;

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &i, sizeof i);

  // CHECK-NOT: omp{{v?}}target
#pragma omp target data map(alloc : i)
#pragma omp target data map(present, alloc : i)
  ;

  // CHECK: i is present
  fprintf(stderr, "i is present\n");

  // CHECK: omp{{v?}}target message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: omp{{v?}}target fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target data map(present, alloc : i)
  ;

  // CHECK-NOT: i is present
  fprintf(stderr, "i is present\n");

  return 0;
}
