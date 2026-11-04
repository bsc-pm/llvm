// RUN: %libomp-compile && env OMP_ENABLE_COMPAT=1 %libomp-run

// gcc/icc target offloading is incompatible with libomp
// UNSUPPORTED: icc, gcc

// __kmpc_dist_dispatch_init_4 (used by other compilers) inside teams

#include <omp.h>
#include <stdio.h>

typedef struct {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  const char *psource;
} ident_t;

static ident_t loc = {0, 2, 0, 0, ";kmp_dist_dispatch_init.c;main;30;1;;"};

extern int __kmpc_global_thread_num(ident_t *);
extern void __kmpc_dist_dispatch_init_4(ident_t *, int, int, int *, int, int,
                                        int, int);
extern int __kmpc_dispatch_next_4(ident_t *, int, int *, int *, int *, int *);

int main() {
  int sum = 0;
#pragma omp teams num_teams(1)
#pragma omp parallel num_threads(2) reduction(+ : sum)
  {
    int gtid = __kmpc_global_thread_num(&loc);
    int last = 0, lb, ub, st;
    // 35: kmp_sch_dynamic_chunked
    __kmpc_dist_dispatch_init_4(&loc, gtid, 35, &last, 0, 99, 1, 1);
    while (__kmpc_dispatch_next_4(&loc, gtid, &last, &lb, &ub, &st))
      for (int i = lb; i <= ub; i++)
        sum += i;
  }
  if (sum != 4950) {
    printf("failed: sum = %d\n", sum);
    return 1;
  }
  printf("passed\n");
  return 0;
}
