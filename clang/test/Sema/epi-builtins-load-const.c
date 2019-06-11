// RUN: %clang_cc1 -target-feature +epi -triple riscv64 -verify -fsyntax-only %s

// expected-no-diagnostics

const double *pd;
const float *pf;

const long *pl;
const int *pi;
const short *ps;
const signed char *pc;

void foo(long gvl)
{
  (void)__builtin_epi_vload_1xf64(pd, gvl);
  (void)__builtin_epi_vload_2xf32(pf, gvl);

  (void)__builtin_epi_vload_1xi64(pl, gvl);
  (void)__builtin_epi_vload_2xi32(pi, gvl);
  (void)__builtin_epi_vload_4xi16(ps, gvl);
  (void)__builtin_epi_vload_8xi8(pc, gvl);
}
