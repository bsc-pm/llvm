// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -verify -fsyntax-only %s

void foo(void)
{
  __epi_f32 va;
  &va; // expected-error {{address of EPI vector requested}}
}

void bar(void *p)
{
  *(__epi_f32 *)p; // expected-error {{pointer to EPI vector type '__epi_f32' (vector of 'float') is invalid}}
}
