// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -verify -fsyntax-only %s

int foo1(void)
{
  __epi_2xf32 va;
  return sizeof(va); // expected-error {{invalid application of 'sizeof' to an EPI vector type}}
}

int foo2(void)
{
  return sizeof(__epi_2xf32); // expected-error {{invalid application of 'sizeof' to an EPI vector type}}
}
