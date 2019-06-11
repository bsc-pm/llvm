// RUN: %clang_cc1 -triple riscv64 -verify -fsyntax-only %s
void foo(void)
{
  long l = __builtin_epi_vreadvl(); // expected-error {{this builtin is only valid when using -mepi flag}}
}
