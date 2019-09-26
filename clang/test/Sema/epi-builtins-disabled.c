// RUN: %clang_cc1 -triple riscv64 -verify -fsyntax-only %s
void foo(void)
{
  long l = __builtin_epi_vsetvlmax(1, 1); // expected-error {{this builtin is only valid when using -mepi flag}}
}
