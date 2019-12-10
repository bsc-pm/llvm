// RUN: %clang_cc1 -triple riscv64 -mepi -verify -fsyntax-only %s

void foo() {
  __epi_1xf64 a;
  a[1]; // expected-error {{subscript of EPI vector}}
}
