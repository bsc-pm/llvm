// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -verify -fsyntax-only %s

void foo() {
  __epi_1xf64 a;
  a[1]; // expected-error {{subscript of EPI vector}}
}
