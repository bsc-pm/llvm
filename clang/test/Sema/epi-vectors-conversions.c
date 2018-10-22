// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -verify -fsyntax-only %s

void foo(void) {
  __epi_i32 vc1;
  __epi_i64 vc2;

  vc1 = vc2; // expected-error-re {{assigning to {{.*}} from incompatible type {{.*}}}}
}

