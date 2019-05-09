// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -verify -fsyntax-only %s

void foo(void)
{
  __epi_2xf32 a;
  __epi_2xf32 *pv; // expected-error {{pointer to EPI vector type '__epi_2xf32' (vector of 'float') is invalid}}
  __epi_2xf32 &rv = a; // expected-error {{reference to EPI vector type '__epi_2xf32' (vector of 'float') is invalid}}
  __epi_2xf32 av[10]; // expected-error {{array of EPI vector type '__epi_2xf32' (vector of 'float') is invalid}}
}

struct A
{
  __epi_2xf32 x; // expected-error {{the EPI vector type '__epi_2xf32' (vector of 'float') cannot be used to declare a structure or union field}}
  static __epi_2xf32 y; // expected-error {{the EPI vector type '__epi_2xf32' (vector of 'float') cannot be used to declare a non-static data member}}
};

__epi_2xf32 x; // expected-error {{declaration of a variable with an EPI vector type not allowed at file scope}}
extern __epi_2xf32 y; // expected-error {{declaration of a variable with an EPI vector type not allowed at file scope}}
static __epi_2xf32 z; // expected-error {{declaration of a variable with an EPI vector type not allowed at file scope}}

void bar(void)
{
  static __epi_2xf32 a; // expected-error {{declaration of a variable with an EPI vector type cannot have 'static' storage duration}}
  extern __epi_2xf32 b; // expected-error {{declaration of a variable with an EPI vector type cannot have 'extern' linkage}}
}
