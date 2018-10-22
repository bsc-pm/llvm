// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -ast-print %s \
// RUN:    | FileCheck %s

void foo(void)
{
  // CHECK: __epi_f16 x1;
  __epi_f16 x1;

  // CHECK: __epi_f32 x2;
  __epi_f32 x2;

  // CHECK: __epi_f64 x3;
  __epi_f64 x3;
  //
  // CHECK: __epi_i8 x4;
  __epi_i8 x4;

  // CHECK: __epi_i16 x5;
  __epi_i16 x5;

  // CHECK: __epi_i32 x6;
  __epi_i32 x6;

  // CHECK: __epi_i64 x7;
  __epi_i64 x7;

  // CHECK: __epi_i1 x8;
  __epi_i1 x8;
}
