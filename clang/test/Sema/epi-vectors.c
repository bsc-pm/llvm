// RUN: %clang_cc1 -triple riscv64 -target-feature +epi -ast-print %s \
// RUN:    | FileCheck %s

// Full register aliases
void foo(void)
{
  // CHECK: __epi_4xf16 x1;
  __epi_f16 x1;

  // CHECK: __epi_2xf32 x2;
  __epi_f32 x2;

  // CHECK: __epi_1xf64 x3;
  __epi_f64 x3;
  //
  // CHECK: __epi_8xi8 x4;
  __epi_i8 x4;

  // CHECK: __epi_4xi16 x5;
  __epi_i16 x5;

  // CHECK: __epi_2xi32 x6;
  __epi_i32 x6;

  // CHECK: __epi_1xi64 x7;
  __epi_i64 x7;

  // CHECK: __epi_1xi1 x8;
  __epi_i1 x8;
}

// All lmul possibilities so far.
void bar(void) {
  // CHECK: __epi_1xi64 x0;
  __epi_1xi64 x0;

  // CHECK: __epi_1xf64 x1;
  __epi_1xf64 x1;

  // CHECK: __epi_2xi64 x2;
  __epi_2xi64 x2;

  // CHECK: __epi_2xf64 x3;
  __epi_2xf64 x3;

  // CHECK: __epi_4xi64 x4;
  __epi_4xi64 x4;

  // CHECK: __epi_4xf64 x5;
  __epi_4xf64 x5;

  // CHECK: __epi_8xi64 x6;
  __epi_8xi64 x6;

  // CHECK: __epi_8xf64 x7;
  __epi_8xf64 x7;

  // CHECK: __epi_2xi32 x8;
  __epi_2xi32 x8;

  // CHECK: __epi_2xf32 x9;
  __epi_2xf32 x9;

  // CHECK: __epi_4xi32 x10;
  __epi_4xi32 x10;

  // CHECK: __epi_4xf32 x11;
  __epi_4xf32 x11;

  // CHECK: __epi_8xi32 x12;
  __epi_8xi32 x12;

  // CHECK: __epi_8xf32 x13;
  __epi_8xf32 x13;

  // CHECK: __epi_16xi32 x14;
  __epi_16xi32 x14;

  // CHECK: __epi_16xf32 x15;
  __epi_16xf32 x15;

  // CHECK: __epi_4xi16 x16;
  __epi_4xi16 x16;

  // CHECK: __epi_4xf16 x17;
  __epi_4xf16 x17;

  // CHECK: __epi_8xi16 x18;
  __epi_8xi16 x18;

  // CHECK: __epi_8xf16 x19;
  __epi_8xf16 x19;

  // CHECK: __epi_16xi16 x20;
  __epi_16xi16 x20;

  // CHECK: __epi_16xf16 x21;
  __epi_16xf16 x21;

  // CHECK: __epi_32xi16 x22;
  __epi_32xi16 x22;

  // CHECK: __epi_32xf16 x23;
  __epi_32xf16 x23;

  // CHECK: __epi_8xi8 x24;
  __epi_8xi8 x24;

  // CHECK: __epi_16xi8 x25;
  __epi_16xi8 x25;

  // CHECK: __epi_32xi8 x26;
  __epi_32xi8 x26;

  // CHECK: __epi_64xi8 x27;
  __epi_64xi8 x27;

  // CHECK: __epi_1xi1 x28;
  __epi_1xi1 x28;

  // CHECK: __epi_2xi1 x29;
  __epi_2xi1 x29;

  // CHECK: __epi_4xi1 x30;
  __epi_4xi1 x30;

  // CHECK: __epi_8xi1 x31;
  __epi_8xi1 x31;

  // CHECK: __epi_16xi1 x32;
  __epi_16xi1 x32;

  // CHECK: __epi_32xi1 x33;
  __epi_32xi1 x33;

  // CHECK: __epi_64xi1 x34;
  __epi_64xi1 x34;
}
