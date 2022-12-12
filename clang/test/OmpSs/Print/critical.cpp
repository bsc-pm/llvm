// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
void foo() {
  T x;
  #pragma oss critical
  x++;
  #pragma oss critical(asdf)
  x++;
}

void bar() {
  int x;
  #pragma oss critical
  x++;
  #pragma oss critical(asdf)
  x++;
  foo<int>();
}

// CHECK:    #pragma oss critical
// CHECK:        x++;
// CHECK:    #pragma oss critical (asdf)
// CHECK:        x++;

// CHECK:    #pragma oss critical
// CHECK:        x++;
// CHECK:    #pragma oss critical (asdf)
// CHECK:        x++;
// CHECK:    #pragma oss critical
// CHECK:        x++;
// CHECK:    #pragma oss critical (asdf)
// CHECK:        x++;

