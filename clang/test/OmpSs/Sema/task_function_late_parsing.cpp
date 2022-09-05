// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - %s
// expected-no-diagnostics

// Late parsing allows to find late defined
// variables. This tests must not emit any diagnostics

struct S {
  #pragma oss task in([1]x)
  void foo(int *y = x);
  #pragma oss task in([1]x)
  void foo1(int *y);
  #pragma oss task in([1]x)
  void foo2(int *y);
  #pragma oss task in([1]x)
  void foo3(int *y = x) {}
  #pragma oss task in([1]x)
  void foo4(int *y) {}
  #pragma oss task in([1]x)
  void foo5(int *y) {}
  static int *x;
};
