// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
void foo() {
  int x;
  #pragma oss task
  #pragma oss critical(asdf)
  x++;
}

void bar() {
  int x;
  #pragma oss critical(asdf)
  x++;
  foo<int>();
}

// CHECK: OSSCriticalDirective
// CHECK: OSSCriticalDirective
// CHECK-NOT: OSSFirstprivateClause
// CHECK: OSSCriticalDirective
