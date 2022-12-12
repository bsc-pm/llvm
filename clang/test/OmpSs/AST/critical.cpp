// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
void foo() {
  #pragma oss critical(asdf)
  {}
}

void bar() {
  #pragma oss critical(asdf)
  {}
  foo<int>();
}

// CHECK: OSSCriticalDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: OSSCriticalDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: OSSCriticalDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>

