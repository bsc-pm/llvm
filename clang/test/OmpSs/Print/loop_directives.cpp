// RUN: %clang_cc1 -verify -fompss-2 -fompss-2=libnodes -ast-print %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
  #pragma oss task
  {
    #pragma oss task for
    for (int i = 0; i < 10; ++i) { }
    #pragma oss taskloop
    for (int i = 0; i < 10; ++i) { }
    #pragma oss taskloop for
    for (int i = 0; i < 10; ++i) { }
    #pragma oss taskiter
    for (int i = 0; i < 10; ++i) { }
    int j = 0;
    #pragma oss taskiter
    while (j > 10) {}
  }
}

// CHECK:     #pragma oss task
// CHECK-NEXT:         {
// CHECK-NEXT:             #pragma oss task for
// CHECK-NEXT:                 for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:                 }
// CHECK-NEXT:             #pragma oss taskloop
// CHECK-NEXT:                 for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:                 }
// CHECK-NEXT:             #pragma oss taskloop for
// CHECK-NEXT:                 for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:                 }
// CHECK-NEXT:             #pragma oss taskiter
// CHECK-NEXT:                 for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:                 }
// CHECK-NEXT:             int j = 0;
// CHECK-NEXT:             #pragma oss taskiter
// CHECK-NEXT:                 while (j > 10)
// CHECK-NEXT:                     {
// CHECK-NEXT:                     }
// CHECK-NEXT:         }

