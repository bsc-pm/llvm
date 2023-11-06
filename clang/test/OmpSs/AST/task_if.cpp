// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
T *foo(T *t) {
  #pragma oss task if(*t)
  {}
  return t;
}

void bar() {
  float *p;
  float *j = foo(p);
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:11, col:26>
// CHECK-NEXT: OSSIfClause {{[a-z0-9]+}} <col:20, col:25>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:23, col:24> '<dependent type>' prefix '*' cannot overflow
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:24> 'T *' lvalue ParmVar {{[a-z0-9]+}} 't' 'T *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:11, col:26>
// CHECK-NEXT: OSSIfClause {{[a-z0-9]+}} <col:20, col:25>
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:23, col:24> 'bool' <FloatingToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:23, col:24> 'float' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:23, col:24> 'float' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:24> 'float *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:24> 'float *' lvalue ParmVar {{[a-z0-9]+}} 't' 'float *'

