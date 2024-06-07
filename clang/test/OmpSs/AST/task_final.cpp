// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
T *foo(T *t) {
  #pragma oss task final(*t)
  {}
  return t;
}

void bar() {
  float *p;
  float *j = foo(p);
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSFinalClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'T' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'T *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'T *' lvalue ParmVar {{[a-z0-9]+}} 't' 'T *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSFinalClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'bool' <FloatingToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'float' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'float' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'float *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'float *' lvalue ParmVar {{[a-z0-9]+}} 't' 'float *'

