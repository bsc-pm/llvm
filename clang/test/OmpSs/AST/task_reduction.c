// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void foo(int x) {
  #pragma oss task reduction(max : x)
  {}
}

// CHECK: OSSReductionClause {{[^ ]*}} <col:{{[^ ]*}}, col:{{[^ ]*}}>
// CHECK-NEXT: DeclRefExpr {{[^ ]*}} <col:{{[^ ]*}}> 'int' lvalue ParmVar {{[^ ]*}} 'x' 'int'
// CHECK-NEXT: OSSSharedClause {{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[^ ]*}} <col:{{[^ ]*}}> 'int' lvalue ParmVar {{[^ ]*}} 'x' 'int'

