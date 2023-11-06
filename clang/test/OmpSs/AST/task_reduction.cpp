// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
void foo(T x) {
  #pragma oss task reduction(max : x)
  {}
}

void bar() {
  foo<int>(1);
}


// CHECK: OSSTaskDirective {{[^ ]*}} <line:{{[^ ]*}}:{{[^ ]*}}, col:{{[^ ]*}}>
// CHECK-NEXT: OSSReductionClause {{[^ ]*}} <col:{{[^ ]*}}, col:{{[^ ]*}}>
// CHECK-NEXT: DeclRefExpr {{[^ ]*}} <col:36> 'T' lvalue ParmVar {{[^ ]*}} 'x' 'T'

// CHECK: OSSTaskDirective {{[^ ]*}} <line:{{[^ ]*}}:{{[^ ]*}}, col:{{[^ ]*}}>
// CHECK-NEXT: OSSReductionClause {{[^ ]*}} <col:{{[^ ]*}}, col:{{[^ ]*}}>
// CHECK-NEXT: DeclRefExpr {{[^ ]*}} <col:{{[^ ]*}}> 'int' lvalue ParmVar {{[^ ]*}} 'x' 'int'
// CHECK-NEXT: OSSSharedClause {{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[^ ]*}} <col:{{[^ ]*}}> 'int' lvalue ParmVar {{[^ ]*}} 'x' 'int'
