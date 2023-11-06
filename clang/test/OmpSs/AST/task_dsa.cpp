// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
T foo(T t) {
  #pragma oss task
  {
    T x = t++;
    #pragma oss task shared(x)
    { x++; }
  }
  #pragma oss taskwait
  return t;
}

void bar()
{
  int i = foo(2);
  float j = foo(2.0f);
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:11, col:19>
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:9:13, col:31>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:22, col:30>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:29> 'T' lvalue Var {{[a-z0-9]+}} 'x' 'T'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:11, col:19>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:8:11> 'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:9:13, col:31>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:22, col:30>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:29> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:11, col:19>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:8:11> 'float' lvalue ParmVar {{[a-z0-9]+}} 't' 'float'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:9:13, col:31>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:22, col:30>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:29> 'float' lvalue Var {{[a-z0-9]+}} 'x' 'float'

