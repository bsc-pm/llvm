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

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'T' lvalue Var {{[a-z0-9]+}} 'x' 'T'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{.*}}:{{.*}}> 'float' lvalue ParmVar {{[a-z0-9]+}} 't' 'float'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:29> 'float' lvalue Var {{[a-z0-9]+}} 'x' 'float'

