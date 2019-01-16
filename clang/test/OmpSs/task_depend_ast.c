// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int main() {
    int a;
    int b[5];
    int *c;
    #pragma oss task depend(in : a, b[0], *c)
    { a = b[0] = *c; }
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'b' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 0
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'c' 'int *'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'b' 'int [5]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'c' 'int *'

int *p;
int foo() {
  #pragma oss task depend(inout: *p)
  {
    #pragma oss task
    {
      *p = 0;
    }
  }
  return 0;
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, line:{{[a-z0-9]+}}:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'

int array[5];
int foo1() {
  #pragma oss task depend(inout: array[2])
  {
    #pragma oss task
    {
      array[2] = 0;
    }
  }
  return 0;
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 2
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, line:{{[a-z0-9]+}}:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'

int foo2() {
  int array[5];
  #pragma oss task depend(inout: array[2])
  {
    #pragma oss task
    {
      array[2] = 0;
    }
  }
  return 0;
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 2
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, line:{{[a-z0-9]+}}:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
