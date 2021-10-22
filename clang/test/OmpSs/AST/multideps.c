// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int main() {
  int v[10];
  int M[10][10];
  #pragma oss task in({v[i], i=0:10:1})
  {}
  #pragma oss task in({v[i], i=0;10:1})
  {}
  #pragma oss task in({v[i], i={0, 1, 2}})
  {}
  #pragma oss task in({M[i][j], i={0, 1, 2}, j={0, 1, 2}})
  {}
}

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 10
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 10
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[10]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int (*)[10]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10][10]' lvalue Var 0x{{.*}} 'M' 'int[10][10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'j' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'j' 'int'
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10][10]' lvalue Var 0x{{.*}} 'M' 'int[10][10]'
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'j' 'int'

