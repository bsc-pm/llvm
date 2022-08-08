// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

// Verifies that we perform type conversion to 'int'

int main() {
  int v[10];
  int M[10][10];
  short lb, ub, step;
  #pragma oss task in({v[i], i=lb:ub:step})
  {}
  #pragma oss task in({v[i], i=lb + lb;ub + ub:step + step})
  {}
  #pragma oss task in({v[i], i={lb, ub, step}})
  {}
  #pragma oss task in({M[i][j], i={lb, ub, step}, j={lb + ub + step}})
  {}
}

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'lb' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'ub' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'step' 'short'

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'lb' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'lb' 'short'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'ub' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'ub' 'short'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'step' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'step' 'short'

// CHECK: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'v' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'lb' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'ub' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'step' 'short'
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>

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
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'lb' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'ub' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'step' 'short'
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[1]'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'lb' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'ub' 'short'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'short' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'short' lvalue Var 0x{{.*}} 'step' 'short'
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>

