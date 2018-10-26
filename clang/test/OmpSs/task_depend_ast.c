// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int main() {
    int a;
    int b[5];
    int *c;
    #pragma oss task depend(in : a, b[0], *c)
    {}
}

// CHECK: OSSTaskDirective
// CHECK-NEXT: OSSDependClause
// CHECK-NEXT: DeclRefExpr 0x{{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var 0x{{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: ArraySubscriptExpr 0x{{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[a-z0-9]+}} <col:{{[0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[a-z0-9]+}} <col:{{[0-9]+}}> 'int [5]' lvalue Var 0x{{[a-z0-9]+}} 'b' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'c' 'int *
