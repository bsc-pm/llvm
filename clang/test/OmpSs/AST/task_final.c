// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void foo() {
    int x;
    #pragma oss task final(x > 2)
    {}
}
// CHECK: OSSFinalClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: BinaryOperator {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' '>'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 2

