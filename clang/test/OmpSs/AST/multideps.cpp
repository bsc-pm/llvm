// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int array[10];

template<typename T>
void foo() {
    #pragma oss task in( { array[i], i=0;10 } )
    {}
}

int main() {
    foo<int>();
}

// template
// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 10
// CHECK-NEXT: <<<NULL>>>

// instantiation
// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: OSSMultiDepExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 10
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
