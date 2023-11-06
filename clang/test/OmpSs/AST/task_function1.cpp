// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
struct S {
    #pragma oss task in(*y)
    void foo(T *y);
};

#pragma oss task if(N) final(N) cost(N) priority(N) in(*x) depend(in: *x)
template<int N, typename T>
void foo1(T *x) {}

#pragma oss task in([N]x) depend(in: [N]x)
template<int N, typename T>
void foo2(T *x) {}

int main() {
    int x;
    foo1<1>(&x);
    foo2<2>(&x);
    S<int> s;
}

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Unknown
// CHECK: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<dependent type>' prefix '*' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'y' 'T *'

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Unknown
// CHECK: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'y' 'int *'

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Unknown
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<dependent type>' prefix '*' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<dependent type>' prefix '*' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Unknown
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'bool' <IntegralToBoolean>
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'bool' <IntegralToBoolean>
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 1
// CHECK: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'x' 'int *'
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'x' 'int *'

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Unknown
// CHECK: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK-NEXT: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Unknown
// CHECK: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'x' 'int *'
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 2
// CHECK-NEXT: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'x' 'int *'
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 2
