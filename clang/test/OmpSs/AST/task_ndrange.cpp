// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) ndrange(N, 1, 1)
template<int N>
void foo();
#pragma oss task device(opencl) ndrange(1, *x, 1)
template<typename T>
void foo1(T *x);

void bad() {
    int x;
    foo<1>();
    foo1(&x);
}

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<dependent type>' prefix '*' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'x' 'int *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
