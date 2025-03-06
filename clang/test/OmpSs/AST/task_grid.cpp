// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) grid(N, 1, 1, N, 1, 1)
template<int N>
void foo();
#pragma oss task device(cuda) grid(1, *x, 1, 1, *x, 1)
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
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' NonTypeTemplateParm 0x{{.*}} 'N' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'T' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'T *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'T' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'T *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'T *' lvalue ParmVar 0x{{.*}} 'x' 'T *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
