// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

struct S {
    #pragma oss task in(*y)
    void foo(int *y);
};

#pragma oss task if(rx)
void foo1(int &rx) {}

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'y' 'int *'

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'bool' <IntegralToBoolean>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.*}} 'rx' 'int &'

