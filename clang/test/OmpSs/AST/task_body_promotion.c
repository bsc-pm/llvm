// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void bar(int i);
int i;
void foo() {
    int array[10];
    #pragma oss task in(array[i])
    {
        i = 4;
    }
    #pragma oss task in(array[i]) cost(i)
    { }
    #pragma oss task in(array[i]) priority(i)
    { }
    #pragma oss task in(array[i]) onready(bar(i))
    { }
}

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSCostClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSPriorityClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: CallExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'void'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'void (int)' Function 0x{{.*}} 'bar' 'void (int)'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[10]' lvalue Var 0x{{.*}} 'array' 'int[10]'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'

