// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss task cost(foo<int>())
    {}
    #pragma oss task cost(n)
    {}
    #pragma oss task cost(vla[1])
    {}
}

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSCostClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: CallExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int':'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int (*)()' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int ()' lvalue Function 0x{{.*}} 'foo' 'int ()' (FunctionTemplate 0x{{.*}} 'foo')

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSCostClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.*}} 'n' 'int'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSCostClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[n]' lvalue Var 0x{{.*}} 'vla' 'int[n]'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

