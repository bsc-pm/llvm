// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -Wno-vla -std=c++11 %s
// expected-no-diagnostics

struct S {
    int x;
    void foo() {
        #pragma oss task onready(x)
        {}
    }
};

template<typename T> T foo() { return T(); }

#pragma oss task onready(vla[3])
void foo1(int n, int *vla[n]) {}

#pragma oss task onready(foo<int *>())
void foo2() {}

void bar(int n) {
    S s;
    s.foo();
    n = -1;
    const int m = -1;
    int *vla[n];
    #pragma oss task onready(vla[3])
    {}
    #pragma oss task onready(foo<int>())
    {}
    #pragma oss task onready(n)
    {}
    #pragma oss task onready(m)
    {}
    #pragma oss task onready(-1)
    {}
    #pragma oss task onready([&n]() {})
    {}
}

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: MemberExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue ->x 0x{{.*}}
// CHECK-NEXT: CXXThisExpr 0x{{.*}} <col:{{.*}}> 'S *' implicit this
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{.*}} <col:{{.*}}> 'S *' implicit this

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int **' lvalue ParmVar 0x{{.*}} 'vla' 'int **'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 3

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: CallExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *(*)()' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *()' lvalue Function 0x{{.*}} 'foo' 'int *()' (FunctionTemplate 0x{{.*}} 'foo')

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ArraySubscriptExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int **' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *[n]' lvalue Var 0x{{.*}} 'vla' 'int *[n]'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 3
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *[n]' lvalue Var 0x{{.*}} 'vla' 'int *[n]'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: CallExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int (*)()' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int ()' lvalue Function 0x{{.*}} 'foo' 'int ()' (FunctionTemplate 0x{{.*}} 'foo')

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.*}} 'n' 'int'
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.*}} 'n' 'int'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'const int' lvalue Var 0x{{.*}} 'm' 'const int'
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'const int' lvalue Var 0x{{.*}} 'm' 'const int'

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' prefix '-'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSOnreadyClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: LambdaExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> '(lambda at task_onready.cpp:40:30)'
// CHECK: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.*}} 'n' 'int'

