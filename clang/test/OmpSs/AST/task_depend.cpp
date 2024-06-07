// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
T *foo(T *t) {
    #pragma oss task depend(in: *t)
    {
        #pragma oss task depend(in: t)
        {}
    }
    return t;
}
struct S {
    static int x;
};

template<typename T>
void kk() {
    #pragma oss task depend(in: T::x)
    { T::x++; }
}

void bar()
{
    int *pi;
    int *pi1 = foo(pi);
    kk<S>();
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'T' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'T *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'T *' lvalue ParmVar {{[a-z0-9]+}} 't' 'T *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'T *' lvalue ParmVar {{[a-z0-9]+}} 't' 'T *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DependentScopeDeclRefExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' lvalue

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'
// CHECK: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'


template<typename T>
void foo1(T t) {
  int array[10][20];
  int *p;
  #pragma oss task depend(in: [t]array, [1][2][3]array)
  { }
  #pragma oss task depend(in: [t]p, [1][2][3]p)
  { }
}

void bar1() {
  foo1(1);
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'T' lvalue ParmVar {{[a-z0-9]+}} 't' 'T'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int[1][2][3][20]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:{{.*}}> 'int (*)[20]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 3

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'T' lvalue ParmVar {{[a-z0-9]+}} 't' 'T'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int[1][2][3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 3

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int[t][20]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int (*)[20]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int[1][2][3][20]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int (*)[20]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 3
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int[t]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:{{.*}}, col:{{.*}}> 'int[1][2][3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{.*}}> 'int' 3
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{.*}}> 'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int'

