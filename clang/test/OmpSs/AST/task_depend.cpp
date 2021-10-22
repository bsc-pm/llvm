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

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:13, col:36>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:22, col:35>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:33, col:34> '<dependent type>' prefix '*' cannot overflow
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'T *' lvalue ParmVar {{[a-z0-9]+}} 't' 'T *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:8:17, col:39>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:26, col:38>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:37> 'T *' lvalue ParmVar {{[a-z0-9]+}} 't' 'T *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:6:13, col:36>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:22, col:35>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:33, col:34> 'int':'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:34> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:8:17, col:39>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:26, col:38>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:37> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:37> 'int *' lvalue ParmVar {{[a-z0-9]+}} 't' 'int *'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:19:13, col:38>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:22, col:37>
// CHECK-NEXT: DependentScopeDeclRefExpr {{[a-z0-9]+}} <col:33, col:36> '<dependent type>' lvalue

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:19:13, col:38>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:22, col:37>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:33, col:36> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:33, col:36> 'int' lvalue Var {{[a-z0-9]+}} 'x' 'int'


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

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:68:11, col:56>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:20, col:55>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:31, col:34> '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:32> 'T' lvalue ParmVar {{[a-z0-9]+}} 't' 'T'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:41, col:50> 'int[1][2][3][20]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:50> 'int (*)[20]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:50> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:42> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:45> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:48> 'int' 3

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:70:11, col:48>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:20, col:47>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:31, col:34> '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:32> 'T' lvalue ParmVar {{[a-z0-9]+}} 't' 'T'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:37, col:46> 'int[1][2][3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:46> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:46> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:38> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:41> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:44> 'int' 3

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:68:11, col:56>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:20, col:55>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:31, col:34> 'int[t][20]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:34> 'int (*)[20]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:32> 'int':'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:32> 'int':'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int':'int'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:41, col:50> 'int[1][2][3][20]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:50> 'int (*)[20]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:50> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:42> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:45> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:48> 'int' 3
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int[10][20]' lvalue Var {{[a-z0-9]+}} 'array' 'int[10][20]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:32> 'int':'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int':'int'

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:70:11, col:48>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:20, col:47>
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:31, col:34> 'int[t]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:34> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:32> 'int':'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:32> 'int':'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int':'int'
// CHECK-NEXT: OSSArrayShapingExpr {{[a-z0-9]+}} <col:37, col:46> 'int[1][2][3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:46> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:46> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:38> 'int' 1
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:41> 'int' 2
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:44> 'int' 3
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:34> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:32> 'int':'int' lvalue ParmVar {{[a-z0-9]+}} 't' 'int':'int'

